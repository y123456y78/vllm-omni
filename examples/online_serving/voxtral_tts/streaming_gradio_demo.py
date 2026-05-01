"""Streaming-input Gradio demo for Voxtral TTS.

Voice-clone (audio-prompted) text-to-speech where the user uploads (or records)
a reference audio sample, types text, and receives per-chunk audio
progressively as the server splits the text on sentence/clause boundaries and
generates each chunk.

Backed by the WebSocket endpoint at /v1/audio/speech/stream
(vllm_omni/entrypoints/openai/serving_speech_stream.py). The server splits
text on sentence (or clause) boundaries; per-sentence generation uses fresh
prefill — there is no KV-cache continuation across chunks. If you need true
fine_cut continuation (mistral-2 tts_demo.py "Streaming TTS (fine_cut)"),
expect minor discontinuities at chunk boundaries here.

Install:
    pip install -e .
    pip install gradio==5.50 mistral_common==1.10.0 websockets

Run:
    python examples/online_serving/voxtral_tts/streaming_gradio_demo.py \
        --host slurm-199-077 --port 8000
"""

import argparse
import json
import logging
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required. Install with: pip install 'vllm-omni[demo]'") from None
try:
    import websockets
except ImportError:
    raise ImportError("websockets is required. Install with: pip install websockets") from None

import httpx
import numpy as np
import soundfile as sf
from mistral_common.tokens.tokenizers.audio import Audio
from text_preprocess import sanitize_tts_input_text_for_demo

logger = logging.getLogger()

LOGFORMAT = "%(asctime)s - %(levelname)s - %(message)s"
TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, force=True, format=LOGFORMAT, datefmt=TIMEFORMAT)
logger.setLevel(logging.INFO)


_MAX_CHUNKS = 16
_VOXTRAL_SAMPLE_RATE = 24_000

_SERVER_CHECK_TIMEOUT = 300.0
_SERVER_CHECK_INTERVAL = 5.0


def wait_for_server(base_url: str, timeout: float = _SERVER_CHECK_TIMEOUT) -> bool:
    """Block until the HTTP health endpoint responds, or timeout."""
    start_time = time.time()
    health_url = base_url.replace("/v1", "") + "/health"
    logger.info(f"Waiting for server at {base_url} to become available...")
    with httpx.Client(timeout=5.0) as client:
        while time.time() - start_time < timeout:
            try:
                if client.get(health_url).status_code == 200:
                    logger.info("Server is now available!")
                    return True
            except Exception:
                pass
            elapsed = time.time() - start_time
            logger.info(f"Server not yet available ({elapsed:.1f}s elapsed), retrying in {_SERVER_CHECK_INTERVAL}s...")
            time.sleep(_SERVER_CHECK_INTERVAL)
    logger.warning(f"Server did not become available within {timeout}s timeout")
    return False


def _load_voice_sample_as_base64(audio_path: str) -> str:
    """Load uploaded audio, downmix to mono, resample to 24 kHz, base64 WAV.

    Mirrors the proven pattern from mistral-2 tts_demo.py so the resulting
    waveform exactly matches what the model was trained against.
    """
    audio_array, sr = sf.read(audio_path)
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    audio = Audio(audio_array=audio_array, sampling_rate=sr)
    audio.resample(_VOXTRAL_SAMPLE_RATE)
    return audio.to_base64("wav")


def _empty_chunk_updates() -> list[Any]:
    """Build a flat list of (row, label, audio) updates that hides every slot."""
    updates: list[Any] = []
    for i in range(_MAX_CHUNKS):
        updates.append(gr.update(visible=False))
        updates.append(gr.update(value=f"**Chunk {i + 1}**"))
        updates.append(None)
    return updates


def _make_outputs(
    full_audio: str | None,
    status: str,
    chunk_updates: list[Any],
) -> tuple[Any, ...]:
    return (full_audio, status, *chunk_updates)


async def _stream_inference(
    voice_sample_path: str | None,
    text: str,
    cfg_alpha: float,
    max_new_tokens: int,
    split_granularity: str,
    ws_url: str,
    model: str,
    outputs_dir: Path,
) -> AsyncGenerator[tuple[Any, ...], None]:
    """Async generator that yields Gradio output tuples as chunks arrive."""
    if not voice_sample_path:
        raise gr.Error("Please upload or record a voice sample.")
    user_text = (text or "").strip()
    if not user_text:
        raise gr.Error("Please enter a text prompt.")

    try:
        text = sanitize_tts_input_text_for_demo(user_text)
    except Exception as exc:
        raise gr.Error(f"Text preprocessing failed: {exc}") from exc

    try:
        ref_audio_b64 = _load_voice_sample_as_base64(voice_sample_path)
    except Exception as exc:
        logger.exception("Failed to load voice sample")
        raise gr.Error(f"Failed to load voice sample: {exc}") from exc

    session_id = uuid.uuid4().hex
    chunk_updates = _empty_chunk_updates()
    yield _make_outputs(None, "Connecting to server...", chunk_updates)

    config_msg = {
        "type": "session.config",
        "model": model,
        "response_format": "wav",
        "ref_audio": ref_audio_b64,
        "max_new_tokens": int(max_new_tokens),
        "split_granularity": split_granularity,
        "extra_params": {"cfg_alpha": float(cfg_alpha)},
    }

    chunk_paths: list[Path] = []
    accum: list[bytes] = []
    current_index = 0
    current_text = ""
    error_message: str | None = None

    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            await ws.send(json.dumps(config_msg))
            await ws.send(json.dumps({"type": "input.text", "text": text}))
            await ws.send(json.dumps({"type": "input.done"}))
            yield _make_outputs(None, "Generating chunks...", chunk_updates)

            while True:
                message = await ws.recv()
                if isinstance(message, (bytes, bytearray)):
                    accum.append(bytes(message))
                    continue

                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "audio.start":
                    current_index = int(msg.get("sentence_index", current_index))
                    current_text = msg.get("sentence_text", "")
                    accum = []

                elif msg_type == "audio.done":
                    if msg.get("error"):
                        error_message = f"Server error on chunk {current_index}"
                        break
                    if current_index >= _MAX_CHUNKS:
                        logger.warning(
                            "Received chunk %d but only %d slots are pre-allocated; ignoring.",
                            current_index,
                            _MAX_CHUNKS,
                        )
                        accum = []
                        continue
                    chunk_path = outputs_dir / f"{session_id}_chunk_{current_index:03d}.wav"
                    chunk_path.write_bytes(b"".join(accum))
                    chunk_paths.append(chunk_path)
                    base = current_index * 3
                    chunk_updates[base] = gr.update(visible=True)
                    chunk_updates[base + 1] = gr.update(
                        value=f"**Chunk {current_index + 1}**: {current_text}"
                    )
                    chunk_updates[base + 2] = str(chunk_path)
                    status = f"Generated chunk {current_index + 1}"
                    accum = []
                    yield _make_outputs(None, status, chunk_updates)

                elif msg_type == "session.done":
                    break

                elif msg_type == "error":
                    error_message = msg.get("message", "unknown server error")
                    break

    except websockets.exceptions.WebSocketException as exc:
        raise gr.Error(f"WebSocket error: {exc}") from exc

    if error_message:
        raise gr.Error(error_message)

    if not chunk_paths:
        raise gr.Error("Server returned no audio chunks.")

    full_audio_path = outputs_dir / f"{session_id}_full.wav"
    pieces: list[np.ndarray] = []
    sample_rate = _VOXTRAL_SAMPLE_RATE
    for path in chunk_paths:
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        pieces.append(data)
        sample_rate = sr
    full = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    sf.write(str(full_audio_path), full, sample_rate)

    yield _make_outputs(
        str(full_audio_path),
        f"Done. {len(chunk_paths)} chunk(s) generated.",
        chunk_updates,
    )


def main(
    model: str,
    host: str,
    port: str,
    output_dir: str | None = None,
    ws_path: str = "/v1/audio/speech/stream",
) -> None:
    base_url = f"http://{host}:{port}/v1"
    ws_url = f"ws://{host}:{port}{ws_path}"
    logger.info(f"Streaming demo using WebSocket: {ws_url}")

    if not wait_for_server(base_url):
        logger.warning("Server unavailable; demo will surface errors on first request.")

    if output_dir is not None:
        outputs_dir = Path(output_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
    else:
        outputs_dir = Path(tempfile.mkdtemp(prefix="voxtral_streaming_demo_"))
        logger.info(f"No --output-dir provided; using temp dir: {outputs_dir}")

    with gr.Blocks(title="Voxtral TTS - Streaming", fill_height=True) as demo:
        gr.Markdown("## Voxtral TTS - Streaming (audio-prompted, voice clone)")
        gr.Markdown(
            "Upload or record a short voice sample (1-30s), enter text, and receive per-chunk audio as the server "
            "splits the text on sentence (or clause) boundaries. **Note**: each chunk is generated with a fresh "
            "prefill - this is not the KV-cache fine_cut continuation from `tts_demo.py`."
        )

        with gr.Row():
            with gr.Column():
                voice_sample = gr.Audio(
                    label="Voice sample (required)",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                text_prompt = gr.Textbox(
                    label="Text prompt",
                    placeholder="Enter the text you want to synthesize...",
                    lines=4,
                )
                cfg_alpha_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.2,
                    label="CFG Alpha",
                    info="Flow-matching guidance strength (default: 1.2)",
                )
                max_new_tokens_slider = gr.Slider(
                    minimum=128,
                    maximum=8192,
                    step=128,
                    value=2048,
                    label="Max new tokens (per chunk)",
                )
                split_granularity_radio = gr.Radio(
                    choices=["sentence", "clause"],
                    value="sentence",
                    label="Split granularity",
                    info="`clause` also splits on commas/semicolons (incl. CJK)",
                )
                with gr.Row():
                    reset_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Generate streaming audio", interactive=False)

            with gr.Column():
                output_audio = gr.Audio(
                    label="Full audio (concatenated)",
                    show_download_button=True,
                    interactive=False,
                    autoplay=False,
                    type="filepath",
                )
                status_box = gr.Markdown("")

                gr.Markdown("### Per-chunk audio")
                chunk_rows: list[gr.Row] = []
                chunk_labels: list[gr.Markdown] = []
                chunk_audios: list[gr.Audio] = []
                for i in range(_MAX_CHUNKS):
                    with gr.Row(visible=False) as chunk_row:
                        chunk_label = gr.Markdown(f"**Chunk {i + 1}**")
                        chunk_audio = gr.Audio(
                            label=f"Chunk {i + 1}",
                            interactive=False,
                            type="filepath",
                            autoplay=False,
                        )
                    chunk_rows.append(chunk_row)
                    chunk_labels.append(chunk_label)
                    chunk_audios.append(chunk_audio)

        outputs_list: list[Any] = [output_audio, status_box]
        for i in range(_MAX_CHUNKS):
            outputs_list.append(chunk_rows[i])
            outputs_list.append(chunk_labels[i])
            outputs_list.append(chunk_audios[i])

        def _toggle_submit(voice_path: str | None, text: str):
            enabled = bool(voice_path) and bool((text or "").strip())
            return gr.update(interactive=enabled)

        voice_sample.change(
            fn=_toggle_submit,
            inputs=[voice_sample, text_prompt],
            outputs=submit_btn,
        )
        text_prompt.change(
            fn=_toggle_submit,
            inputs=[voice_sample, text_prompt],
            outputs=submit_btn,
        )

        async def _on_submit(
            voice_path: str | None,
            text: str,
            cfg_alpha: float,
            max_new_tokens: int,
            split_granularity: str,
        ) -> AsyncGenerator[tuple[Any, ...], None]:
            async for update in _stream_inference(
                voice_sample_path=voice_path,
                text=text,
                cfg_alpha=cfg_alpha,
                max_new_tokens=max_new_tokens,
                split_granularity=split_granularity,
                ws_url=ws_url,
                model=model,
                outputs_dir=outputs_dir,
            ):
                yield update

        submit_btn.click(
            fn=_on_submit,
            inputs=[
                voice_sample,
                text_prompt,
                cfg_alpha_slider,
                max_new_tokens_slider,
                split_granularity_radio,
            ],
            outputs=outputs_list,
        )

        def _on_reset() -> tuple[Any, ...]:
            base: list[Any] = [
                None,                          # voice_sample
                "",                            # text_prompt
                1.2,                           # cfg_alpha_slider
                2048,                          # max_new_tokens_slider
                "sentence",                    # split_granularity_radio
                gr.update(interactive=False),  # submit_btn
                None,                          # output_audio
                "",                            # status_box
            ]
            base.extend(_empty_chunk_updates())
            return tuple(base)

        reset_outputs: list[Any] = [
            voice_sample,
            text_prompt,
            cfg_alpha_slider,
            max_new_tokens_slider,
            split_granularity_radio,
            submit_btn,
            output_audio,
            status_box,
        ]
        for i in range(_MAX_CHUNKS):
            reset_outputs.append(chunk_rows[i])
            reset_outputs.append(chunk_labels[i])
            reset_outputs.append(chunk_audios[i])

        reset_btn.click(
            fn=_on_reset,
            inputs=[],
            outputs=reset_outputs,
        )

    launch_kwargs: dict[str, Any] = {
        "server_name": "0.0.0.0",
        "share": True,
        "allowed_paths": [str(outputs_dir)],
    }
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxtral TTS streaming-input Gradio demo")
    parser.add_argument("--model", type=str, default="mistralai/Voxtral-4B-TTS-2603", help="Model name on HF / served name")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="8091")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for per-chunk and full-audio WAVs. If unset, a temp dir is used.",
    )
    parser.add_argument(
        "--ws-path",
        type=str,
        default="/v1/audio/speech/stream",
        help="WebSocket path on the server.",
    )
    args = parser.parse_args()

    main(
        model=args.model,
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        ws_path=args.ws_path,
    )
