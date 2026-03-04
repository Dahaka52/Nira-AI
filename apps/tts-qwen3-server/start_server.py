#!/usr/bin/env python3

import argparse
import asyncio
import contextlib
import io
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


def _to_pcm16_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)

    arr = np.asarray(chunk)
    if arr.size == 0:
        return b""

    if arr.dtype == np.int16:
        return arr.tobytes()

    if arr.dtype.kind == "f":
        arr = np.clip(arr.astype(np.float32), -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()

    return arr.astype(np.int16).tobytes()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    instruct: Optional[str] = None
    speaker: Optional[str] = None
    language: str = "Auto"
    provider: Optional[str] = None
    gpu_id: Optional[int] = None
    model_id: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    sample_rate: Optional[int] = None
    sample_width: Optional[int] = None
    channels: Optional[int] = None
    max_new_tokens: Optional[int] = None
    emit_every_frames: Optional[int] = None
    decode_window_frames: Optional[int] = None
    first_chunk_emit_every: Optional[int] = None
    first_chunk_decode_window: Optional[int] = None
    first_chunk_frames: Optional[int] = None
    overlap_samples: Optional[int] = None
    use_optimized_decode: Optional[bool] = None
    repetition_penalty: Optional[float] = None
    repetition_penalty_window: Optional[int] = None
    do_sample: Optional[bool] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6116
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    speaker: str = "Ryan"
    language: str = "Auto"
    provider: str = "cuda"
    gpu_id: int = 1
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    sample_rate: int = 24000
    sample_width: int = 2
    channels: int = 1
    emit_every_frames: int = 12
    decode_window_frames: int = 80
    first_chunk_emit_every: int = 5
    first_chunk_decode_window: int = 48
    first_chunk_frames: int = 48
    overlap_samples: int = 512
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.0
    repetition_penalty_window: int = 100
    max_concurrent: int = 1
    instruct_prefix: str = ""
    use_compile: bool = False
    preload_on_start: bool = True
    warmup_on_start: bool = True
    warmup_text: str = "Привет."
    warmup_emit_every_frames: int = 4
    warmup_decode_window_frames: int = 64
    warmup_max_new_tokens: int = 96
    warmup_speaker: str = ""
    warmup_language: str = ""
    warmup_chunks: int = 2


class Qwen3Runtime:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self._model = None
        self._model_id_loaded = None
        self._attn_impl_loaded = None
        self._lock = threading.Lock()
        self._import_error = None
        self._load_error = None

        try:
            with io.StringIO() as _stdout, io.StringIO() as _stderr:
                with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
                    import qwen_tts  # noqa: F401
                captured = (_stdout.getvalue() + _stderr.getvalue()).strip()
            if captured:
                lower = captured.lower()
                if "flash-attn is not installed" in lower:
                    logging.warning(
                        "flash-attn is unavailable on this runtime; using attn_implementation=%s",
                        self.cfg.attn_implementation or "default",
                    )
                elif "sox could not be found" in lower:
                    logging.warning("SoX binary not found in PATH for qwen_tts runtime.")
                else:
                    logging.info("qwen_tts import notes: %s", captured.replace("\n", " ")[:400])
        except Exception as exc:
            self._import_error = exc
            logging.error("Qwen3 runtime unavailable: %s", exc)

    def is_ready(self) -> bool:
        return self._import_error is None and self._load_error is None

    def status_payload(self) -> Dict[str, Any]:
        payload = {
            "ok": True,
            "provider": "qwen3",
            "ready": self.is_ready(),
            "model_loaded": bool(self._model is not None),
            "model_id_loaded": self._model_id_loaded,
            "runtime_provider": self.cfg.provider,
            "runtime_gpu_id": self.cfg.gpu_id,
            "runtime_device": self.cfg.device,
            "runtime_attn_implementation": self.cfg.attn_implementation,
        }
        if self._import_error is not None:
            payload["detail"] = f"qwen_tts import failed: {self._import_error}"
        if self._load_error is not None:
            payload["detail"] = f"model load failed: {self._load_error}"
        return payload

    def _resolve_dtype(self, torch_mod, dtype_name: str):
        name = str(dtype_name or "bfloat16").strip().lower()
        if name in ("float16", "fp16"):
            return torch_mod.float16
        if name in ("float32", "fp32"):
            return torch_mod.float32
        return torch_mod.bfloat16

    def _ensure_model_loaded(
        self,
        model_id: str,
        device: str,
        dtype: str,
        decode_window_frames: int,
        attn_implementation: str | None,
    ) -> None:
        if self._import_error is not None:
            raise RuntimeError(f"qwen_tts import failed: {self._import_error}")

        with self._lock:
            if (
                self._model is not None
                and self._model_id_loaded == model_id
                and self._attn_impl_loaded == (attn_implementation or None)
            ):
                return

            try:
                import torch
                from qwen_tts import Qwen3TTSModel

                torch_dtype = self._resolve_dtype(torch, dtype)
                use_compile = bool(getattr(self.cfg, "use_compile", False))
                if use_compile:
                    try:
                        import importlib.util

                        if importlib.util.find_spec("triton") is None:
                            logging.warning(
                                "Qwen3 runtime: use_compile=1 requested, but triton is missing. Falling back to use_compile=0."
                            )
                            use_compile = False
                    except Exception:
                        use_compile = False

                # Better defaults for tensor core throughput on RTX 50xx.
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

                model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    attn_implementation=(attn_implementation or None),
                )

                if hasattr(model, "enable_streaming_optimizations"):
                    try:
                        if use_compile:
                            logging.info("Qwen3 runtime: enabling compile-based streaming optimizations.")
                        else:
                            logging.info("Qwen3 runtime: compile optimizations are disabled (use_compile=false).")
                        model.enable_streaming_optimizations(
                            decode_window_frames=int(decode_window_frames),
                            use_compile=use_compile,
                            compile_mode="reduce-overhead",
                        )
                    except Exception as exc:
                        logging.warning("enable_streaming_optimizations failed: %s", exc)

                self._model = model
                self._model_id_loaded = model_id
                self._attn_impl_loaded = attn_implementation or None
                self._load_error = None
                logging.info("Qwen3 model loaded: %s (attn=%s)", model_id, self._attn_impl_loaded or "default")
            except Exception as exc:
                self._load_error = exc
                self._model = None
                self._model_id_loaded = None
                self._attn_impl_loaded = None
                raise

    def preload_and_warmup(self) -> None:
        if not bool(getattr(self.cfg, "preload_on_start", True)):
            return

        cfg = self.cfg
        provider = str(cfg.provider or "cuda").strip().lower()
        if provider.startswith("cpu"):
            device = "cpu"
        else:
            device = str(cfg.device or "cuda:0").strip() or "cuda:0"

        self._ensure_model_loaded(
            model_id=str(cfg.model_id),
            device=device,
            dtype=str(cfg.dtype),
            decode_window_frames=int(cfg.decode_window_frames),
            attn_implementation=str(cfg.attn_implementation or "").strip() or None,
        )

        if not bool(getattr(cfg, "warmup_on_start", True)):
            return

        warmup_text = str(getattr(cfg, "warmup_text", "") or "").strip()
        if not warmup_text:
            return

        req = TTSRequest(
            text=warmup_text,
            speaker=(str(getattr(cfg, "warmup_speaker", "") or "").strip() or cfg.speaker),
            language=(str(getattr(cfg, "warmup_language", "") or "").strip() or cfg.language),
            provider=cfg.provider,
            gpu_id=cfg.gpu_id,
            model_id=cfg.model_id,
            device=cfg.device,
            dtype=cfg.dtype,
            attn_implementation=cfg.attn_implementation,
            emit_every_frames=int(getattr(cfg, "warmup_emit_every_frames", 4)),
            decode_window_frames=int(getattr(cfg, "warmup_decode_window_frames", 64)),
            max_new_tokens=int(getattr(cfg, "warmup_max_new_tokens", 96)),
            overlap_samples=int(getattr(cfg, "overlap_samples", 0)),
            do_sample=False,
            top_p=0.9,
            top_k=50,
            temperature=0.8,
        )

        started = time.perf_counter()
        chunks = 0
        samples = 0
        max_chunks = max(1, int(getattr(cfg, "warmup_chunks", 2)))
        try:
            for pcm_bytes, sr in self.stream_custom_voice(req):
                chunks += 1
                if pcm_bytes:
                    samples += int(len(pcm_bytes) // 2)  # mono int16
                if chunks >= max_chunks:
                    break
        except Exception as exc:
            logging.warning("Qwen3 runtime warmup failed: %s", exc)
            return

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        audio_ms = int((samples / max(1, int(sr))) * 1000) if samples > 0 else 0
        logging.info(
            "Qwen3 runtime warmup done: chunks=%s elapsed_ms=%s audio_ms=%s",
            chunks,
            elapsed_ms,
            audio_ms,
        )

    def stream_custom_voice(self, req: TTSRequest) -> Iterable[tuple[bytes, int]]:
        cfg = self.cfg
        model_id = req.model_id or cfg.model_id
        provider = str(req.provider or cfg.provider or "cuda").strip().lower()
        gpu_id = int(req.gpu_id if req.gpu_id is not None else cfg.gpu_id)
        if req.device and str(req.device).strip():
            device = str(req.device).strip()
        else:
            if provider.startswith("cpu"):
                device = "cpu"
            else:
                device = cfg.device or f"cuda:{gpu_id}"
        dtype = req.dtype or cfg.dtype
        attn_implementation = req.attn_implementation or cfg.attn_implementation
        decode_window_frames = int(req.decode_window_frames or cfg.decode_window_frames)
        emit_every_frames = int(req.emit_every_frames or cfg.emit_every_frames)
        first_chunk_emit_every = int(req.first_chunk_emit_every or cfg.first_chunk_emit_every)
        first_chunk_decode_window = int(req.first_chunk_decode_window or cfg.first_chunk_decode_window)
        first_chunk_frames = int(req.first_chunk_frames or cfg.first_chunk_frames)
        overlap_samples = int(req.overlap_samples or cfg.overlap_samples)
        use_optimized_decode = bool(req.use_optimized_decode) if req.use_optimized_decode is not None else True
        repetition_penalty = float(req.repetition_penalty or cfg.repetition_penalty)
        repetition_penalty_window = int(req.repetition_penalty_window or cfg.repetition_penalty_window)
        max_new_tokens = int(req.max_new_tokens or cfg.max_new_tokens)
        speaker = req.speaker or cfg.speaker
        language = req.language or cfg.language
        instruct = req.instruct or ""
        if cfg.instruct_prefix:
            instruct = f"{cfg.instruct_prefix} {instruct}".strip()

        self._ensure_model_loaded(
            model_id=model_id,
            device=device,
            dtype=dtype,
            decode_window_frames=decode_window_frames,
            attn_implementation=attn_implementation,
        )

        stream_kwargs = {
            "text": req.text,
            "speaker": speaker,
            "language": language,
            "emit_every_frames": emit_every_frames,
            "decode_window_frames": decode_window_frames,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "repetition_penalty_window": repetition_penalty_window,
            "first_chunk_emit_every": first_chunk_emit_every,
            "first_chunk_decode_window": first_chunk_decode_window,
            "first_chunk_frames": first_chunk_frames,
            "overlap_samples": overlap_samples,
            "use_optimized_decode": use_optimized_decode,
        }
        if req.do_sample is not None:
            stream_kwargs["do_sample"] = bool(req.do_sample)
            if req.top_p is not None:
                stream_kwargs["top_p"] = float(req.top_p)
            if req.top_k is not None:
                stream_kwargs["top_k"] = int(req.top_k)
            if req.temperature is not None:
                stream_kwargs["temperature"] = float(req.temperature)
        if instruct:
            stream_kwargs["instruct"] = instruct

        stream_fn = getattr(self._model, "stream_generate_custom_voice", None)
        if stream_fn is None:
            # Compatibility fallback for official qwen_tts builds that only expose
            # generate_custom_voice() (non-streaming API).
            gen_fn = getattr(self._model, "generate_custom_voice", None)
            if gen_fn is None:
                raise RuntimeError(
                    "Loaded qwen_tts model does not support stream_generate_custom_voice() "
                    "or generate_custom_voice()."
                )

            logging.warning(
                "Qwen3 runtime fallback: stream_generate_custom_voice() is unavailable. "
                "Using generate_custom_voice() chunked fallback."
            )
            gen_kwargs = {
                "text": req.text,
                "speaker": speaker,
                "language": language,
                "max_new_tokens": max_new_tokens,
            }
            if instruct:
                gen_kwargs["instruct"] = instruct
            # keep only conservative generation kwargs supported across builds
            gen_kwargs.update({
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.8,
            })
            wavs, sr = gen_fn(**gen_kwargs)
            if not wavs:
                return

            pcm_all = _to_pcm16_bytes(wavs[0])
            if not pcm_all:
                return

            sr_i = int(sr)
            chunk_ms = 40
            chunk_samples = max(1, int(sr_i * (chunk_ms / 1000.0)))
            chunk_bytes = chunk_samples * 2  # mono int16
            for off in range(0, len(pcm_all), chunk_bytes):
                part = pcm_all[off:off + chunk_bytes]
                if part:
                    yield part, sr_i
            return

        try:
            iterator = stream_fn(**stream_kwargs)
        except TypeError:
            # Fallback for older forks without first-chunk / repetition parameters.
            for key in (
                "first_chunk_emit_every",
                "first_chunk_decode_window",
                "first_chunk_frames",
                "repetition_penalty",
                "repetition_penalty_window",
            ):
                stream_kwargs.pop(key, None)
            iterator = stream_fn(**stream_kwargs)

        for chunk, sr in iterator:
            pcm = _to_pcm16_bytes(chunk)
            if pcm:
                yield pcm, int(sr)


def build_app(cfg: ServerConfig) -> FastAPI:
    app = FastAPI(title="Qwen3 TTS Sidecar", version="0.1.0")
    runtime = Qwen3Runtime(cfg)
    semaphore = asyncio.Semaphore(max(1, int(cfg.max_concurrent)))

    @app.on_event("startup")
    async def _startup_preload():
        await asyncio.to_thread(runtime.preload_and_warmup)

    @app.get("/health")
    async def health():
        return JSONResponse(runtime.status_payload())

    @app.post("/v1/tts/stream")
    async def tts_stream(req: TTSRequest):
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text must be non-empty")
        if req.sample_width and int(req.sample_width) != 2:
            raise HTTPException(status_code=400, detail="Only PCM16 stream is supported (sample_width=2).")
        if req.channels and int(req.channels) != 1:
            raise HTTPException(status_code=400, detail="Only mono stream is supported (channels=1).")

        if not runtime.is_ready() and runtime._import_error is not None:
            raise HTTPException(status_code=503, detail=f"Runtime unavailable: {runtime._import_error}")

        async with semaphore:
            async def stream_bytes():
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                sentinel = object()
                started = time.perf_counter()

                def worker():
                    try:
                        for pcm_bytes, _sr in runtime.stream_custom_voice(req):
                            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", pcm_bytes))
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, (sentinel, None))

                threading.Thread(target=worker, daemon=True).start()

                first = True
                while True:
                    tag, payload = await queue.get()
                    if tag is sentinel:
                        break
                    if tag == "error":
                        logging.error("Qwen3 sidecar stream worker failed: %s", payload)
                        break
                    if first:
                        first = False
                        logging.info("Qwen3 sidecar first-audio latency_ms=%s", int((time.perf_counter() - started) * 1000))
                    if payload:
                        yield payload

            try:
                headers = {
                    "x-sample-rate": str(int(req.sample_rate or cfg.sample_rate)),
                    "x-sample-width": str(cfg.sample_width),
                    "x-channels": str(cfg.channels),
                }
                return StreamingResponse(stream_bytes(), media_type="application/octet-stream", headers=headers)
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6116)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--speaker", type=str, default="Ryan")
    parser.add_argument("--language", type=str, default="Auto")
    parser.add_argument("--provider", type=str, default="cuda", help="cpu|cuda")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU id for CUDA mode")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", help="eager|sdpa|flash_attention_2")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--sample_width", type=int, default=2)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--emit_every_frames", type=int, default=12)
    parser.add_argument("--decode_window_frames", type=int, default=80)
    parser.add_argument("--first_chunk_emit_every", type=int, default=5)
    parser.add_argument("--first_chunk_decode_window", type=int, default=48)
    parser.add_argument("--first_chunk_frames", type=int, default=48)
    parser.add_argument("--overlap_samples", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty_window", type=int, default=100)
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--instruct_prefix", type=str, default="")
    parser.add_argument("--use_compile", type=int, default=0)
    parser.add_argument("--preload_on_start", type=int, default=1)
    parser.add_argument("--warmup_on_start", type=int, default=1)
    parser.add_argument("--warmup_text", type=str, default="Привет.")
    parser.add_argument("--warmup_emit_every_frames", type=int, default=4)
    parser.add_argument("--warmup_decode_window_frames", type=int, default=64)
    parser.add_argument("--warmup_max_new_tokens", type=int, default=96)
    parser.add_argument("--warmup_speaker", type=str, default="")
    parser.add_argument("--warmup_language", type=str, default="")
    parser.add_argument("--warmup_chunks", type=int, default=2)
    ns = parser.parse_args()
    return ServerConfig(
        host=ns.host,
        port=ns.port,
        model_id=ns.model_id,
        speaker=ns.speaker,
        language=ns.language,
        provider=ns.provider,
        gpu_id=ns.gpu_id,
        device=ns.device,
        dtype=ns.dtype,
        attn_implementation=ns.attn_implementation,
        sample_rate=ns.sample_rate,
        sample_width=ns.sample_width,
        channels=ns.channels,
        emit_every_frames=ns.emit_every_frames,
        decode_window_frames=ns.decode_window_frames,
        first_chunk_emit_every=ns.first_chunk_emit_every,
        first_chunk_decode_window=ns.first_chunk_decode_window,
        first_chunk_frames=ns.first_chunk_frames,
        overlap_samples=ns.overlap_samples,
        max_new_tokens=ns.max_new_tokens,
        repetition_penalty=ns.repetition_penalty,
        repetition_penalty_window=ns.repetition_penalty_window,
        max_concurrent=ns.max_concurrent,
        instruct_prefix=ns.instruct_prefix,
        use_compile=bool(int(ns.use_compile)),
        preload_on_start=bool(int(ns.preload_on_start)),
        warmup_on_start=bool(int(ns.warmup_on_start)),
        warmup_text=ns.warmup_text,
        warmup_emit_every_frames=ns.warmup_emit_every_frames,
        warmup_decode_window_frames=ns.warmup_decode_window_frames,
        warmup_max_new_tokens=ns.warmup_max_new_tokens,
        warmup_speaker=ns.warmup_speaker,
        warmup_language=ns.warmup_language,
        warmup_chunks=ns.warmup_chunks,
    )


def main():
    cfg = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [qwen3_tts_sidecar] %(message)s",
    )
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
