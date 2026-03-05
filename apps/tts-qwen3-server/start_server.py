#!/usr/bin/env python3

import argparse
import asyncio
import concurrent.futures
import logging
import os
import sys
import threading
import time
import traceback
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
    model_config = {"extra": "ignore"}

    text: str = Field(..., min_length=1)
    language: Optional[str] = None

    provider: Optional[str] = None
    gpu_id: Optional[int] = None
    model_id: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None

    sample_rate: Optional[int] = None
    sample_width: Optional[int] = None
    channels: Optional[int] = None

    runtime_flavor: Optional[str] = None
    voice_mode: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: Optional[bool] = None

    emit_every_frames: Optional[int] = None
    decode_window_frames: Optional[int] = None
    overlap_samples: Optional[int] = None
    max_frames: Optional[int] = None
    use_optimized_decode: Optional[bool] = None

    do_sample: Optional[bool] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6116

    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    language: str = "russian"
    provider: str = "cuda"
    gpu_id: int = 1
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    sample_rate: int = 24000
    sample_width: int = 2
    channels: int = 1

    runtime_flavor: str = "dffdeeq"
    voice_mode: str = "voice_clone"
    qwen_tts_repo_path: str = ""

    ref_audio_path: str = ""
    ref_text: str = ""
    x_vector_only_mode: bool = True

    emit_every_frames: int = 4
    decode_window_frames: int = 80
    overlap_samples: int = 0
    max_frames: int = 10000
    use_optimized_decode: bool = True

    do_sample: bool = False
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 0.8

    dynamic_max_frames: bool = False
    dynamic_chars_per_second: float = 11.5
    dynamic_frame_budget_mul: float = 1.05
    dynamic_min_frames: int = 20
    dynamic_max_frames_cap: int = 10000

    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_use_cuda_graphs: bool = False
    compile_codebook_predictor: bool = True
    compile_talker: bool = True
    compile_cudagraph_skip_dynamic_graphs: bool = True

    max_concurrent: int = 1
    preload_on_start: bool = True
    warmup_on_start: bool = False
    warmup_text: str = "Привет."
    warmup_chunks: int = 2

    first_chunk_timeout_s: float = 30.0
    stream_idle_timeout_s: float = 30.0


class Qwen3Runtime:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg

        self._model = None
        self._model_signature = None
        self._lock = threading.Lock()

        self._import_error = None
        self._load_error = None

        self._qwen_tts_repo_path = ""
        self._qwen_tts_module_file = None
        self._qwen_tts_capabilities = {
            "stream_generate_voice_clone": False,
            "stream_generate_custom_voice": False,
        }

        self._vc_prompt_cache = None
        self._vc_prompt_cache_key = None
        try:
            self.cfg.runtime_flavor = self._normalize_runtime_flavor(self.cfg.runtime_flavor)
            self.cfg.voice_mode = self._normalize_voice_mode(self.cfg.voice_mode)
        except Exception as exc:
            self._import_error = exc
            logging.error("Qwen3 runtime config error: %s", exc)
            return

        try:
            self._qwen_tts_repo_path = self._resolve_qwen_tts_repo_path(self.cfg.qwen_tts_repo_path)
            if self._qwen_tts_repo_path:
                self._inject_qwen_tts_repo_path(self._qwen_tts_repo_path)
        except Exception as exc:
            self._import_error = exc
            logging.error("Qwen3 runtime path error: %s", exc)
            return

        try:
            import qwen_tts
            from qwen_tts import Qwen3TTSModel

            self._qwen_tts_module_file = getattr(qwen_tts, "__file__", None)
            self._qwen_tts_capabilities = {
                "stream_generate_voice_clone": bool(hasattr(Qwen3TTSModel, "stream_generate_voice_clone")),
                "stream_generate_custom_voice": bool(hasattr(Qwen3TTSModel, "stream_generate_custom_voice")),
            }
            if not self._qwen_tts_capabilities["stream_generate_voice_clone"]:
                raise RuntimeError("Loaded qwen_tts runtime does not expose stream_generate_voice_clone().")

            logging.info("Qwen3 runtime qwen_tts module: %s", self._qwen_tts_module_file)
        except Exception as exc:
            self._import_error = exc
            logging.error("Qwen3 runtime unavailable: %s", exc)

    @staticmethod
    def _normalize_runtime_flavor(value: Any) -> str:
        mode = str(value or "dffdeeq").strip().lower().replace("-", "_")
        if mode in {"", "none", "null", "auto", "original", "dffdeeq"}:
            return "dffdeeq"
        raise ValueError("runtime_flavor must be 'dffdeeq' (or alias auto/original).")

    @staticmethod
    def _normalize_voice_mode(value: Any) -> str:
        mode = str(value or "voice_clone").strip().lower().replace("-", "_")
        if mode in {"voice_clone", "clone"}:
            return "voice_clone"
        raise ValueError("voice_mode must be voice_clone for dffdeeq runtime.")

    @staticmethod
    def _resolve_qwen_tts_repo_path(path_value: Any) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return ""
        raw = os.path.expandvars(raw)
        raw = os.path.expanduser(raw)
        abs_path = os.path.abspath(raw)
        package_dir = os.path.join(abs_path, "qwen_tts")
        if not os.path.isdir(abs_path):
            raise FileNotFoundError(f"qwen_tts repo path does not exist: {abs_path}")
        if not os.path.isdir(package_dir):
            raise FileNotFoundError(f"qwen_tts package directory is missing: {package_dir}")
        return abs_path

    @staticmethod
    def _resolve_dtype(torch_mod, dtype_name: str):
        name = str(dtype_name or "bfloat16").strip().lower()
        if name in {"float16", "fp16"}:
            return torch_mod.float16
        if name in {"float32", "fp32"}:
            return torch_mod.float32
        return torch_mod.bfloat16

    @staticmethod
    def _resolve_ref_audio_path(path_value: Any) -> str:
        path_s = str(path_value or "").strip()
        if not path_s:
            return ""
        path_s = os.path.expandvars(path_s)
        path_s = os.path.expanduser(path_s)
        return os.path.abspath(path_s)

    @staticmethod
    def _pick(req_value: Any, cfg_value: Any) -> Any:
        return cfg_value if req_value is None else req_value

    def _inject_qwen_tts_repo_path(self, repo_path: str) -> None:
        if not repo_path:
            return
        normalized = os.path.normcase(os.path.abspath(repo_path))
        existing = [os.path.normcase(os.path.abspath(p)) for p in sys.path if isinstance(p, str) and p]
        if normalized in existing:
            return
        sys.path.insert(0, repo_path)
        logging.info("Qwen3 runtime: injected qwen_tts repo path: %s", repo_path)

    def is_ready(self) -> bool:
        return self._import_error is None and self._load_error is None

    def status_payload(self) -> Dict[str, Any]:
        payload = {
            "ok": True,
            "provider": "qwen3",
            "ready": self.is_ready(),
            "model_loaded": bool(self._model is not None),
            "runtime_flavor": self.cfg.runtime_flavor,
            "voice_mode": self.cfg.voice_mode,
            "runtime_provider": self.cfg.provider,
            "runtime_gpu_id": self.cfg.gpu_id,
            "runtime_device": self.cfg.device,
            "runtime_attn_implementation": self.cfg.attn_implementation,
            "runtime_use_compile": bool(self.cfg.use_compile),
            "runtime_compile_mode": self.cfg.compile_mode,
            "runtime_compile_cudagraph_skip_dynamic_graphs": bool(self.cfg.compile_cudagraph_skip_dynamic_graphs),
            "qwen_tts_repo_path": self._qwen_tts_repo_path or None,
            "qwen_tts_module_file": self._qwen_tts_module_file,
            "qwen_tts_capabilities": dict(self._qwen_tts_capabilities),
            "voice_clone_prompt_cached": bool(self._vc_prompt_cache is not None),
        }
        if self._import_error is not None:
            payload["detail"] = f"qwen_tts import failed: {self._import_error}"
        if self._load_error is not None:
            payload["detail"] = f"model load failed: {self._load_error}"
        return payload

    def _reset_loaded_model(self) -> None:
        self._model = None
        self._model_signature = None
        self._vc_prompt_cache = None
        self._vc_prompt_cache_key = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_model_loaded(
        self,
        model_id: str,
        provider: str,
        gpu_id: int,
        device: str,
        dtype: str,
        attn_implementation: Optional[str],
        decode_window_frames: int,
    ) -> None:
        if self._import_error is not None:
            raise RuntimeError(f"qwen_tts import failed: {self._import_error}")

        provider_s = str(provider or self.cfg.provider).strip().lower()
        if provider_s.startswith("cpu"):
            resolved_device = "cpu"
        else:
            resolved_device = str(device or self.cfg.device or f"cuda:{int(gpu_id)}").strip()

        signature = (
            str(model_id),
            provider_s,
            int(gpu_id),
            str(resolved_device),
            str(dtype),
            str(attn_implementation or ""),
            bool(self.cfg.use_compile),
            int(decode_window_frames),
        )

        with self._lock:
            if self._model is not None and self._model_signature == signature:
                return

            self._reset_loaded_model()
            try:
                import torch
                from qwen_tts import Qwen3TTSModel

                torch_dtype = self._resolve_dtype(torch, dtype)

                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass

                if bool(self.cfg.use_compile):
                    try:
                        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = bool(
                            self.cfg.compile_cudagraph_skip_dynamic_graphs
                        )
                    except Exception:
                        pass

                model = Qwen3TTSModel.from_pretrained(
                    str(model_id),
                    device_map=resolved_device,
                    torch_dtype=torch_dtype,
                    attn_implementation=(attn_implementation or None),
                )

                if bool(self.cfg.use_compile):
                    try:
                        model.enable_streaming_optimizations(
                            decode_window_frames=int(decode_window_frames),
                            use_compile=True,
                            use_cuda_graphs=bool(self.cfg.compile_use_cuda_graphs),
                            compile_mode=str(self.cfg.compile_mode or "reduce-overhead"),
                            compile_codebook_predictor=bool(self.cfg.compile_codebook_predictor),
                            compile_talker=bool(self.cfg.compile_talker),
                        )
                    except TypeError:
                        model.enable_streaming_optimizations(
                            decode_window_frames=int(decode_window_frames),
                            use_compile=True,
                            compile_mode=str(self.cfg.compile_mode or "reduce-overhead"),
                        )

                self._model = model
                self._model_signature = signature
                self._load_error = None
                logging.info(
                    "Qwen3 model loaded: %s (provider=%s gpu_id=%s device=%s compile=%s mode=%s skip_dyn_cudagraphs=%s)",
                    model_id,
                    provider_s,
                    int(gpu_id),
                    resolved_device,
                    int(bool(self.cfg.use_compile)),
                    str(self.cfg.compile_mode or "reduce-overhead"),
                    int(bool(self.cfg.compile_cudagraph_skip_dynamic_graphs)),
                )
            except Exception as exc:
                self._load_error = exc
                self._reset_loaded_model()
                raise

    def _estimate_max_frames(self, text: str, requested_max_frames: int) -> int:
        limit = max(16, int(requested_max_frames))
        if not bool(self.cfg.dynamic_max_frames):
            return limit

        norm = " ".join(str(text or "").split())
        text_len = len(norm)
        punct = sum(1 for ch in norm if ch in ".!?;:")

        est_seconds = (float(text_len) / float(max(0.1, self.cfg.dynamic_chars_per_second))) + (0.08 * punct) + 0.35
        est_frames = int(est_seconds * 12.0 * float(self.cfg.dynamic_frame_budget_mul))
        est_frames = max(int(self.cfg.dynamic_min_frames), est_frames)
        est_frames = min(int(self.cfg.dynamic_max_frames_cap), est_frames)

        return max(16, min(limit, est_frames))

    def _get_or_build_voice_clone_prompt(
        self,
        ref_audio_path: str,
        ref_text: Optional[str],
        x_vector_only_mode: bool,
    ):
        if self._model is None:
            raise RuntimeError("Qwen3 model is not loaded for voice clone prompt creation.")

        ref_audio_abs = self._resolve_ref_audio_path(ref_audio_path)
        if not ref_audio_abs:
            raise ValueError("voice_clone mode requires ref_audio_path.")
        if not os.path.exists(ref_audio_abs):
            raise FileNotFoundError(f"voice_clone ref audio not found: {ref_audio_abs}")

        ref_text_norm = (str(ref_text).strip() if ref_text is not None else "")
        cache_key = (ref_audio_abs, ref_text_norm, bool(x_vector_only_mode))
        if self._vc_prompt_cache is not None and self._vc_prompt_cache_key == cache_key:
            return self._vc_prompt_cache

        prompt_items = self._model.create_voice_clone_prompt(
            ref_audio=ref_audio_abs,
            ref_text=(ref_text_norm if ref_text_norm else None),
            x_vector_only_mode=bool(x_vector_only_mode),
        )
        if not prompt_items:
            raise RuntimeError("Failed to build voice_clone prompt from reference audio.")

        self._vc_prompt_cache = prompt_items[0]
        self._vc_prompt_cache_key = cache_key
        logging.info(
            "Qwen3 voice_clone prompt cached: ref_audio=%s x_vector_only_mode=%s ref_text=%s",
            ref_audio_abs,
            bool(x_vector_only_mode),
            ("set" if ref_text_norm else "empty"),
        )
        return self._vc_prompt_cache

    def stream_voice_clone(self, req: TTSRequest) -> Iterable[tuple[bytes, int]]:
        cfg = self.cfg

        text = str(req.text or "").strip()
        if not text:
            raise ValueError("text must be non-empty")

        runtime_flavor = self._normalize_runtime_flavor(self._pick(req.runtime_flavor, cfg.runtime_flavor))
        if runtime_flavor != "dffdeeq":
            raise RuntimeError("Only runtime_flavor=dffdeeq is supported by this sidecar build.")

        voice_mode = self._normalize_voice_mode(self._pick(req.voice_mode, cfg.voice_mode))
        if voice_mode != "voice_clone":
            raise RuntimeError("Only voice_mode=voice_clone is supported by this sidecar build.")

        model_id = str(self._pick(req.model_id, cfg.model_id))
        provider = str(self._pick(req.provider, cfg.provider) or "cuda").strip().lower()
        gpu_id = int(self._pick(req.gpu_id, cfg.gpu_id))
        device = str(self._pick(req.device, cfg.device) or "").strip()
        dtype = str(self._pick(req.dtype, cfg.dtype))
        attn_implementation = self._pick(req.attn_implementation, cfg.attn_implementation)

        language = str(self._pick(req.language, cfg.language) or "Auto")
        emit_every_frames = max(1, int(self._pick(req.emit_every_frames, cfg.emit_every_frames)))
        decode_window_frames = max(8, int(self._pick(req.decode_window_frames, cfg.decode_window_frames)))
        overlap_samples = max(0, int(self._pick(req.overlap_samples, cfg.overlap_samples)))
        max_frames_requested = max(16, int(self._pick(req.max_frames, cfg.max_frames)))
        max_frames = self._estimate_max_frames(text, max_frames_requested)

        use_optimized_decode = (
            bool(req.use_optimized_decode)
            if req.use_optimized_decode is not None
            else bool(cfg.use_optimized_decode)
        )

        do_sample = bool(req.do_sample) if req.do_sample is not None else bool(cfg.do_sample)
        top_p = float(req.top_p) if req.top_p is not None else float(cfg.top_p)
        top_k = int(req.top_k) if req.top_k is not None else int(cfg.top_k)
        temperature = float(req.temperature) if req.temperature is not None else float(cfg.temperature)

        ref_audio_path = str(self._pick(req.ref_audio_path, cfg.ref_audio_path) or "")
        ref_text_raw = self._pick(req.ref_text, cfg.ref_text)
        ref_text = str(ref_text_raw).strip() if ref_text_raw is not None else ""
        x_vector_only_mode = bool(self._pick(req.x_vector_only_mode, cfg.x_vector_only_mode))
        self._ensure_model_loaded(
            model_id=model_id,
            provider=provider,
            gpu_id=gpu_id,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            decode_window_frames=decode_window_frames,
        )

        voice_clone_prompt = self._get_or_build_voice_clone_prompt(
            ref_audio_path=ref_audio_path,
            ref_text=(ref_text if ref_text else None),
            x_vector_only_mode=x_vector_only_mode,
        )

        stream_kwargs = {
            "text": text,
            "language": language,
            "voice_clone_prompt": voice_clone_prompt,
            "emit_every_frames": emit_every_frames,
            "decode_window_frames": decode_window_frames,
            "overlap_samples": overlap_samples,
            "max_frames": max_frames,
            "use_optimized_decode": use_optimized_decode,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        }

        for chunk, sr in self._model.stream_generate_voice_clone(**stream_kwargs):
            pcm = _to_pcm16_bytes(chunk)
            if pcm:
                yield pcm, int(sr)

    def preload_and_warmup(self) -> None:
        if not bool(self.cfg.preload_on_start):
            return

        try:
            self._ensure_model_loaded(
                model_id=str(self.cfg.model_id),
                provider=str(self.cfg.provider),
                gpu_id=int(self.cfg.gpu_id),
                device=str(self.cfg.device),
                dtype=str(self.cfg.dtype),
                attn_implementation=str(self.cfg.attn_implementation or "").strip() or None,
                decode_window_frames=int(self.cfg.decode_window_frames),
            )
            self._get_or_build_voice_clone_prompt(
                ref_audio_path=str(self.cfg.ref_audio_path or ""),
                ref_text=(str(self.cfg.ref_text or "").strip() or None),
                x_vector_only_mode=bool(self.cfg.x_vector_only_mode),
            )
        except Exception as exc:
            logging.warning("Qwen3 preload failed: %s", exc)
            return

        if not bool(self.cfg.warmup_on_start):
            return

        warmup_text = str(self.cfg.warmup_text or "").strip()
        if not warmup_text:
            return

        request = TTSRequest(
            text=warmup_text,
            language=self.cfg.language,
            provider=self.cfg.provider,
            gpu_id=self.cfg.gpu_id,
            model_id=self.cfg.model_id,
            device=self.cfg.device,
            dtype=self.cfg.dtype,
            attn_implementation=self.cfg.attn_implementation,
            runtime_flavor=self.cfg.runtime_flavor,
            voice_mode=self.cfg.voice_mode,
            ref_audio_path=self.cfg.ref_audio_path,
            ref_text=self.cfg.ref_text,
            x_vector_only_mode=bool(self.cfg.x_vector_only_mode),
            emit_every_frames=int(self.cfg.emit_every_frames),
            decode_window_frames=int(self.cfg.decode_window_frames),
            overlap_samples=int(self.cfg.overlap_samples),
            max_frames=int(self.cfg.max_frames),
            use_optimized_decode=bool(self.cfg.use_optimized_decode),
            do_sample=False,
            top_p=float(self.cfg.top_p),
            top_k=int(self.cfg.top_k),
            temperature=float(self.cfg.temperature),
        )

        started = time.perf_counter()
        chunks = 0
        samples = 0
        sr = int(self.cfg.sample_rate)
        max_chunks = max(1, int(self.cfg.warmup_chunks))

        try:
            for pcm_bytes, sr in self.stream_voice_clone(request):
                chunks += 1
                samples += int(len(pcm_bytes) // 2)
                if chunks >= max_chunks:
                    break
        except Exception as exc:
            logging.warning("Qwen3 runtime warmup failed: %s", exc)
            return

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        audio_ms = int((samples / max(1, int(sr))) * 1000) if samples > 0 else 0
        logging.info("Qwen3 runtime warmup done: chunks=%s elapsed_ms=%s audio_ms=%s", chunks, elapsed_ms, audio_ms)


def build_app(cfg: ServerConfig) -> FastAPI:
    app = FastAPI(title="Qwen3 TTS Sidecar (dffdeeq)", version="1.0.0")
    runtime = Qwen3Runtime(cfg)
    semaphore = asyncio.Semaphore(max(1, int(cfg.max_concurrent)))
    inference_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="qwen3_sidecar_inference",
    )

    @app.on_event("startup")
    async def _startup_preload():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(inference_executor, runtime.preload_and_warmup)

    @app.on_event("shutdown")
    async def _shutdown_executor():
        inference_executor.shutdown(wait=False, cancel_futures=True)

    @app.get("/health")
    async def health():
        return JSONResponse(runtime.status_payload())

    @app.post("/v1/tts/stream")
    async def tts_stream(req: TTSRequest):
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text must be non-empty")

        if req.sample_width is not None and int(req.sample_width) != 2:
            raise HTTPException(status_code=400, detail="Only PCM16 stream is supported (sample_width=2).")
        if req.channels is not None and int(req.channels) != 1:
            raise HTTPException(status_code=400, detail="Only mono stream is supported (channels=1).")

        try:
            runtime._normalize_runtime_flavor(req.runtime_flavor if req.runtime_flavor is not None else cfg.runtime_flavor)
            runtime._normalize_voice_mode(req.voice_mode if req.voice_mode is not None else cfg.voice_mode)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not runtime.is_ready() and runtime._import_error is not None:
            raise HTTPException(status_code=503, detail=f"Runtime unavailable: {runtime._import_error}")
        async with semaphore:
            async def stream_bytes():
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                sentinel = object()
                started = time.perf_counter()

                first_chunk_timeout_s = max(3.0, float(cfg.first_chunk_timeout_s))
                stream_idle_timeout_s = max(3.0, float(cfg.stream_idle_timeout_s))

                def worker():
                    try:
                        for pcm_bytes, _sr in runtime.stream_voice_clone(req):
                            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", pcm_bytes))
                    except Exception as exc:
                        err = {"error": repr(exc), "traceback": traceback.format_exc()}
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", err))
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, (sentinel, None))

                loop.run_in_executor(inference_executor, worker)

                first = True
                while True:
                    timeout_s = first_chunk_timeout_s if first else stream_idle_timeout_s
                    try:
                        tag, payload = await asyncio.wait_for(queue.get(), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        if first:
                            logging.error("Qwen3 sidecar timeout before first chunk (>%ss).", timeout_s)
                        else:
                            logging.warning("Qwen3 sidecar stream idle timeout (>%ss).", timeout_s)
                        return

                    if tag is sentinel:
                        break

                    if tag == "error":
                        if isinstance(payload, dict):
                            logging.error("Qwen3 sidecar stream failed: %s", payload.get("error"))
                            tb = str(payload.get("traceback", "") or "").strip()
                            if tb:
                                logging.error("Qwen3 sidecar traceback:\n%s", tb)
                        else:
                            logging.error("Qwen3 sidecar stream failed: %s", payload)
                        return

                    if payload:
                        if first:
                            first = False
                            latency_ms = int((time.perf_counter() - started) * 1000)
                            logging.info("Qwen3 sidecar first-audio latency_ms=%s", latency_ms)
                        yield payload

            headers = {
                "x-sample-rate": str(int(req.sample_rate or cfg.sample_rate)),
                "x-sample-width": str(int(cfg.sample_width)),
                "x-channels": str(int(cfg.channels)),
            }
            return StreamingResponse(stream_bytes(), media_type="application/octet-stream", headers=headers)

    return app


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6116)

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--language", type=str, default="russian")
    parser.add_argument("--provider", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--sample_width", type=int, default=2)
    parser.add_argument("--channels", type=int, default=1)

    parser.add_argument("--runtime_flavor", type=str, default="dffdeeq")
    parser.add_argument("--voice_mode", type=str, default="voice_clone")
    parser.add_argument("--qwen_tts_repo_path", type=str, default="")

    parser.add_argument("--ref_audio_path", type=str, default="")
    parser.add_argument("--ref_text", type=str, default="")
    parser.add_argument("--x_vector_only_mode", type=int, default=1)

    parser.add_argument("--emit_every_frames", type=int, default=4)
    parser.add_argument("--decode_window_frames", type=int, default=80)
    parser.add_argument("--overlap_samples", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=10000)
    parser.add_argument("--use_optimized_decode", type=int, default=1)

    parser.add_argument("--do_sample", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--dynamic_max_frames", type=int, default=0)
    parser.add_argument("--dynamic_chars_per_second", type=float, default=11.5)
    parser.add_argument("--dynamic_frame_budget_mul", type=float, default=1.05)
    parser.add_argument("--dynamic_min_frames", type=int, default=20)
    parser.add_argument("--dynamic_max_frames_cap", type=int, default=10000)
    parser.add_argument("--use_compile", type=int, default=0)
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--compile_use_cuda_graphs", type=int, default=0)
    parser.add_argument("--compile_codebook_predictor", type=int, default=1)
    parser.add_argument("--compile_talker", type=int, default=1)
    parser.add_argument("--compile_cudagraph_skip_dynamic_graphs", type=int, default=1)

    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--preload_on_start", type=int, default=1)
    parser.add_argument("--warmup_on_start", type=int, default=0)
    parser.add_argument("--warmup_text", type=str, default="Привет.")
    parser.add_argument("--warmup_chunks", type=int, default=2)

    parser.add_argument("--first_chunk_timeout_s", type=float, default=30.0)
    parser.add_argument("--stream_idle_timeout_s", type=float, default=30.0)

    ns = parser.parse_args()

    return ServerConfig(
        host=ns.host,
        port=ns.port,
        model_id=ns.model_id,
        language=ns.language,
        provider=ns.provider,
        gpu_id=int(ns.gpu_id),
        device=ns.device,
        dtype=ns.dtype,
        attn_implementation=ns.attn_implementation,
        sample_rate=int(ns.sample_rate),
        sample_width=int(ns.sample_width),
        channels=int(ns.channels),
        runtime_flavor=ns.runtime_flavor,
        voice_mode=ns.voice_mode,
        qwen_tts_repo_path=ns.qwen_tts_repo_path,
        ref_audio_path=ns.ref_audio_path,
        ref_text=ns.ref_text,
        x_vector_only_mode=bool(int(ns.x_vector_only_mode)),
        emit_every_frames=int(ns.emit_every_frames),
        decode_window_frames=int(ns.decode_window_frames),
        overlap_samples=int(ns.overlap_samples),
        max_frames=int(ns.max_frames),
        use_optimized_decode=bool(int(ns.use_optimized_decode)),
        do_sample=bool(int(ns.do_sample)),
        top_p=float(ns.top_p),
        top_k=int(ns.top_k),
        temperature=float(ns.temperature),
        dynamic_max_frames=bool(int(ns.dynamic_max_frames)),
        dynamic_chars_per_second=float(ns.dynamic_chars_per_second),
        dynamic_frame_budget_mul=float(ns.dynamic_frame_budget_mul),
        dynamic_min_frames=int(ns.dynamic_min_frames),
        dynamic_max_frames_cap=int(ns.dynamic_max_frames_cap),
        use_compile=bool(int(ns.use_compile)),
        compile_mode=str(ns.compile_mode),
        compile_use_cuda_graphs=bool(int(ns.compile_use_cuda_graphs)),
        compile_codebook_predictor=bool(int(ns.compile_codebook_predictor)),
        compile_talker=bool(int(ns.compile_talker)),
        compile_cudagraph_skip_dynamic_graphs=bool(int(ns.compile_cudagraph_skip_dynamic_graphs)),
        max_concurrent=int(ns.max_concurrent),
        preload_on_start=bool(int(ns.preload_on_start)),
        warmup_on_start=bool(int(ns.warmup_on_start)),
        warmup_text=str(ns.warmup_text),
        warmup_chunks=int(ns.warmup_chunks),
        first_chunk_timeout_s=float(ns.first_chunk_timeout_s),
        stream_idle_timeout_s=float(ns.stream_idle_timeout_s),
    )


def main() -> None:
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [qwen3_tts_sidecar] %(message)s")
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
