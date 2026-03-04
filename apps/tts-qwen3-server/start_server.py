#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
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
    voice_mode: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: Optional[bool] = None
    max_new_tokens: Optional[int] = None
    max_frames: Optional[int] = None
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
    voice_mode: str = "custom_voice"  # custom_voice | voice_clone
    ref_audio_path: str = ""
    ref_text: str = ""
    x_vector_only_mode: bool = True
    preload_voice_clone_prompt: bool = True
    emit_every_frames: int = 12
    decode_window_frames: int = 80
    first_chunk_emit_every: int = 5
    first_chunk_decode_window: int = 48
    first_chunk_frames: int = 48
    overlap_samples: int = 512
    max_new_tokens: int = 1024
    max_frames: int = 1024
    use_optimized_decode: bool = True
    repetition_penalty: float = 1.0
    repetition_penalty_window: int = 100
    max_concurrent: int = 1
    instruct_prefix: str = ""
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_use_cuda_graphs: bool = False
    compile_codebook_predictor: bool = False
    compile_talker: bool = True
    preload_on_start: bool = True
    warmup_on_start: bool = True
    warmup_text: str = "Привет."
    warmup_emit_every_frames: int = 4
    warmup_decode_window_frames: int = 64
    warmup_max_new_tokens: int = 96
    warmup_speaker: str = ""
    warmup_language: str = ""
    warmup_chunks: int = 2
    first_chunk_timeout_s: float = 18.0
    stream_idle_timeout_s: float = 20.0


class Qwen3Runtime:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self._model = None
        self._model_id_loaded = None
        self._attn_impl_loaded = None
        self._compile_loaded = None
        self._lock = threading.Lock()
        self._import_error = None
        self._load_error = None
        self._compile_runtime_disabled = False
        self._vc_prompt_cache = None
        self._vc_prompt_cache_key = None

        try:
            import importlib.util
            import qwen_tts  # noqa: F401

            if importlib.util.find_spec("flash_attn") is None:
                logging.warning(
                    "flash-attn is unavailable on this runtime; using attn_implementation=%s",
                    self.cfg.attn_implementation or "default",
                )
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
            "compile_loaded": self._compile_loaded,
            "compile_runtime_disabled": self._compile_runtime_disabled,
            "runtime_provider": self.cfg.provider,
            "runtime_gpu_id": self.cfg.gpu_id,
            "runtime_device": self.cfg.device,
            "runtime_attn_implementation": self.cfg.attn_implementation,
            "voice_mode": self.cfg.voice_mode,
            "voice_clone_prompt_cached": bool(self._vc_prompt_cache is not None),
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

    def _reset_loaded_model(self) -> None:
        self._model = None
        self._model_id_loaded = None
        self._attn_impl_loaded = None
        self._compile_loaded = None
        self._vc_prompt_cache = None
        self._vc_prompt_cache_key = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _looks_like_compile_failure(exc: Exception) -> bool:
        text = f"{repr(exc)}\n{traceback.format_exc()}".lower()
        markers = (
            "torch._inductor",
            "cudagraph",
            "triton",
            "compile",
            "_is_key_in_tls",
            "assertionerror",
        )
        return any(marker in text for marker in markers)

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
                and self._compile_loaded == (bool(getattr(self.cfg, "use_compile", False)) and not self._compile_runtime_disabled)
            ):
                return

            try:
                import torch
                from qwen_tts import Qwen3TTSModel

                torch_dtype = self._resolve_dtype(torch, dtype)
                use_compile = bool(getattr(self.cfg, "use_compile", False)) and not self._compile_runtime_disabled
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
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass

                compile_mode = str(getattr(self.cfg, "compile_mode", "reduce-overhead") or "reduce-overhead")
                compile_use_cuda_graphs = bool(getattr(self.cfg, "compile_use_cuda_graphs", False))
                compile_codebook_predictor = bool(getattr(self.cfg, "compile_codebook_predictor", False))
                compile_talker = bool(getattr(self.cfg, "compile_talker", True))

                # Dynamic-shape cudagraph churn can destabilize long-running sessions on Windows.
                # Prefer skipping dynamic graph capture unless explicitly requested.
                if use_compile:
                    try:
                        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
                        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
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
                            logging.info(
                                "Qwen3 runtime: enabling compile optimizations "
                                "(mode=%s, cuda_graphs=%s, codebook_predictor=%s, talker=%s).",
                                compile_mode,
                                compile_use_cuda_graphs,
                                compile_codebook_predictor,
                                compile_talker,
                            )
                        else:
                            logging.info("Qwen3 runtime: compile optimizations are disabled (use_compile=false).")
                        try:
                            model.enable_streaming_optimizations(
                                decode_window_frames=int(decode_window_frames),
                                use_compile=use_compile,
                                use_cuda_graphs=compile_use_cuda_graphs,
                                compile_mode=compile_mode,
                                compile_codebook_predictor=compile_codebook_predictor,
                                compile_talker=compile_talker,
                            )
                        except TypeError:
                            # Fallback for forks that expose shorter signature.
                            model.enable_streaming_optimizations(
                                decode_window_frames=int(decode_window_frames),
                                use_compile=use_compile,
                                compile_mode=compile_mode,
                            )
                    except Exception as exc:
                        logging.warning("enable_streaming_optimizations failed: %s", exc)

                self._model = model
                self._model_id_loaded = model_id
                self._attn_impl_loaded = attn_implementation or None
                self._compile_loaded = use_compile
                self._load_error = None
                logging.info(
                    "Qwen3 model loaded: %s (attn=%s, compile=%s)",
                    model_id,
                    self._attn_impl_loaded or "default",
                    use_compile,
                )
            except Exception as exc:
                self._load_error = exc
                self._reset_loaded_model()
                raise

    @staticmethod
    def _normalize_voice_mode(value: Any) -> str:
        mode = str(value or "custom_voice").strip().lower()
        if mode in ("clone", "voice-clone", "voice_clone"):
            return "voice_clone"
        return "custom_voice"

    @staticmethod
    def _resolve_ref_audio_path(path_value: Any) -> str:
        path_s = str(path_value or "").strip()
        if not path_s:
            return ""
        path_s = os.path.expandvars(path_s)
        path_s = os.path.expanduser(path_s)
        return os.path.abspath(path_s)

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

        if (
            self._normalize_voice_mode(getattr(cfg, "voice_mode", "custom_voice")) == "voice_clone"
            and bool(getattr(cfg, "preload_voice_clone_prompt", True))
        ):
            try:
                self._get_or_build_voice_clone_prompt(
                    ref_audio_path=str(getattr(cfg, "ref_audio_path", "") or ""),
                    ref_text=(str(getattr(cfg, "ref_text", "") or "").strip() or None),
                    x_vector_only_mode=bool(getattr(cfg, "x_vector_only_mode", True)),
                )
            except Exception as exc:
                logging.warning("Qwen3 voice_clone prompt preload failed: %s", exc)

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
            max_frames=int(getattr(cfg, "warmup_max_new_tokens", 96)),
            overlap_samples=int(getattr(cfg, "overlap_samples", 0)),
            use_optimized_decode=bool(getattr(cfg, "use_optimized_decode", True)),
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
        def _pick(req_value: Any, cfg_value: Any) -> Any:
            # Important: preserve explicit falsy values from request (e.g. 0).
            return cfg_value if req_value is None else req_value

        cfg = self.cfg
        model_id = str(_pick(req.model_id, cfg.model_id))
        provider = str(_pick(req.provider, cfg.provider) or "cuda").strip().lower()
        gpu_id = int(_pick(req.gpu_id, cfg.gpu_id))
        if req.device and str(req.device).strip():
            device = str(req.device).strip()
        else:
            if provider.startswith("cpu"):
                device = "cpu"
            else:
                device = cfg.device or f"cuda:{gpu_id}"
        dtype = str(_pick(req.dtype, cfg.dtype))
        attn_implementation = _pick(req.attn_implementation, cfg.attn_implementation)
        decode_window_frames = int(_pick(req.decode_window_frames, cfg.decode_window_frames))
        emit_every_frames = int(_pick(req.emit_every_frames, cfg.emit_every_frames))
        first_chunk_emit_every = int(_pick(req.first_chunk_emit_every, cfg.first_chunk_emit_every))
        first_chunk_decode_window = int(_pick(req.first_chunk_decode_window, cfg.first_chunk_decode_window))
        first_chunk_frames = int(_pick(req.first_chunk_frames, cfg.first_chunk_frames))
        overlap_samples = int(_pick(req.overlap_samples, cfg.overlap_samples))
        use_optimized_decode = (
            bool(req.use_optimized_decode)
            if req.use_optimized_decode is not None
            else bool(getattr(cfg, "use_optimized_decode", True))
        )
        repetition_penalty = float(_pick(req.repetition_penalty, cfg.repetition_penalty))
        repetition_penalty_window = int(_pick(req.repetition_penalty_window, cfg.repetition_penalty_window))
        max_new_tokens = int(_pick(req.max_new_tokens, cfg.max_new_tokens))
        max_frames = int(_pick(req.max_frames, getattr(cfg, "max_frames", max_new_tokens)))
        if req.max_frames is None:
            # stream_generate_custom_voice in qwen_tts uses max_frames (not max_new_tokens)
            # as the effective generation cap; keep them aligned by default.
            max_frames = max_new_tokens
        max_frames = max(32, max_frames)
        speaker = str(_pick(req.speaker, cfg.speaker))
        language = str(_pick(req.language, cfg.language))
        instruct = str(req.instruct or "")
        voice_mode = self._normalize_voice_mode(_pick(req.voice_mode, getattr(cfg, "voice_mode", "custom_voice")))
        ref_audio_path = str(_pick(req.ref_audio_path, getattr(cfg, "ref_audio_path", "")) or "")
        ref_text_raw = _pick(req.ref_text, getattr(cfg, "ref_text", ""))
        ref_text = (str(ref_text_raw).strip() if ref_text_raw is not None else "")
        x_vector_only_mode = bool(_pick(req.x_vector_only_mode, getattr(cfg, "x_vector_only_mode", True)))
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
            "language": language,
            "emit_every_frames": emit_every_frames,
            "decode_window_frames": decode_window_frames,
            "max_frames": max_frames,
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
        else:
            # Keep stable deterministic defaults if request did not override sampling.
            stream_kwargs["do_sample"] = False

        if voice_mode == "voice_clone":
            vc_prompt = self._get_or_build_voice_clone_prompt(
                ref_audio_path=ref_audio_path,
                ref_text=(ref_text if ref_text else None),
                x_vector_only_mode=x_vector_only_mode,
            )
            stream_kwargs.update(
                {
                    "voice_clone_prompt": vc_prompt,
                    "ref_audio": self._resolve_ref_audio_path(ref_audio_path),
                    "ref_text": (ref_text if ref_text else None),
                    "x_vector_only_mode": x_vector_only_mode,
                    "non_streaming_mode": False,
                }
            )
        else:
            stream_kwargs.update(
                {
                    "speaker": speaker,
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": repetition_penalty,
                    "repetition_penalty_window": repetition_penalty_window,
                    "first_chunk_emit_every": first_chunk_emit_every,
                    "first_chunk_decode_window": first_chunk_decode_window,
                    "first_chunk_frames": first_chunk_frames,
                }
            )
            if instruct:
                stream_kwargs["instruct"] = instruct

        for attempt in (1, 2):
            try:
                if voice_mode == "voice_clone":
                    stream_fn = getattr(self._model, "stream_generate_voice_clone", None)
                    if stream_fn is None:
                        gen_fn = getattr(self._model, "generate_voice_clone", None)
                        if gen_fn is None:
                            raise RuntimeError(
                                "Loaded qwen_tts model does not support stream_generate_voice_clone() "
                                "or generate_voice_clone()."
                            )
                        logging.warning(
                            "Qwen3 runtime fallback: stream_generate_voice_clone() is unavailable. "
                            "Using generate_voice_clone() chunked fallback."
                        )
                        wavs, sr = gen_fn(
                            text=req.text,
                            language=language,
                            voice_clone_prompt=stream_kwargs.get("voice_clone_prompt"),
                            do_sample=bool(stream_kwargs.get("do_sample", False)),
                            top_p=float(stream_kwargs.get("top_p", 0.9)),
                            top_k=int(stream_kwargs.get("top_k", 50)),
                            temperature=float(stream_kwargs.get("temperature", 0.8)),
                        )
                        if not wavs:
                            return
                        pcm_all = _to_pcm16_bytes(wavs[0])
                        if not pcm_all:
                            return
                        sr_i = int(sr)
                        chunk_ms = 40
                        chunk_samples = max(1, int(sr_i * (chunk_ms / 1000.0)))
                        chunk_bytes = chunk_samples * 2
                        for off in range(0, len(pcm_all), chunk_bytes):
                            part = pcm_all[off:off + chunk_bytes]
                            if part:
                                yield part, sr_i
                        return
                else:
                    stream_fn = getattr(self._model, "stream_generate_custom_voice", None)
                    if stream_fn is None:
                        # Compatibility fallback for runtimes without custom streaming.
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
                        gen_kwargs.update(
                            {
                                "do_sample": True,
                                "top_p": 0.9,
                                "temperature": 0.8,
                            }
                        )
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
                    # Fallback for older forks with narrower signatures.
                    if voice_mode == "voice_clone":
                        for key in ("voice_clone_prompt",):
                            stream_kwargs.pop(key, None)
                    else:
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
                return
            except Exception as exc:
                can_retry_non_compile = (
                    attempt == 1
                    and not self._compile_runtime_disabled
                    and self._looks_like_compile_failure(exc)
                )
                if not can_retry_non_compile:
                    raise

                logging.warning(
                    "Qwen3 compile path failed (%s). Disabling compile for runtime and retrying once.",
                    repr(exc),
                )
                self._compile_runtime_disabled = True
                self._reset_loaded_model()
                self._ensure_model_loaded(
                    model_id=model_id,
                    device=device,
                    dtype=dtype,
                    decode_window_frames=decode_window_frames,
                    attn_implementation=attn_implementation,
                )


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
                first_chunk_timeout_s = max(3.0, float(cfg.first_chunk_timeout_s))
                stream_idle_timeout_s = max(3.0, float(cfg.stream_idle_timeout_s))

                def worker():
                    try:
                        for pcm_bytes, _sr in runtime.stream_custom_voice(req):
                            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", pcm_bytes))
                    except Exception as exc:
                        err = {
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", err))
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, (sentinel, None))

                threading.Thread(target=worker, daemon=True).start()

                first = True
                while True:
                    timeout_s = first_chunk_timeout_s if first else stream_idle_timeout_s
                    try:
                        tag, payload = await asyncio.wait_for(queue.get(), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        if first:
                            logging.error(
                                "Qwen3 sidecar stream timeout before first chunk (>%ss).",
                                timeout_s,
                            )
                            return
                        logging.warning("Qwen3 sidecar stream idle timeout (>%ss). Closing stream.", timeout_s)
                        break
                    if tag is sentinel:
                        break
                    if tag == "error":
                        if isinstance(payload, dict):
                            logging.error("Qwen3 sidecar stream worker failed: %s", payload.get("error"))
                            tb = str(payload.get("traceback", "") or "").strip()
                            if tb:
                                logging.error("Qwen3 sidecar worker traceback:\n%s", tb)
                            detail = str(payload.get("error", "stream worker failure"))
                        else:
                            detail = str(payload or "stream worker failure")
                            logging.error("Qwen3 sidecar stream worker failed: %s", detail)
                        if first:
                            # Return an empty stream body gracefully. Client side detects
                            # zero-byte 200 as failure and can restart/retry without abrupt
                            # socket resets from ASGI exception bubbling.
                            return
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
    parser.add_argument("--voice_mode", type=str, default="custom_voice")
    parser.add_argument("--ref_audio_path", type=str, default="")
    parser.add_argument("--ref_text", type=str, default="")
    parser.add_argument("--x_vector_only_mode", type=int, default=1)
    parser.add_argument("--preload_voice_clone_prompt", type=int, default=1)
    parser.add_argument("--emit_every_frames", type=int, default=12)
    parser.add_argument("--decode_window_frames", type=int, default=80)
    parser.add_argument("--first_chunk_emit_every", type=int, default=5)
    parser.add_argument("--first_chunk_decode_window", type=int, default=48)
    parser.add_argument("--first_chunk_frames", type=int, default=48)
    parser.add_argument("--overlap_samples", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_frames", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty_window", type=int, default=100)
    parser.add_argument("--use_optimized_decode", type=int, default=1)
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--instruct_prefix", type=str, default="")
    parser.add_argument("--use_compile", type=int, default=0)
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--compile_use_cuda_graphs", type=int, default=0)
    parser.add_argument("--compile_codebook_predictor", type=int, default=0)
    parser.add_argument("--compile_talker", type=int, default=1)
    parser.add_argument("--preload_on_start", type=int, default=1)
    parser.add_argument("--warmup_on_start", type=int, default=1)
    parser.add_argument("--warmup_text", type=str, default="Привет.")
    parser.add_argument("--warmup_emit_every_frames", type=int, default=4)
    parser.add_argument("--warmup_decode_window_frames", type=int, default=64)
    parser.add_argument("--warmup_max_new_tokens", type=int, default=96)
    parser.add_argument("--warmup_speaker", type=str, default="")
    parser.add_argument("--warmup_language", type=str, default="")
    parser.add_argument("--warmup_chunks", type=int, default=2)
    parser.add_argument("--first_chunk_timeout_s", type=float, default=18.0)
    parser.add_argument("--stream_idle_timeout_s", type=float, default=20.0)
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
        voice_mode=ns.voice_mode,
        ref_audio_path=ns.ref_audio_path,
        ref_text=ns.ref_text,
        x_vector_only_mode=bool(int(ns.x_vector_only_mode)),
        preload_voice_clone_prompt=bool(int(ns.preload_voice_clone_prompt)),
        emit_every_frames=ns.emit_every_frames,
        decode_window_frames=ns.decode_window_frames,
        first_chunk_emit_every=ns.first_chunk_emit_every,
        first_chunk_decode_window=ns.first_chunk_decode_window,
        first_chunk_frames=ns.first_chunk_frames,
        overlap_samples=ns.overlap_samples,
        max_new_tokens=ns.max_new_tokens,
        max_frames=ns.max_frames,
        use_optimized_decode=bool(int(ns.use_optimized_decode)),
        repetition_penalty=ns.repetition_penalty,
        repetition_penalty_window=ns.repetition_penalty_window,
        max_concurrent=ns.max_concurrent,
        instruct_prefix=ns.instruct_prefix,
        use_compile=bool(int(ns.use_compile)),
        compile_mode=ns.compile_mode,
        compile_use_cuda_graphs=bool(int(ns.compile_use_cuda_graphs)),
        compile_codebook_predictor=bool(int(ns.compile_codebook_predictor)),
        compile_talker=bool(int(ns.compile_talker)),
        preload_on_start=bool(int(ns.preload_on_start)),
        warmup_on_start=bool(int(ns.warmup_on_start)),
        warmup_text=ns.warmup_text,
        warmup_emit_every_frames=ns.warmup_emit_every_frames,
        warmup_decode_window_frames=ns.warmup_decode_window_frames,
        warmup_max_new_tokens=ns.warmup_max_new_tokens,
        warmup_speaker=ns.warmup_speaker,
        warmup_language=ns.warmup_language,
        warmup_chunks=ns.warmup_chunks,
        first_chunk_timeout_s=float(ns.first_chunk_timeout_s),
        stream_idle_timeout_s=float(ns.stream_idle_timeout_s),
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
