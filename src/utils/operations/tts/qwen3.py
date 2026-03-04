import asyncio
import logging
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, Iterable, Optional

import httpx

from utils.processes import ProcessType, TTSProcessRunner

from .base import TTSOperation


_INLINE_EMOTION_RE = re.compile(r"^\s*[\(\[]\s*([^\)\]]{1,48})\s*[\)\]]\s*[:,-]?\s*(.+)$", re.UNICODE)


def _float_to_pcm16_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required to convert Qwen3 TTS chunks to PCM16 bytes.") from exc

    arr = np.asarray(chunk)
    if arr.size == 0:
        return b""

    if arr.dtype == np.int16:
        return arr.tobytes()

    if arr.dtype.kind == "f":
        arr = np.clip(arr.astype(np.float32), -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()

    return arr.astype(np.int16).tobytes()


class Qwen3TTS(TTSOperation):
    """
    Qwen3 TTS provider with two runtime modes:
    - sidecar: HTTP streaming from apps/tts-qwen3-server (recommended)
    - local: in-process qwen_tts model streaming
    """

    def __init__(self):
        super().__init__("qwen3")

        # Runtime mode
        self.mode = "sidecar"  # sidecar | local

        # Sidecar config
        self.base_url = "http://127.0.0.1:6116"
        self.stream_endpoint = "/v1/tts/stream"
        self.health_endpoint = "/health"
        self.request_timeout_s = 45.0
        self.connect_timeout_s = 5.0
        self.sidecar_read_chunk_bytes = 8192

        # Output audio format policy
        self.sample_rate = 24000
        self.sample_width = 2
        self.channels = 1

        # Model/runtime options
        self.model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        self.language = "Auto"
        self.speaker = "Ryan"
        self.provider = "cuda"  # cpu | cuda
        self.gpu_id = 1
        self.device = "cuda:1"
        self.dtype = "bfloat16"
        self.attn_implementation = "sdpa"
        self.max_new_tokens = 1024
        self.do_sample = False
        self.top_p = 0.9
        self.top_k = 50
        self.temperature = 0.8

        # Streaming tuning (fork-dependent; silently downgraded if unsupported)
        self.emit_every_frames = 12
        self.decode_window_frames = 80
        self.first_chunk_emit_every = 5
        self.first_chunk_decode_window = 48
        self.first_chunk_frames = 48
        self.overlap_samples = 512
        self.repetition_penalty = 1.0
        self.repetition_penalty_window = 100

        # Emotion handling
        self.instruct_prefix = ""
        self.emotion_map: Dict[str, str] = {
            "joy": "Говори радостно, тепло и с легкой улыбкой.",
            "sadness": "Говори мягко, спокойнее и чуть тише.",
            "anger": "Говори напряженно и уверенно, без крика.",
            "neutral": "Говори естественно и ровно.",
        }

        # Process lifecycle
        self.process_autostart = True
        self.process_startup_retries = 3
        self.process_startup_backoff_s = 0.6
        self.process_config: Dict[str, Any] = {}
        self._runner: Optional[TTSProcessRunner] = None

        # Local runtime (lazy-loaded)
        self.local_preload = False
        self._local_model = None

    async def start(self) -> None:
        await super().start()

        if self.mode == "sidecar":
            if self.process_autostart:
                self._runner = TTSProcessRunner(
                    link_id=f"tts_{self.op_id}",
                    process_type=ProcessType.QWEN3_TTS,
                    process_config=self.process_config,
                    startup_retries=self.process_startup_retries,
                    startup_backoff_s=self.process_startup_backoff_s,
                )
                await self._runner.ensure_healthy()
                health = await self._runner.health()
                if health.get("port"):
                    self.base_url = f"http://127.0.0.1:{int(health['port'])}"
            await self._ensure_sidecar_ready()
            return

        if self.local_preload:
            await self._ensure_local_model_loaded()

    async def close(self) -> None:
        await super().close()
        if self._runner is not None:
            await self._runner.close()
            self._runner = None

        if self._local_model is not None:
            try:
                del self._local_model
                self._local_model = None
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                self._local_model = None

    async def configure(self, config_d: Dict[str, Any]):
        if "mode" in config_d:
            self.mode = str(config_d["mode"]).strip().lower()
        if "base_url" in config_d:
            self.base_url = str(config_d["base_url"]).rstrip("/")
        if "stream_endpoint" in config_d:
            self.stream_endpoint = "/" + str(config_d["stream_endpoint"]).lstrip("/")
        if "health_endpoint" in config_d:
            self.health_endpoint = "/" + str(config_d["health_endpoint"]).lstrip("/")
        if "request_timeout_s" in config_d:
            self.request_timeout_s = float(config_d["request_timeout_s"])
        if "connect_timeout_s" in config_d:
            self.connect_timeout_s = float(config_d["connect_timeout_s"])
        if "sidecar_read_chunk_bytes" in config_d:
            self.sidecar_read_chunk_bytes = int(config_d["sidecar_read_chunk_bytes"])

        if "sample_rate" in config_d:
            self.sample_rate = int(config_d["sample_rate"])
        if "sample_width" in config_d:
            self.sample_width = int(config_d["sample_width"])
        if "channels" in config_d:
            self.channels = int(config_d["channels"])

        if "model_id" in config_d:
            self.model_id = str(config_d["model_id"])
        if "language" in config_d:
            self.language = str(config_d["language"])
        if "speaker" in config_d:
            self.speaker = str(config_d["speaker"])
        if "provider" in config_d:
            self.provider = str(config_d["provider"]).strip().lower()
        if "gpu_id" in config_d:
            self.gpu_id = int(config_d["gpu_id"])
        device_overridden = False
        if "device" in config_d:
            self.device = str(config_d["device"]).strip()
            device_overridden = True
        if not device_overridden:
            if self.provider.startswith("cpu"):
                self.device = "cpu"
            else:
                self.device = f"cuda:{self.gpu_id}"
        if "dtype" in config_d:
            self.dtype = str(config_d["dtype"])
        if "attn_implementation" in config_d:
            self.attn_implementation = str(config_d["attn_implementation"]).strip()
        if "max_new_tokens" in config_d:
            self.max_new_tokens = int(config_d["max_new_tokens"])
        if "do_sample" in config_d:
            self.do_sample = bool(config_d["do_sample"])
        if "top_p" in config_d:
            self.top_p = float(config_d["top_p"])
        if "top_k" in config_d:
            self.top_k = int(config_d["top_k"])
        if "temperature" in config_d:
            self.temperature = float(config_d["temperature"])

        if "emit_every_frames" in config_d:
            self.emit_every_frames = int(config_d["emit_every_frames"])
        if "decode_window_frames" in config_d:
            self.decode_window_frames = int(config_d["decode_window_frames"])
        if "first_chunk_emit_every" in config_d:
            self.first_chunk_emit_every = int(config_d["first_chunk_emit_every"])
        if "first_chunk_decode_window" in config_d:
            self.first_chunk_decode_window = int(config_d["first_chunk_decode_window"])
        if "first_chunk_frames" in config_d:
            self.first_chunk_frames = int(config_d["first_chunk_frames"])
        if "overlap_samples" in config_d:
            self.overlap_samples = int(config_d["overlap_samples"])
        if "repetition_penalty" in config_d:
            self.repetition_penalty = float(config_d["repetition_penalty"])
        if "repetition_penalty_window" in config_d:
            self.repetition_penalty_window = int(config_d["repetition_penalty_window"])

        if "instruct_prefix" in config_d:
            self.instruct_prefix = str(config_d["instruct_prefix"]).strip()
        if isinstance(config_d.get("emotion_map"), dict):
            self.emotion_map = {str(k).strip().lower(): str(v).strip() for k, v in config_d["emotion_map"].items()}

        if "local_preload" in config_d:
            self.local_preload = bool(config_d["local_preload"])

        if "process_autostart" in config_d:
            self.process_autostart = bool(config_d["process_autostart"])
        if "process_startup_retries" in config_d:
            self.process_startup_retries = int(config_d["process_startup_retries"])
        if "process_startup_backoff_s" in config_d:
            self.process_startup_backoff_s = float(config_d["process_startup_backoff_s"])

        process_cfg = {}
        if isinstance(config_d.get("process"), dict):
            process_cfg.update(config_d["process"])
        # Backward-compatible top-level overrides.
        for key in (
            "host",
            "port",
            "python_executable",
            "model_id",
            "speaker",
            "language",
            "provider",
            "gpu_id",
            "device",
            "dtype",
            "attn_implementation",
            "sample_rate",
            "sample_width",
            "channels",
            "emit_every_frames",
            "decode_window_frames",
            "first_chunk_emit_every",
            "first_chunk_decode_window",
            "first_chunk_frames",
            "overlap_samples",
            "max_new_tokens",
            "repetition_penalty",
            "repetition_penalty_window",
            "max_concurrent",
            "sox_bin_dir",
            "use_compile",
            "compile_mode",
            "compile_use_cuda_graphs",
            "compile_codebook_predictor",
            "preload_on_start",
            "warmup_on_start",
            "warmup_text",
            "warmup_emit_every_frames",
            "warmup_decode_window_frames",
            "warmup_max_new_tokens",
            "warmup_speaker",
            "warmup_language",
            "warmup_chunks",
            "first_chunk_timeout_s",
            "stream_idle_timeout_s",
        ):
            # Keep explicit nested process.* as source of truth.
            # Top-level fields are only backward-compatible defaults.
            if key in config_d and key not in process_cfg:
                process_cfg[key] = config_d[key]
        self.process_config = process_cfg
        if self._runner is not None:
            self._runner.set_process_config(self.process_config)

        if "port" in process_cfg and "base_url" not in config_d:
            self.base_url = f"http://127.0.0.1:{int(process_cfg['port'])}"

        assert self.mode in ("sidecar", "local")
        assert self.provider in ("cpu", "cuda")
        assert self.gpu_id >= 0
        assert self.request_timeout_s > 0
        assert self.connect_timeout_s > 0
        assert self.sidecar_read_chunk_bytes > 0
        assert self.sample_rate > 0
        assert self.sample_width in (1, 2, 4)
        assert self.channels > 0
        assert self.emit_every_frames > 0
        assert self.decode_window_frames > 0
        assert self.first_chunk_decode_window > 0
        assert self.first_chunk_frames >= 0
        assert self.overlap_samples >= 0
        assert self.max_new_tokens > 0
        assert self.top_p > 0 and self.top_p <= 1.0
        assert self.top_k >= 0
        assert self.temperature > 0
        assert self.process_startup_retries > 0
        assert self.process_startup_backoff_s > 0

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "base_url": self.base_url,
            "stream_endpoint": self.stream_endpoint,
            "health_endpoint": self.health_endpoint,
            "request_timeout_s": self.request_timeout_s,
            "connect_timeout_s": self.connect_timeout_s,
            "sidecar_read_chunk_bytes": self.sidecar_read_chunk_bytes,
            "sample_rate": self.sample_rate,
            "sample_width": self.sample_width,
            "channels": self.channels,
            "model_id": self.model_id,
            "language": self.language,
            "speaker": self.speaker,
            "provider": self.provider,
            "gpu_id": self.gpu_id,
            "device": self.device,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "emit_every_frames": self.emit_every_frames,
            "decode_window_frames": self.decode_window_frames,
            "first_chunk_emit_every": self.first_chunk_emit_every,
            "first_chunk_decode_window": self.first_chunk_decode_window,
            "first_chunk_frames": self.first_chunk_frames,
            "overlap_samples": self.overlap_samples,
            "repetition_penalty": self.repetition_penalty,
            "repetition_penalty_window": self.repetition_penalty_window,
            "instruct_prefix": self.instruct_prefix,
            "emotion_map": self.emotion_map,
            "local_preload": self.local_preload,
            "process_autostart": self.process_autostart,
            "process_startup_retries": self.process_startup_retries,
            "process_startup_backoff_s": self.process_startup_backoff_s,
            "process": self.process_config,
        }

    async def _parse_chunk(self, chunk_in: Dict[str, Any]) -> Dict[str, Any]:
        parsed = await super()._parse_chunk(chunk_in)
        # Ensure "emotion" exists as explicit key for downstream logic.
        parsed.setdefault("emotion", None)
        return parsed

    async def _ensure_sidecar_ready(self) -> None:
        url = f"{self.base_url}{self.health_endpoint}"
        timeout = httpx.Timeout(timeout=self.request_timeout_s, connect=self.connect_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise RuntimeError(f"Qwen3 TTS sidecar health check failed: {response.status_code} at {url}")
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("ready") is False:
                    detail = payload.get("detail") or payload.get("error") or "sidecar reported not ready"
                    raise RuntimeError(f"Qwen3 TTS sidecar not ready: {detail}")
            except ValueError:
                # Non-JSON health response is acceptable as long as status is 200.
                return

    async def _ensure_local_model_loaded(self):
        if self._local_model is not None:
            return

        def _load():
            import torch
            from qwen_tts import Qwen3TTSModel

            dtype = torch.bfloat16
            dtype_name = self.dtype.strip().lower()
            if dtype_name in ("float16", "fp16"):
                dtype = torch.float16
            elif dtype_name in ("float32", "fp32"):
                dtype = torch.float32

            model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=self.device,
                attn_implementation=(self.attn_implementation or None),
            )

            # Optimization hooks are optional across forks.
            if hasattr(model, "enable_streaming_optimizations"):
                try:
                    model.enable_streaming_optimizations(
                        decode_window_frames=self.decode_window_frames,
                        use_compile=True,
                        compile_mode="reduce-overhead",
                    )
                except Exception as exc:
                    logging.warning("Qwen3 local optimize hook failed: %s", exc)
            return model

        self._local_model = await asyncio.to_thread(_load)
        logging.info("Qwen3 TTS local model loaded: %s", self.model_id)

    def _split_emotion_tag(self, content: str) -> tuple[str, Optional[str]]:
        text = str(content or "").strip()
        if not text:
            return "", None
        match = _INLINE_EMOTION_RE.match(text)
        if not match:
            return text, None
        emotion_tag = match.group(1).strip()
        stripped_text = match.group(2).strip()
        return stripped_text, emotion_tag if emotion_tag else None

    def _build_instruct(self, emotion: Optional[str], inline_emotion: Optional[str]) -> str:
        parts = []
        if self.instruct_prefix:
            parts.append(self.instruct_prefix)

        merged_emotion = inline_emotion or (str(emotion).strip() if emotion else None)
        if merged_emotion:
            key = merged_emotion.lower()
            mapped = self.emotion_map.get(key)
            if mapped:
                parts.append(mapped)
            else:
                parts.append(f"Говори {merged_emotion}.")

        return " ".join([p.strip() for p in parts if p and p.strip()]).strip()

    def _build_stream_kwargs(self, content: str, instruct: str) -> Dict[str, Any]:
        kwargs = {
            "text": content,
            "language": self.language,
            "emit_every_frames": self.emit_every_frames,
            "decode_window_frames": self.decode_window_frames,
            "overlap_samples": self.overlap_samples,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "repetition_penalty_window": self.repetition_penalty_window,
            "do_sample": bool(self.do_sample),
        }
        if self.do_sample:
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k
            kwargs["temperature"] = self.temperature
        if self.speaker:
            kwargs["speaker"] = self.speaker
        if instruct:
            kwargs["instruct"] = instruct
        if self.first_chunk_emit_every > 0:
            kwargs["first_chunk_emit_every"] = self.first_chunk_emit_every
            kwargs["first_chunk_decode_window"] = self.first_chunk_decode_window
            kwargs["first_chunk_frames"] = self.first_chunk_frames
        return kwargs

    async def _stream_local_bytes(self, content: str, instruct: str):
        await self._ensure_local_model_loaded()
        model = self._local_model
        stream_kwargs = self._build_stream_kwargs(content=content, instruct=instruct)

        queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        loop = asyncio.get_running_loop()
        sentinel = object()

        def _worker():
            try:
                stream_fn = getattr(model, "stream_generate_custom_voice", None)
                if stream_fn is None:
                    raise RuntimeError("Qwen3 local model does not expose stream_generate_custom_voice().")

                def _iterate_with_fallback(kwargs: Dict[str, Any]) -> Iterable:
                    try:
                        return stream_fn(**kwargs)
                    except TypeError:
                        stripped = dict(kwargs)
                        for k in (
                            "first_chunk_emit_every",
                            "first_chunk_decode_window",
                            "first_chunk_frames",
                            "overlap_samples",
                            "repetition_penalty",
                            "repetition_penalty_window",
                        ):
                            stripped.pop(k, None)
                        return stream_fn(**stripped)

                for chunk, sr in _iterate_with_fallback(stream_kwargs):
                    pcm_bytes = _float_to_pcm16_bytes(chunk)
                    if pcm_bytes:
                        loop.call_soon_threadsafe(queue.put_nowait, {"audio_bytes": pcm_bytes, "sr": int(sr)})
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, {"error": exc})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, dict) and "error" in item:
                raise RuntimeError(f"Qwen3 local streaming failed: {item['error']}")
            yield item

    async def _stream_sidecar_bytes(self, content: str, instruct: str):
        req_device = self.device
        if self.provider.startswith("cpu"):
            req_device = "cpu"
        elif self.process_autostart and isinstance(self.process_config, dict):
            proc_provider = str(self.process_config.get("provider", self.provider)).strip().lower()
            if proc_provider.startswith("cuda"):
                proc_device = str(self.process_config.get("device", "")).strip()
                req_device = proc_device or "cuda:0"

        payload = {
            "text": content,
            "instruct": instruct,
            "language": self.language,
            "speaker": self.speaker,
            "provider": self.provider,
            "gpu_id": self.gpu_id,
            "model_id": self.model_id,
            "device": req_device,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(self.do_sample),
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "emit_every_frames": self.emit_every_frames,
            "decode_window_frames": self.decode_window_frames,
            "overlap_samples": self.overlap_samples,
            "first_chunk_emit_every": self.first_chunk_emit_every,
            "first_chunk_decode_window": self.first_chunk_decode_window,
            "first_chunk_frames": self.first_chunk_frames,
            "repetition_penalty": self.repetition_penalty,
            "repetition_penalty_window": self.repetition_penalty_window,
            "sample_rate": self.sample_rate,
            "sample_width": self.sample_width,
            "channels": self.channels,
        }
        url = f"{self.base_url}{self.stream_endpoint}"
        timeout = httpx.Timeout(timeout=self.request_timeout_s, connect=self.connect_timeout_s)
        max_attempts = 2 if (self.process_autostart and self._runner is not None) else 1
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            emitted_chunks = 0
            emitted_bytes = 0
            started_at = time.perf_counter()
            first_chunk_ms: Optional[int] = None
            last_chunk_at: Optional[float] = None
            max_gap_ms = 0.0
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code != 200:
                            message = (await response.aread()).decode("utf-8", errors="ignore")
                            raise RuntimeError(f"Qwen3 sidecar stream failed ({response.status_code}): {message}")
                        for_header_sr = response.headers.get("x-sample-rate")
                        if for_header_sr:
                            try:
                                self.sample_rate = int(for_header_sr)
                            except Exception:
                                pass

                        async for raw_chunk in response.aiter_bytes(chunk_size=self.sidecar_read_chunk_bytes):
                            if raw_chunk:
                                now = time.perf_counter()
                                if last_chunk_at is not None:
                                    gap_ms = (now - last_chunk_at) * 1000.0
                                    if gap_ms > max_gap_ms:
                                        max_gap_ms = gap_ms
                                else:
                                    first_chunk_ms = int((now - started_at) * 1000)
                                last_chunk_at = now
                                emitted_chunks += 1
                                emitted_bytes += len(raw_chunk)
                                yield {
                                    "audio_bytes": bytes(raw_chunk),
                                    "sr": int(self.sample_rate),
                                }

                if emitted_chunks <= 0:
                    raise RuntimeError("Qwen3 sidecar returned 200 but produced no audio bytes.")
                total_ms = int((time.perf_counter() - started_at) * 1000)
                audio_s = float(emitted_bytes) / float(max(1, int(self.sample_rate) * int(self.sample_width) * int(self.channels)))
                total_s = float(total_ms) / 1000.0
                rtf = (total_s / audio_s) if audio_s > 0 else -1.0
                logging.info(
                    "Qwen3 stream stats: chunks=%s bytes=%s first_chunk_ms=%s max_gap_ms=%.1f total_ms=%s audio_s=%.3f rtf=%.3f text_len=%s",
                    emitted_chunks,
                    emitted_bytes,
                    (first_chunk_ms if first_chunk_ms is not None else -1),
                    max_gap_ms,
                    total_ms,
                    audio_s,
                    rtf,
                    len(content),
                )
                return
            except Exception as exc:
                last_exc = exc
                can_retry = attempt < max_attempts and emitted_chunks == 0
                if not can_retry:
                    break
                logging.warning(
                    "Qwen3 sidecar request failed before first chunk (attempt %s/%s): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                try:
                    await self._runner.ensure_healthy()
                    health = await self._runner.health()
                    if health.get("port"):
                        self.base_url = f"http://127.0.0.1:{int(health['port'])}"
                        url = f"{self.base_url}{self.stream_endpoint}"
                    await asyncio.sleep(0.2)
                except Exception as restart_exc:
                    logging.warning("Qwen3 sidecar auto-recovery failed: %s", restart_exc)
                    break

        raise RuntimeError(f"Qwen3 sidecar stream failed: {last_exc}")

    async def _generate(
        self,
        content: str = None,
        emotion: str = None,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        speaker_id: str = None,
        input_timestamp_ms: int = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        started = time.perf_counter()
        clean_text, inline_emotion = self._split_emotion_tag(content)
        if not clean_text:
            return

        instruct = self._build_instruct(emotion=emotion, inline_emotion=inline_emotion)
        stream_iter = self._stream_sidecar_bytes if self.mode == "sidecar" else self._stream_local_bytes

        first_chunk = True
        async for out in stream_iter(clean_text, instruct):
            chunk = {
                "audio_bytes": out["audio_bytes"],
                "sr": int(out.get("sr", self.sample_rate)),
                "sw": int(self.sample_width),
                "ch": int(self.channels),
                "source_id": source_id,
                "turn_id": turn_id,
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "input_timestamp_ms": input_timestamp_ms,
                "provider": self.op_id,
            }
            if first_chunk:
                first_chunk = False
                chunk["tts_provider_latency_ms"] = int((time.perf_counter() - started) * 1000)
            yield chunk
