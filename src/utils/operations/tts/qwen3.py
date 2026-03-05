import asyncio
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from utils.processes import ProcessType, TTSProcessRunner

from .base import TTSOperation


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on", "y"}:
        return True
    if s in {"0", "false", "no", "off", "n", ""}:
        return False
    return default


class Qwen3TTS(TTSOperation):
    """
    Qwen3 TTS provider bound to dffdeeq voice_clone sidecar runtime.
    """

    def __init__(self):
        super().__init__("qwen3")

        self.base_url = "http://127.0.0.1:6116"
        self.stream_endpoint = "/v1/tts/stream"
        self.health_endpoint = "/health"
        self.request_timeout_s = 60.0
        self.connect_timeout_s = 10.0
        self.sidecar_read_chunk_bytes = 8192

        self.sample_rate = 24000
        self.sample_width = 2
        self.channels = 1

        self.runtime_flavor = "dffdeeq"
        self.voice_mode = "voice_clone"
        self.ref_audio_path = ""
        self.ref_text = ""
        self.x_vector_only_mode = True

        self.model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        self.language = "russian"
        self.provider = "cuda"
        self.gpu_id = 1
        self.device = "cuda:0"
        self.dtype = "bfloat16"
        self.attn_implementation = "flash_attention_2"

        self.emit_every_frames = 4
        self.decode_window_frames = 80
        self.overlap_samples = 0
        self.max_frames = 10000
        self.use_optimized_decode = True

        self.do_sample = False
        self.top_p = 0.9
        self.top_k = 50
        self.temperature = 0.8

        self.max_text_chars = 800
        self.dynamic_max_frames = False
        self.dynamic_chars_per_second = 11.5
        self.dynamic_frame_budget_mul = 1.05
        self.dynamic_min_frames = 20
        self.dynamic_max_frames_cap = 10000

        self.process_autostart = True
        self.process_startup_retries = 3
        self.process_startup_backoff_s = 0.6
        self.process_config: Dict[str, Any] = {}
        self._runner: Optional[TTSProcessRunner] = None

    async def start(self) -> None:
        await super().start()

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

    async def close(self) -> None:
        await super().close()
        if self._runner is not None:
            await self._runner.close()
            self._runner = None

    async def configure(self, config_d: Dict[str, Any]):
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

        if "runtime_flavor" in config_d:
            self.runtime_flavor = str(config_d["runtime_flavor"]).strip().lower().replace("-", "_")
        if "voice_mode" in config_d:
            self.voice_mode = str(config_d["voice_mode"]).strip().lower().replace("-", "_")
        if "ref_audio_path" in config_d:
            self.ref_audio_path = str(config_d["ref_audio_path"]).strip()
        if "ref_text" in config_d:
            self.ref_text = str(config_d["ref_text"]).strip()
        if "x_vector_only_mode" in config_d:
            self.x_vector_only_mode = _as_bool(config_d["x_vector_only_mode"], default=True)

        if "model_id" in config_d:
            self.model_id = str(config_d["model_id"])
        if "language" in config_d:
            self.language = str(config_d["language"])
        if "provider" in config_d:
            self.provider = str(config_d["provider"]).strip().lower()
        if "gpu_id" in config_d:
            self.gpu_id = int(config_d["gpu_id"])
        if "device" in config_d:
            self.device = str(config_d["device"]).strip()
        if "dtype" in config_d:
            self.dtype = str(config_d["dtype"])
        if "attn_implementation" in config_d:
            self.attn_implementation = str(config_d["attn_implementation"]).strip()

        if "emit_every_frames" in config_d:
            self.emit_every_frames = int(config_d["emit_every_frames"])
        if "decode_window_frames" in config_d:
            self.decode_window_frames = int(config_d["decode_window_frames"])
        if "overlap_samples" in config_d:
            self.overlap_samples = int(config_d["overlap_samples"])
        if "max_frames" in config_d:
            self.max_frames = int(config_d["max_frames"])
        if "use_optimized_decode" in config_d:
            self.use_optimized_decode = _as_bool(config_d["use_optimized_decode"], default=True)

        if "do_sample" in config_d:
            self.do_sample = _as_bool(config_d["do_sample"], default=False)
        if "top_p" in config_d:
            self.top_p = float(config_d["top_p"])
        if "top_k" in config_d:
            self.top_k = int(config_d["top_k"])
        if "temperature" in config_d:
            self.temperature = float(config_d["temperature"])

        if "max_text_chars" in config_d:
            self.max_text_chars = int(config_d["max_text_chars"])
        if "dynamic_max_frames" in config_d:
            self.dynamic_max_frames = _as_bool(config_d["dynamic_max_frames"], default=True)
        if "dynamic_chars_per_second" in config_d:
            self.dynamic_chars_per_second = float(config_d["dynamic_chars_per_second"])
        if "dynamic_frame_budget_mul" in config_d:
            self.dynamic_frame_budget_mul = float(config_d["dynamic_frame_budget_mul"])
        if "dynamic_min_frames" in config_d:
            self.dynamic_min_frames = int(config_d["dynamic_min_frames"])
        if "dynamic_max_frames_cap" in config_d:
            self.dynamic_max_frames_cap = int(config_d["dynamic_max_frames_cap"])

        if "process_autostart" in config_d:
            self.process_autostart = _as_bool(config_d["process_autostart"], default=True)
        if "process_startup_retries" in config_d:
            self.process_startup_retries = int(config_d["process_startup_retries"])
        if "process_startup_backoff_s" in config_d:
            self.process_startup_backoff_s = float(config_d["process_startup_backoff_s"])

        process_cfg = {}
        if isinstance(config_d.get("process"), dict):
            process_cfg.update(config_d["process"])
        for key in (
            "host",
            "port",
            "python_executable",
            "model_id",
            "language",
            "provider",
            "gpu_id",
            "device",
            "dtype",
            "attn_implementation",
            "sample_rate",
            "sample_width",
            "channels",
            "runtime_flavor",
            "voice_mode",
            "qwen_tts_repo_path",
            "ref_audio_path",
            "ref_text",
            "x_vector_only_mode",
            "emit_every_frames",
            "decode_window_frames",
            "overlap_samples",
            "max_frames",
            "use_optimized_decode",
            "do_sample",
            "top_p",
            "top_k",
            "temperature",
            "dynamic_max_frames",
            "dynamic_chars_per_second",
            "dynamic_frame_budget_mul",
            "dynamic_min_frames",
            "dynamic_max_frames_cap",
            "max_concurrent",
            "use_compile",
            "compile_mode",
            "compile_use_cuda_graphs",
            "compile_codebook_predictor",
            "compile_talker",
            "compile_cudagraph_skip_dynamic_graphs",
            "preload_on_start",
            "warmup_on_start",
            "warmup_text",
            "warmup_chunks",
            "startup_health_timeout_s",
            "first_chunk_timeout_s",
            "stream_idle_timeout_s",
            "sox_bin_dir",
        ):
            if key in config_d and key not in process_cfg:
                process_cfg[key] = config_d[key]

        if "runtime_flavor" in process_cfg and "runtime_flavor" not in config_d:
            self.runtime_flavor = str(process_cfg["runtime_flavor"]).strip().lower().replace("-", "_")
        if "voice_mode" in process_cfg and "voice_mode" not in config_d:
            self.voice_mode = str(process_cfg["voice_mode"]).strip().lower().replace("-", "_")

        self.process_config = process_cfg
        if self._runner is not None:
            self._runner.set_process_config(self.process_config)

        if "port" in process_cfg and "base_url" not in config_d:
            self.base_url = f"http://127.0.0.1:{int(process_cfg['port'])}"

        if self.runtime_flavor in {"auto", "original", ""}:
            self.runtime_flavor = "dffdeeq"

        assert self.runtime_flavor == "dffdeeq"
        assert self.voice_mode in {"voice_clone", "clone"}
        assert self.provider in {"cpu", "cuda"}
        assert self.gpu_id >= 0
        assert self.request_timeout_s > 0
        assert self.connect_timeout_s > 0
        assert self.sidecar_read_chunk_bytes >= 128
        assert self.sample_rate > 0
        assert self.sample_width in {1, 2, 4}
        assert self.channels > 0
        assert self.emit_every_frames > 0
        assert self.decode_window_frames > 0
        assert self.max_frames > 0
        assert self.top_p > 0 and self.top_p <= 1.0
        assert self.top_k >= 0
        assert self.temperature > 0
        assert self.max_text_chars >= 16
        assert self.dynamic_chars_per_second > 0
        assert self.dynamic_frame_budget_mul > 0
        assert self.dynamic_min_frames > 0
        assert self.dynamic_max_frames_cap >= self.dynamic_min_frames
        assert self.process_startup_retries > 0
        assert self.process_startup_backoff_s > 0

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "stream_endpoint": self.stream_endpoint,
            "health_endpoint": self.health_endpoint,
            "request_timeout_s": self.request_timeout_s,
            "connect_timeout_s": self.connect_timeout_s,
            "sidecar_read_chunk_bytes": self.sidecar_read_chunk_bytes,
            "sample_rate": self.sample_rate,
            "sample_width": self.sample_width,
            "channels": self.channels,
            "runtime_flavor": self.runtime_flavor,
            "voice_mode": self.voice_mode,
            "ref_audio_path": self.ref_audio_path,
            "ref_text": self.ref_text,
            "x_vector_only_mode": self.x_vector_only_mode,
            "model_id": self.model_id,
            "language": self.language,
            "provider": self.provider,
            "gpu_id": self.gpu_id,
            "device": self.device,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "emit_every_frames": self.emit_every_frames,
            "decode_window_frames": self.decode_window_frames,
            "overlap_samples": self.overlap_samples,
            "max_frames": self.max_frames,
            "use_optimized_decode": self.use_optimized_decode,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_text_chars": self.max_text_chars,
            "dynamic_max_frames": self.dynamic_max_frames,
            "dynamic_chars_per_second": self.dynamic_chars_per_second,
            "dynamic_frame_budget_mul": self.dynamic_frame_budget_mul,
            "dynamic_min_frames": self.dynamic_min_frames,
            "dynamic_max_frames_cap": self.dynamic_max_frames_cap,
            "process_autostart": self.process_autostart,
            "process_startup_retries": self.process_startup_retries,
            "process_startup_backoff_s": self.process_startup_backoff_s,
            "process": self.process_config,
        }

    async def _parse_chunk(self, chunk_in: Dict[str, Any]) -> Dict[str, Any]:
        parsed = await super()._parse_chunk(chunk_in)
        parsed.setdefault("source_id", None)
        parsed.setdefault("turn_id", None)
        parsed.setdefault("utterance_id", None)
        parsed.setdefault("speaker_id", None)
        parsed.setdefault("input_timestamp_ms", None)
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
                return

    def _effective_max_frames(self, text: str) -> int:
        limit = max(16, int(self.max_frames))
        if not self.dynamic_max_frames:
            return limit

        norm = re.sub(r"\s+", " ", str(text or "")).strip()
        text_len = len(norm)
        punct = sum(1 for ch in norm if ch in ".!?;:")

        est_seconds = (float(text_len) / float(self.dynamic_chars_per_second)) + (0.08 * punct) + 0.35
        est_frames = int(est_seconds * 12.0 * float(self.dynamic_frame_budget_mul))
        est_frames = max(int(self.dynamic_min_frames), est_frames)
        est_frames = min(int(self.dynamic_max_frames_cap), est_frames)
        return max(16, min(limit, est_frames))

    async def _stream_sidecar_bytes(self, text: str):
        payload = {
            "text": text,
            "runtime_flavor": self.runtime_flavor,
            "voice_mode": self.voice_mode,
            "ref_audio_path": self.ref_audio_path,
            "ref_text": self.ref_text,
            "x_vector_only_mode": bool(self.x_vector_only_mode),
            "language": self.language,
            "provider": self.provider,
            "gpu_id": self.gpu_id,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "sample_rate": self.sample_rate,
            "sample_width": self.sample_width,
            "channels": self.channels,
            "emit_every_frames": self.emit_every_frames,
            "decode_window_frames": self.decode_window_frames,
            "overlap_samples": self.overlap_samples,
            "max_frames": self._effective_max_frames(text),
            "use_optimized_decode": bool(self.use_optimized_decode),
            "do_sample": bool(self.do_sample),
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
        }

        url = f"{self.base_url}{self.stream_endpoint}"
        timeout = httpx.Timeout(timeout=self.request_timeout_s, connect=self.connect_timeout_s)

        max_attempts = 3 if (self.process_autostart and self._runner is not None) else 1
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            emitted_chunks = 0
            emitted_bytes = 0
            started_at = time.perf_counter()
            first_chunk_ms: Optional[int] = None
            last_chunk_at: Optional[float] = None
            max_gap_ms = 0.0
            trailing = b""
            pending_chunk: Optional[Dict[str, Any]] = None

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code != 200:
                            message = (await response.aread()).decode("utf-8", errors="ignore")
                            raise RuntimeError(f"Qwen3 sidecar stream failed ({response.status_code}): {message}")

                        header_sr = response.headers.get("x-sample-rate")
                        if header_sr:
                            try:
                                self.sample_rate = int(header_sr)
                            except Exception:
                                pass

                        async for raw_chunk in response.aiter_bytes(chunk_size=self.sidecar_read_chunk_bytes):
                            if not raw_chunk:
                                continue

                            if trailing:
                                raw_chunk = trailing + raw_chunk
                                trailing = b""

                            if len(raw_chunk) % int(self.sample_width):
                                aligned_len = len(raw_chunk) - (len(raw_chunk) % int(self.sample_width))
                                trailing = raw_chunk[aligned_len:]
                                raw_chunk = raw_chunk[:aligned_len]

                            if not raw_chunk:
                                continue

                            now = time.perf_counter()
                            if last_chunk_at is None:
                                first_chunk_ms = int((now - started_at) * 1000)
                            else:
                                gap_ms = (now - last_chunk_at) * 1000.0
                                if gap_ms > max_gap_ms:
                                    max_gap_ms = gap_ms
                            last_chunk_at = now

                            emitted_chunks += 1
                            emitted_bytes += len(raw_chunk)
                            if pending_chunk is not None:
                                yield pending_chunk
                            pending_chunk = {"audio_bytes": bytes(raw_chunk), "sr": int(self.sample_rate)}

                        if trailing:
                            logging.warning("Qwen3 sidecar dropped trailing unaligned bytes: %s", len(trailing))

                if emitted_chunks <= 0:
                    raise RuntimeError("Qwen3 sidecar returned 200 but produced no audio bytes.")

                total_ms = int((time.perf_counter() - started_at) * 1000)
                audio_s = float(emitted_bytes) / float(max(1, int(self.sample_rate) * int(self.sample_width) * int(self.channels)))
                total_s = float(total_ms) / 1000.0
                rtf = (total_s / audio_s) if audio_s > 0 else -1.0
                stream_stats = {
                    "tts_chunks": int(emitted_chunks),
                    "tts_total_ms": int(total_ms),
                    "tts_audio_s": float(audio_s),
                    "tts_rtf": float(rtf),
                    "tts_first_chunk_ms": int(first_chunk_ms if first_chunk_ms is not None else -1),
                    "tts_max_gap_ms": float(max_gap_ms),
                    "tts_text_len": int(len(text)),
                }
                if pending_chunk is not None:
                    pending_chunk.update(stream_stats)
                    yield pending_chunk
                logging.info(
                    "Qwen3 stream stats: chunks=%s bytes=%s first_chunk_ms=%s max_gap_ms=%.1f total_ms=%s audio_s=%.3f rtf=%.3f text_len=%s",
                    emitted_chunks,
                    emitted_bytes,
                    (first_chunk_ms if first_chunk_ms is not None else -1),
                    max_gap_ms,
                    total_ms,
                    audio_s,
                    rtf,
                    len(text),
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
                    await asyncio.sleep(min(1.2, 0.35 * float(attempt)))
                except Exception as restart_exc:
                    logging.warning("Qwen3 sidecar auto-recovery failed: %s", restart_exc)
                    break

        raise RuntimeError(f"Qwen3 sidecar stream failed: {last_exc}")

    async def _generate(
        self,
        content: str = None,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        speaker_id: str = None,
        input_timestamp_ms: int = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        text = str(content or "").strip()
        if not text:
            return

        if len(text) > int(self.max_text_chars):
            text = text[: int(self.max_text_chars)].rstrip()

        started = time.perf_counter()
        first_chunk = True
        async for out in self._stream_sidecar_bytes(text):
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
            for key in (
                "tts_chunks",
                "tts_total_ms",
                "tts_audio_s",
                "tts_rtf",
                "tts_first_chunk_ms",
                "tts_max_gap_ms",
                "tts_text_len",
            ):
                if key in out:
                    chunk[key] = out[key]
            yield chunk
