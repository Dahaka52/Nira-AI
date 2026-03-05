import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict

import numpy as np
import torch
import websockets

from utils.processes import ProcessType, STTProcessRunner

from .base import STTOperation


_SHERPA_PRESETS = {
    "low_latency": {
        "model_variant": "int8",
        "decoding_method": "greedy_search",
        "num_active_paths": 2,
        "use_endpoint": 0,
    },
    "balanced": {
        "model_variant": "fp32",
        "decoding_method": "modified_beam_search",
        "num_active_paths": 4,
        "use_endpoint": 0,
    },
    "noisy_room": {
        "model_variant": "fp32",
        "decoding_method": "modified_beam_search",
        "num_active_paths": 8,
        "use_endpoint": 1,
        "hotwords_score": 3.5,
    },
}


class SherpaSTT(STTOperation):
    def __init__(self):
        super().__init__("sherpa")
        self.ws_url = "ws://127.0.0.1:6006"
        self.provider = "cpu"
        self.language = "ru"
        self.request_timeout_s = 20.0
        self.stream_chunk_ms = 200
        self.normalize_output = True

        self.process_autostart = True
        self.process_config: Dict[str, Any] = {}
        self._runner: STTProcessRunner | None = None
        self.process_startup_retries = 3
        self.process_startup_backoff_s = 0.5

        self.enable_te = False
        self.te_apply = None
        self.te_model_dir = os.path.abspath(os.path.join(os.getcwd(), "models", "silero_models"))

    async def start(self) -> None:
        await super().start()

        if self.process_autostart:
            self._runner = STTProcessRunner(
                link_id=f"stt_{self.op_id}",
                process_type=ProcessType.SHERPA,
                process_config=self.process_config,
                startup_retries=self.process_startup_retries,
                startup_backoff_s=self.process_startup_backoff_s,
            )
            await self._runner.ensure_healthy()
            health = await self._runner.health()
            if health.get("port"):
                self.ws_url = f"ws://127.0.0.1:{health['port']}"

        if self.enable_te and self.te_apply is None:
            asyncio.create_task(self._load_te_model())

    async def close(self) -> None:
        await super().close()
        if self._runner is not None:
            await self._runner.close()
            self._runner = None

    async def configure(self, config_d: Dict[str, Any]):
        if "ws_url" in config_d:
            self.ws_url = str(config_d["ws_url"])
        if "provider" in config_d:
            self.provider = str(config_d["provider"])
        if "language" in config_d:
            self.language = str(config_d["language"])
        if "request_timeout_s" in config_d:
            self.request_timeout_s = float(config_d["request_timeout_s"])
        if "stream_chunk_ms" in config_d:
            self.stream_chunk_ms = int(config_d["stream_chunk_ms"])
        if "normalize_output" in config_d:
            self.normalize_output = bool(config_d["normalize_output"])
        if "process_autostart" in config_d:
            self.process_autostart = bool(config_d["process_autostart"])
        if "process_startup_retries" in config_d:
            self.process_startup_retries = int(config_d["process_startup_retries"])
        if "process_startup_backoff_s" in config_d:
            self.process_startup_backoff_s = float(config_d["process_startup_backoff_s"])
        if "enable_te" in config_d:
            self.enable_te = bool(config_d["enable_te"])
        if "te_model_dir" in config_d:
            self.te_model_dir = os.path.abspath(str(config_d["te_model_dir"]))

        # Backward compatible process config fields from top-level STT config.
        top_level_process_fields = {
            "provider",
            "gpu_id",
            "model_dir",
            "model_variant",
            "decoding_method",
            "num_active_paths",
            "use_endpoint",
            "hotwords_file",
            "hotwords_score",
            "bpe_vocab",
            "encoder",
            "decoder",
            "joiner",
            "tokens",
            "port",
        }
        process_cfg = {}
        if isinstance(config_d.get("process"), dict):
            process_cfg.update(config_d["process"])
        for key in top_level_process_fields:
            if key in config_d:
                process_cfg[key] = config_d[key]

        process_cfg.setdefault("provider", self.provider)

        preset = str(process_cfg.get("preset", "balanced")).strip().lower()
        if preset in _SHERPA_PRESETS:
            for k, v in _SHERPA_PRESETS[preset].items():
                process_cfg.setdefault(k, v)

        self.process_config = process_cfg
        if self._runner is not None:
            self._runner.set_process_config(self.process_config)

        # If explicit port is set but ws_url isn't manually overridden, align ws_url.
        if "port" in process_cfg and "ws_url" not in config_d:
            try:
                self.ws_url = f"ws://127.0.0.1:{int(process_cfg['port'])}"
            except Exception:
                pass

        assert self.request_timeout_s > 0
        assert self.stream_chunk_ms > 0
        assert self.process_startup_retries > 0
        assert self.process_startup_backoff_s > 0

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "ws_url": self.ws_url,
            "provider": self.provider,
            "language": self.language,
            "request_timeout_s": self.request_timeout_s,
            "stream_chunk_ms": self.stream_chunk_ms,
            "normalize_output": self.normalize_output,
            "process_autostart": self.process_autostart,
            "process_startup_retries": self.process_startup_retries,
            "process_startup_backoff_s": self.process_startup_backoff_s,
            "process": self.process_config,
            "enable_te": self.enable_te,
            "te_model_dir": self.te_model_dir,
        }

    async def _load_te_model(self):
        try:
            logging.info("Sherpa STT: loading local Silero TE model...")
            weights_path = os.path.join(self.te_model_dir, "src", "silero", "model", "v2_4lang_q.pt")
            if not os.path.exists(weights_path):
                logging.warning("Sherpa STT: TE weights missing at %s. Punctuation disabled.", weights_path)
                return

            _, _, _, _, apply_te = await asyncio.to_thread(
                torch.hub.load,
                repo_or_dir=self.te_model_dir,
                model="silero_te",
                source="local",
                force_reload=False,
            )
            self.te_apply = apply_te
            logging.info("Sherpa STT: Silero TE loaded.")
        except Exception as e:
            logging.warning("Sherpa STT: failed to load TE model: %s", e)

    def _normalize_text(self, text: str) -> str:
        if not self.normalize_output:
            return text
        out = re.sub(r"\s+", " ", str(text or "").strip())
        return out

    async def _send_audio_to_ws(self, audio_bytes: bytes, sr: int, sw: int, ch: int) -> tuple[str, bool]:
        if sw != 2:
            raise ValueError(f"Sherpa STT expects 16-bit PCM input, got {sw * 8}-bit.")

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_np.astype(np.float32) / 32768.0
        if ch > 1:
            audio_float32 = audio_float32.reshape(-1, ch).mean(axis=1)

        # Keep transport chunk length aligned with incoming audio sample rate.
        chunk_samples = max(1, int((sr or 16000) * (self.stream_chunk_ms / 1000.0)))

        transcription = ""
        timed_out = False
        async with websockets.connect(self.ws_url, max_size=4 * 1024 * 1024) as websocket:
            for i in range(0, len(audio_float32), chunk_samples):
                await websocket.send(audio_float32[i:i + chunk_samples].tobytes())
            await websocket.send("Done")

            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=self.request_timeout_s)
                except asyncio.TimeoutError:
                    logging.warning("Sherpa STT: receive timeout from %s", self.ws_url)
                    timed_out = True
                    break
                except websockets.exceptions.ConnectionClosedOK:
                    break

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                text = str(data.get("text", "")).strip()
                if text:
                    transcription = text

        return transcription, timed_out

    async def _recover_sidecar(self) -> bool:
        if not self.process_autostart or self._runner is None:
            return False
        try:
            await self._runner.restart()
            health = await self._runner.health()
            if health.get("port"):
                self.ws_url = f"ws://127.0.0.1:{health['port']}"
            return bool(health.get("running"))
        except Exception as e:
            logging.error("Sherpa STT recovery failed: %s", e, exc_info=True)
            return False

    async def _generate(
        self,
        prompt: str = None,
        audio_bytes: bytes = None,
        sr: int = None,
        sw: int = None,
        ch: int = None,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        speaker_id: str = None,
        input_timestamp_ms: int = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        started = time.perf_counter()
        text = ""
        stt_error = None
        try:
            text, timed_out = await self._send_audio_to_ws(audio_bytes, sr, sw, ch)
            if timed_out and not text:
                stt_error = "timeout"
        except (
            OSError,
            websockets.exceptions.InvalidURI,
            websockets.exceptions.InvalidHandshake,
            websockets.exceptions.ConnectionClosedError,
        ) as e:
            logging.error("Sherpa STT connection error: %s", e)
            recovered = await self._recover_sidecar()
            if recovered:
                stt_error = "restarting"
                try:
                    text, timed_out = await self._send_audio_to_ws(audio_bytes, sr, sw, ch)
                    if timed_out and not text:
                        stt_error = "timeout"
                    elif text:
                        stt_error = None
                except Exception as retry_err:
                    logging.error("Sherpa STT retry after recovery failed: %s", retry_err)
                    stt_error = "unavailable"
            else:
                stt_error = "unavailable"
        except Exception as e:
            logging.error("Sherpa STT generation error: %s", e, exc_info=True)
            recovered = await self._recover_sidecar()
            if recovered:
                stt_error = "restarting"
                try:
                    text, timed_out = await self._send_audio_to_ws(audio_bytes, sr, sw, ch)
                    if timed_out and not text:
                        stt_error = "timeout"
                    elif text:
                        stt_error = None
                except Exception as retry_err:
                    logging.error("Sherpa STT retry after recovery failed: %s", retry_err)
                    stt_error = "unavailable"
            else:
                stt_error = "unavailable"

        if text and self.te_apply:
            try:
                text = self.te_apply(text, lan=self.language)
            except Exception as e:
                logging.warning("Sherpa STT: TE apply failed: %s", e)

        text = self._normalize_text(text)
        latency_ms = int((time.perf_counter() - started) * 1000)
        logging.info(
            "Sherpa STT final: len=%s latency_ms=%s source=%s turn=%s utterance=%s",
            len(text),
            latency_ms,
            source_id,
            turn_id,
            utterance_id,
        )

        yield {
            "text": text,
            "is_final": True,
            "confidence": None,
            "provider": self.op_id,
            "source_id": source_id,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "input_timestamp_ms": input_timestamp_ms,
            "stt_latency_ms": latency_ms,
            "stt_error": stt_error,
        }
