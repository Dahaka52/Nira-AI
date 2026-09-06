import asyncio
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict

import librosa
import numpy as np
import sherpa_onnx
from sherpa_onnx.offline_recognizer import (
    FeatureExtractorConfig,
    HomophoneReplacerConfig,
    OfflineModelConfig,
    OfflineNemoEncDecCtcModelConfig,
    OfflineRecognizer,
    OfflineRecognizerConfig,
    _Recognizer,
)

from .base import STTOperation


class GigaAMSTT(STTOperation):
    """
    Offline Speech-to-Text operation powered by Sber GigaAM-v3 (NeMo CTC) via sherpa-onnx.
    Features:
      - Ultra-fast (<50ms) inference on CPU (RTF ~0.02)
      - Native Russian punctuation and capitalization
      - 0 YouTube subtitle hallucinations
      - Automatic Gain Normalization (AGC) for quiet speech
      - Hotwords boost (hotwords_file, hotwords_score)
      - CTC blank penalty adjustment
      - Alias/Homophone auto-replacement for character/user names
    """

    def __init__(self):
        super().__init__("gigaam")
        self.recognizer: OfflineRecognizer | None = None
        self.model_path = os.path.abspath(os.path.join(os.getcwd(), "models", "gigaam", "gigaam_v3_e2e_ctc_int8.onnx"))
        self.tokens_path = os.path.abspath(os.path.join(os.getcwd(), "models", "gigaam", "gigaam_v3_e2e_ctc_tokens.txt"))
        self.num_threads = 4
        self.sample_rate = 16000
        self.feature_dim = 64
        self.decoding_method = "greedy_search"
        self.provider = "cpu"
        self.debug = False

        # Tuning knobs
        self.blank_penalty = 0.0
        self.hotwords_file = ""
        self.hotwords_score = 2.0
        self.gain_normalization = True
        self.target_peak = 0.95
        self.max_gain = 4.0
        self.alias_map = {
            "мира": "Нира",
            "миру": "Ниру",
            "миром": "Нирой",
            "мире": "Нире",
            "миры": "Ниры",
        }

    async def configure(self, config_d: Dict[str, Any]):
        if "model_path" in config_d:
            self.model_path = os.path.abspath(str(config_d["model_path"]))
        if "tokens_path" in config_d:
            self.tokens_path = os.path.abspath(str(config_d["tokens_path"]))
        if "num_threads" in config_d:
            self.num_threads = int(config_d["num_threads"])
        if "sample_rate" in config_d:
            self.sample_rate = int(config_d["sample_rate"])
        if "feature_dim" in config_d:
            self.feature_dim = int(config_d["feature_dim"])
        if "decoding_method" in config_d:
            self.decoding_method = str(config_d["decoding_method"])
        if "provider" in config_d:
            self.provider = str(config_d["provider"]).lower()
        if "debug" in config_d:
            self.debug = bool(config_d["debug"])

        # Fine-tuning knobs
        if "blank_penalty" in config_d:
            self.blank_penalty = float(config_d["blank_penalty"])
        if "hotwords_file" in config_d and config_d["hotwords_file"]:
            self.hotwords_file = os.path.abspath(str(config_d["hotwords_file"]))
        if "hotwords_score" in config_d:
            self.hotwords_score = float(config_d["hotwords_score"])
        if "gain_normalization" in config_d:
            self.gain_normalization = bool(config_d["gain_normalization"])
        if "target_peak" in config_d:
            self.target_peak = float(config_d["target_peak"])
        if "max_gain" in config_d:
            self.max_gain = float(config_d["max_gain"])
        if "alias_map" in config_d and isinstance(config_d["alias_map"], dict):
            self.alias_map.update(config_d["alias_map"])

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "tokens_path": self.tokens_path,
            "num_threads": self.num_threads,
            "sample_rate": self.sample_rate,
            "feature_dim": self.feature_dim,
            "decoding_method": self.decoding_method,
            "provider": self.provider,
            "debug": self.debug,
            "blank_penalty": self.blank_penalty,
            "hotwords_file": self.hotwords_file,
            "hotwords_score": self.hotwords_score,
            "gain_normalization": self.gain_normalization,
            "target_peak": self.target_peak,
            "max_gain": self.max_gain,
            "alias_map": self.alias_map,
        }

    async def start(self) -> None:
        await super().start()

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"GigaAM ONNX model file not found: {self.model_path}")
        if not os.path.isfile(self.tokens_path):
            raise FileNotFoundError(f"GigaAM tokens file not found: {self.tokens_path}")

        logging.info(
            f"Initializing GigaAM STT (sherpa-onnx NeMo CTC): model={self.model_path}, "
            f"threads={self.num_threads}, provider={self.provider}, blank_penalty={self.blank_penalty}, "
            f"hotwords={'enabled' if self.hotwords_file and os.path.isfile(self.hotwords_file) else 'none'}"
        )

        def _init_recognizer():
            model_config = OfflineModelConfig(
                nemo_ctc=OfflineNemoEncDecCtcModelConfig(model=self.model_path),
                tokens=self.tokens_path,
                num_threads=self.num_threads,
                debug=self.debug,
                provider=self.provider,
                model_type="nemo_ctc",
            )
            feat_config = FeatureExtractorConfig(
                sampling_rate=self.sample_rate,
                feature_dim=self.feature_dim,
            )
            recognizer_config = OfflineRecognizerConfig(
                feat_config=feat_config,
                model_config=model_config,
                decoding_method=self.decoding_method,
                hotwords_file=self.hotwords_file if (self.hotwords_file and os.path.isfile(self.hotwords_file)) else "",
                hotwords_score=self.hotwords_score,
                blank_penalty=self.blank_penalty,
            )
            rec = OfflineRecognizer.__new__(OfflineRecognizer)
            rec.recognizer = _Recognizer(recognizer_config)
            rec.config = recognizer_config

            # Warm up inference graph with a short dummy stream
            warmup_stream = rec.create_stream()
            warmup_stream.accept_waveform(16000, np.zeros(1600, dtype=np.float32))
            rec.decode_stream(warmup_stream)
            return rec

        loop = asyncio.get_running_loop()
        self.recognizer = await loop.run_in_executor(None, _init_recognizer)
        logging.info("GigaAM STT initialized and warmed up successfully.")

    async def close(self) -> None:
        self.recognizer = None
        await super().close()

    def _apply_aliases(self, text: str) -> str:
        if not text or not self.alias_map:
            return text
        result = text
        for src, dst in self.alias_map.items():
            pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
            result = pattern.sub(dst, result)
        return result

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
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not audio_bytes or self.recognizer is None:
            yield {"text": "", "is_final": True}
            return

        # 1. Read raw PCM bytes
        if sw == 2:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_np.astype(np.float32) / 32768.0
        elif sw == 4:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int32)
            audio_float32 = audio_np.astype(np.float32) / 2147483648.0
        else:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_np.astype(np.float32) / 32768.0

        # 2. Downmix multi-channel to mono
        if ch and ch > 1:
            audio_float32 = audio_float32.reshape(-1, ch).mean(axis=1)

        # 3. Resample to 16 kHz if necessary
        input_sr = sr or 16000
        if input_sr != 16000:
            audio_float32 = librosa.resample(audio_float32, orig_sr=input_sr, target_sr=16000)

        # 4. Gain Normalization (AGC) for quiet speech
        if self.gain_normalization and len(audio_float32) > 0:
            max_val = float(np.max(np.abs(audio_float32)))
            if max_val > 1e-4:
                gain = min(self.max_gain, self.target_peak / max_val)
                audio_float32 = audio_float32 * gain

        audio_float32 = np.ascontiguousarray(audio_float32, dtype=np.float32)

        # 5. Run inference in thread pool
        def _transcribe() -> str:
            t0 = time.perf_counter()
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio_float32)
            self.recognizer.decode_stream(stream)
            res_text = stream.result.text.strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Post-processing: alias / homophone replacement
            if res_text:
                res_text = self._apply_aliases(res_text)

            logging.debug(f"[GigaAM STT] Transcribed in {elapsed_ms:.1f}ms: '{res_text}'")
            return res_text

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, _transcribe)
            yield {"text": text, "is_final": True}
        except Exception as e:
            logging.error(f"GigaAM STT transcription error: {e}", exc_info=True)
            yield {"text": "", "is_final": True}
