import time
import asyncio
import logging
from typing import Dict, Any, AsyncGenerator

import numpy as np
import librosa
from faster_whisper import WhisperModel

from .base import STTOperation

class FasterWhisperSTT(STTOperation):
    def __init__(self):
        super().__init__("faster_whisper")
        self.model = None

    async def configure(self, config_d: Dict[str, Any]):
        self.model_size = config_d.get("model_size", "large-v3")
        self.device = config_d.get("device", "cuda")
        self.compute_type = config_d.get("compute_type", "float16")
        self.language = config_d.get("language", "ru")
        self.vad_filter = config_d.get("vad_filter", True)
        self.vad_parameters = config_d.get("vad_parameters", None)

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters,
        }

    async def start(self) -> None:
        await super().start()
        logging.info(f"Loading Faster Whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")
        # Загрузка модели может заблокировать event loop на пару секунд при старте (это нормально)
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        logging.info("Faster Whisper model loaded successfully.")

    async def close(self) -> None:
        self.model = None
        await super().close()

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
        
        if not audio_bytes:
            yield {"text": "", "is_final": True}
            return

        import wave
        with wave.open("C:\\Nirmita\\scratch\\test_discord.wav", "wb") as wf:
            wf.setnchannels(ch if ch else 1)
            wf.setsampwidth(sw if sw else 2)
            wf.setframerate(sr if sr else 16000)
            wf.writeframes(audio_bytes)

        # 1. Читаем сырые байты PCM 16-bit
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # 2. Переводим во float32
        audio_float32 = audio_np.astype(np.float32) / 32768.0

        # 3. Сводим в моно, если аудио стерео
        if ch and ch > 1:
            audio_float32 = audio_float32.reshape(-1, ch).mean(axis=1)

        # 4. Ресемплим в 16000Hz (Whisper строго требует 16kHz)
        input_sr = sr or 16000
        if input_sr != 16000:
            audio_float32 = librosa.resample(audio_float32, orig_sr=input_sr, target_sr=16000)

        # 5. Функция для запуска распознавания (будет выполнена в пуле потоков)
        def _transcribe():
            segments, info = self.model.transcribe(
                audio_float32, 
                language=self.language, 
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters if self.vad_parameters else None,
                condition_on_previous_text=False,
                initial_prompt=prompt if prompt else None
            )
            # Собираем все кусочки текста в одну строку
            text = " ".join([segment.text for segment in segments]).strip()
            return text

        loop = asyncio.get_running_loop()
        try:
            # Запускаем в ThreadPoolExecutor, чтобы не блокировать асинхронный цикл JAIson
            text = await loop.run_in_executor(None, _transcribe)
            yield {"text": text, "is_final": True}
        except Exception as e:
            logging.error(f"Faster Whisper transcription failed: {e}")
            yield {"text": "", "is_final": True}
