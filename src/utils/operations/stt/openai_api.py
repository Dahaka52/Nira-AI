import os
import io
import wave
import logging
from typing import Dict, Any, AsyncGenerator

from openai import AsyncOpenAI

from .base import STTOperation

class OpenAISTT(STTOperation):
    def __init__(self):
        super().__init__("openai_stt")
        self.client = None
        
        self.base_url = "https://api.openai.com/v1"
        self.api_key = ""
        self.model = "whisper-large-v3-turbo"
        self.language = "ru"

    async def configure(self, config_d: Dict[str, Any]):
        self.base_url = config_d.get("base_url", self.base_url)
        self.api_key = config_d.get("api_key", self.api_key)
        self.model = config_d.get("model", self.model)
        self.language = config_d.get("language", self.language)

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "language": self.language
        }

    async def start(self) -> None:
        await super().start()
        api_key = self.api_key
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
            
        if not api_key:
            logging.warning("API key is not set for OpenAISTT. Requests will likely fail.")
        
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
        logging.info(f"OpenAISTT initialized for model {self.model} at {self.base_url}")

    async def close(self) -> None:
        await super().close()
        if self.client:
            await self.client.close()
            self.client = None

    def _pcm_to_wav(self, audio_bytes: bytes, sr: int, sw: int, ch: int) -> bytes:
        """Конвертирует сырой PCM в WAV-контейнер в памяти."""
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(ch)
            wav_file.setsampwidth(sw)
            wav_file.setframerate(sr)
            wav_file.writeframes(audio_bytes)
        return wav_io.getvalue()

    async def _generate(
        self,
        prompt: str = None,
        audio_bytes: bytes = None,
        sr: int = None,
        sw: int = None,
        ch: int = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        
        if not audio_bytes or len(audio_bytes) == 0:
            yield {"text": "", "is_final": True}
            return
            
        try:
            input_sr = sr or 16000
            input_ch = ch or 1
            audio_to_send = audio_bytes
            
            # Groq Docs: "Our speech-to-text models will downsample audio to 16KHz mono...
            # This preprocessing can be performed client-side... For lower latency, convert your files to wav format."
            # Сжимаем 48kHz Stereo в 16kHz Mono, чтобы файл стал в 6 раз меньше, что сильно ускорит upload.
            if input_sr != 16000 or input_ch != 1:
                import numpy as np
                import librosa
                
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # VAD Filter: проверяем громкость (RMS)
                rms = np.sqrt(np.mean(audio_np**2))
                if rms < 0.005:
                    yield {"text": "", "is_final": True}
                    return

                # Сводим стерео в моно, чтобы Whisper на сервере не путался
                if input_ch > 1:
                    audio_np = audio_np.reshape(-1, input_ch).mean(axis=1)
                # Ресемплим в 16kHz
                if input_sr != 16000:
                    audio_np = librosa.resample(audio_np, orig_sr=input_sr, target_sr=16000)
                    
                audio_np = (audio_np * 32767.0).astype(np.int16)
                audio_to_send = audio_np.tobytes()
                
            # 1. Конвертируем PCM в WAV (теперь это 16kHz Mono)
            wav_bytes = self._pcm_to_wav(audio_to_send, 16000, sw or 2, 1)
            
            # 2. Создаем виртуальный файл для OpenAI
            audio_file = ("audio.wav", wav_bytes, "audio/wav")
            
            # 3. Отправляем запрос
            response = await self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=self.language,
                temperature=0.0,
                prompt="Нира",
                response_format="json"
            )
            
            text = response.text.strip()
            
            # Жесткий фильтр галлюцинаций (Whisper часто придумывает это в тишине)
            hallucinations = ["продолжение следует", "спасибо за просмотр", "спасибо.", "полплеер", "привет, тут", "субтитры", "amara", "подписывайтесь", "субтитры от"]
            for h in hallucinations:
                if h in text.lower():
                    text = ""
                    break
                    
            yield {"text": text, "is_final": True}
            
        except Exception as e:
            logging.error(f"OpenAISTT transcription failed: {e}")
            yield {"text": "", "is_final": True}
