import os
import io
import json
import base64
import logging
import asyncio
from typing import Dict, Any, AsyncGenerator

import websockets

from .base import STTOperation

class GoogleLiveSTT(STTOperation):
    def __init__(self):
        super().__init__("google_live_stt")
        self.api_key = ""
        self.model = "gemini-3.5-transcribe-live"
        self.ws_url_template = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"

    async def configure(self, config_d: Dict[str, Any]):
        self.api_key = config_d.get("api_key", self.api_key)
        self.model = config_d.get("model", self.model)

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "model": self.model
        }

    async def start(self) -> None:
        await super().start()
        api_key = self.api_key
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
            
        if not api_key:
            logging.warning("API key is not set for GoogleLiveSTT. Requests will fail.")
        
        self.api_key = api_key
        logging.info(f"GoogleLiveSTT initialized for model {self.model}")

    async def close(self) -> None:
        await super().close()

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
            
        ws_url = self.ws_url_template.format(api_key=self.api_key)
        
        try:
            async with websockets.connect(ws_url) as ws:
                # 1. Отправляем Setup пакет
                setup_msg = {
                    "setup": {
                        "model": f"models/{self.model}",
                        "generationConfig": {
                            "responseModalities": ["TEXT"],
                        },
                        "inputAudioTranscription": {
                            "languageCodes": []
                        }
                    }
                }
                await ws.send(json.dumps(setup_msg))
                
                # 2. Ждём подтверждения Setup (опционально, но сервер ответит setupComplete)
                # 3. Отправляем Аудио
                
                # При необходимости ресемплим аудио в 16000Hz Mono для Google
                input_sr = sr or 16000
                input_ch = ch or 1
                
                audio_to_send = audio_bytes
                
                if input_sr != 16000 or input_ch != 1:
                    import numpy as np
                    import librosa
                    
                    # Декодируем 16-bit PCM
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Если стерео, усредняем до моно
                    if input_ch > 1:
                        audio_np = audio_np.reshape(-1, input_ch).mean(axis=1)
                        
                    # Ресемплим в 16kHz
                    if input_sr != 16000:
                        audio_np = librosa.resample(audio_np, orig_sr=input_sr, target_sr=16000)
                        
                    # Конвертируем обратно в 16-bit PCM
                    audio_np = (audio_np * 32767.0).astype(np.int16)
                    audio_to_send = audio_np.tobytes()

                mime_type = "audio/pcm;rate=16000"
                audio_base64 = base64.b64encode(audio_to_send).decode('utf-8')
                audio_msg = {
                    "realtimeInput": {
                        "mediaChunks": [{
                            "mimeType": mime_type,
                            "data": audio_base64
                        }]
                    }
                }
                
                # ИСПРАВЛЕНИЕ: Gemini Live API требует ключ "mediaChunks" для BidiGenerateContent, 
                # НО в доках написано "audio" для транскрипции! Меняем на audio.
                audio_msg = {
                    "realtimeInput": {
                        "audio": {
                            "mimeType": mime_type,
                            "data": audio_base64
                        }
                    }
                }
                # Отправляем аудио
                await ws.send(json.dumps(audio_msg))
                
                # 4. Отправляем сигнал окончания потока
                end_msg = {
                    "realtimeInput": {
                        "audioStreamEnd": True
                    }
                }
                await ws.send(json.dumps(end_msg))
                
                # 5. Слушаем ответы сервера, пока не получим финальную транскрипцию
                final_text = ""
                while True:
                    try:
                        response_str = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        response = json.loads(response_str)
                        content = response.get("serverContent", {})
                        
                        # Дебаг-логирование ответа
                        logging.debug(f"[Google STT] Received: {response}")
                        
                        if "inputTranscription" in content:
                            text = content["inputTranscription"].get("text", "")
                            if text:
                                final_text = text
                                break # Получили финальную транскрипцию!
                                
                        if content.get("turnComplete"):
                            break # Сервер завершил обработку этого куска
                                
                    except asyncio.TimeoutError:
                        logging.warning("[Google STT] Timeout waiting for response")
                        break # Если сервер молчит 15 секунд, завершаем
                    except websockets.exceptions.ConnectionClosed as e:
                        logging.warning(f"[Google STT] Connection closed by server: {e.code} {e.reason}")
                        break
                        
                yield {"text": final_text.strip(), "is_final": True}
                
        except Exception as e:
            logging.error(f"GoogleLiveSTT transcription failed: {e}")
            yield {"text": "", "is_final": True}
