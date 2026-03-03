import asyncio
import json
import base64
import wave
import numpy as np
import websockets
import logging
import torch
from typing import Dict, Any, AsyncGenerator

from utils.config import Config
from utils.processes import ProcessManager, ProcessType
from .base import STTOperation

class SherpaSTT(STTOperation):
    def __init__(self):
        super().__init__("sherpa")
        self.ws_url = "ws://localhost:6006"
        self.provider = "cuda" # Default, wait for config update
        self.te_apply = None
        
    async def start(self) -> None:
        """General setup needed to start generated"""
        await super().start()
        await ProcessManager().link(self.op_id, ProcessType.SHERPA)
        
        # [FIX] Асинхронная загрузка Silero TE, чтобы не блокировать старт всей системы
        if self.te_apply is None:
            # Запускаем загрузку в отдельном таске
            asyncio.create_task(self._load_te_model())

    async def _load_te_model(self):
        try:
            import os
            logging.info("Sherpa STT: Background loading Silero TE model (OFFLINE-ONLY)...")
            
            # Путь к локально скопированным моделям
            te_model_dir = os.path.abspath(os.path.join(os.getcwd(), "models", "silero_models"))
            
            if not os.path.exists(te_model_dir):
                logging.error(f"Sherpa STT: Local TE models not found at {te_model_dir}. Offline mode required, skipping TE.")
                return 

            # Проверяем наличие весов модели.
            # v2_4lang_q.pt должен лежать в src/silero/model/ внутри te_model_dir
            weights_path = os.path.join(te_model_dir, "src", "silero", "model", "v2_4lang_q.pt")
            if not os.path.exists(weights_path):
                logging.error(f"Sherpa STT: Silero TE weights missing at {weights_path}. Punctuation disabled.")
                return 

            # torch.hub.load с source='local' не лезет в интернет
            model, example_texts, languages, punct, apply_te = await asyncio.to_thread(
                torch.hub.load,
                repo_or_dir=te_model_dir,
                model='silero_te',
                source='local',
                force_reload=False
            )
            self.te_apply = apply_te
            logging.info("Sherpa STT: Silero TE model loaded from local disk.")
        except Exception as e:
            logging.error(f"Sherpa STT: Offline load of Silero TE failed: {e}. Punctuation will be disabled.")
        
    async def close(self) -> None:
        """Clean up resources before unloading"""
        await super().close()
        await ProcessManager().unlink(self.op_id, ProcessType.SHERPA)
        
    async def configure(self, config_d: Dict[str, Any]):
        """Configure operation-specific configuration"""
        if "ws_url" in config_d:
            self.ws_url = str(config_d['ws_url'])
        if "provider" in config_d:
            self.provider = str(config_d['provider'])
            
    async def get_configuration(self) -> Dict[str, Any]:
        """Returns values of configurable fields"""
        return {
            "ws_url": self.ws_url,
            "provider": self.provider
        }
    
    async def _send_audio_to_ws(self, audio_bytes: bytes, sr: int, sw: int, ch: int) -> str:
        """Helper to send audio to the websocket and get the recognition result."""
        # Sherpa expects mono 16000Hz PCM 16-bit by default, converted to float32
        # Here we convert the received PCM16 bytes into a float32 array [-1.0, 1.0]
        try:
            # We assume 16-bit PCM for now.
            if sw != 2:
               raise ValueError(f"Sherpa STT expects 16-bit audio, got {sw*8}-bit.")
            
            # Convert to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            logging.info(f"Sherpa STT received {len(audio_bytes)} bytes, {len(audio_np)} samples, sr={sr}, sw={sw}, ch={ch}")
            
            # Convert to float32 in range [-1.0, 1.0]
            audio_float32 = audio_np.astype(np.float32) / 32768.0
            
            # Ensure it is mono (if necessary)
            if ch > 1:
                audio_float32 = audio_float32.reshape(-1, ch).mean(axis=1)
                
            logging.debug(f"Sherpa STT processed float32 buffer of {len(audio_float32)} samples.")
                
            chunk_samples = max(1, int((sr or 16000) * 0.2))  # 200ms chunks

            async with websockets.connect(self.ws_url, max_size=4 * 1024 * 1024) as websocket:
                # Send the float32 stream in chunks to avoid oversized messages
                for i in range(0, len(audio_float32), chunk_samples):
                    await websocket.send(audio_float32[i:i + chunk_samples].tobytes())
                # Send end of stream message
                await websocket.send("Done")
                logging.debug("Sherpa STT: Sent phrase and Done.")
                
                # Receive transcript
                transcription = ""
                while True:
                    try:
                        message = await websocket.recv()
                        logging.debug(f"Sherpa STT: Raw message from WS: {message}")
                        data = json.loads(message)
                        # Sherpa protocol sends JSON with "text" field
                        if "text" in data:
                            transcription = data["text"]
                            logging.debug(f"Sherpa STT: Interim/Final transcription: '{transcription}'")
                        
                    except websockets.exceptions.ConnectionClosedOK:
                        logging.debug("Sherpa STT: WebSocket connection closed normally.")
                        break # Normal closure handled
                    except Exception as e:
                        # Some messages might not be JSON or other errors
                        if not isinstance(e, json.JSONDecodeError):
                            logging.error(f"Sherpa STT Error reading from WebSocket: {e}")
                        else:
                            logging.debug(f"Sherpa STT: Skipped non-JSON message: {message}")
                        break
                        
                logging.info(f"Sherpa STT: Finished with transcription: '{transcription.strip()}'")
                return transcription.strip()
                
        except Exception as e:
            logging.error(f"Sherpa STT error processing audio: {e}")
            return ""

    async def _generate(self, prompt: str = None, audio_bytes: bytes = None, sr: int = None, sw: int = None, ch: int = None, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a output stream"""
        # Call websocket
        transcription = await self._send_audio_to_ws(audio_bytes, sr, sw, ch)
        
        # [ADD] Применяем Silero TE, если модель загружена
        if transcription and self.te_apply:
            try:
                # Ограничиваем длину текста для TE (опционально)
                transcription = self.te_apply(transcription, lan='ru')
                logging.debug(f"Sherpa STT: Enhanced transcription: '{transcription}'")
            except Exception as e:
                logging.warning(f"Sherpa STT: TE application failed: {e}")
        
        yield {
            "transcription": transcription
        }
