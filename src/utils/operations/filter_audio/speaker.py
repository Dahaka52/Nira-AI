from typing import Dict, Any, AsyncGenerator
import logging
import queue
import threading
import numpy as np
import sounddevice as sd

from .base import FilterAudioOperation

class SpeakerOutputFilter(FilterAudioOperation):
    """
    Фильтр аудио, который воспроизводит проходящий через него звук на выбранное устройство 
    (например, стандартное аудиоустройство Windows или виртуальный кабель), при этом пропуская чанки дальше без изменений.
    """
    def __init__(self):
        super().__init__("speaker")
        self.device_name = None
        self.device_index = None
        self.enabled = False
        self.vol = 1.0
        
        self.q = queue.Queue()
        self._playback_thread = None
        self._stop_event = threading.Event()
        self._stream = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            # Очищаем очередь, чтобы не доигрывать старый звук
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
        logging.info(f"SpeakerOutputFilter: enabled set to {self.enabled}")

    def stop_audio(self) -> None:
        """Мгновенно прерывает текущее воспроизведение и очищает очередь."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except Exception:
                break
        if self._stream:
            try:
                self._stream.abort()
                self._stream.start()
            except Exception:
                pass

    async def start(self) -> None:
        await super().start()
        self._stop_event.clear()
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()
        logging.info("SpeakerOutputFilter: операция запущена (вокер запущен)")

    async def close(self) -> None:
        self._stop_event.set()
        
        # Flush queue
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
                
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)
            
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            
        await super().close()
        logging.info("SpeakerOutputFilter: операция остановлена")

    async def configure(self, config_d: Dict[str, Any]) -> None:
        self.device_name = config_d.get("device_name")
        self.device_index = config_d.get("device_index")
        if "enabled" in config_d:
            self.enabled = bool(config_d["enabled"])
        self.vol = float(config_d.get("vol", config_d.get("volume", 1.0)))
        logging.info(f"SpeakerOutputFilter: настроен name={self.device_name}, index={self.device_index}, enabled={self.enabled}, vol={self.vol}")

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "device_index": self.device_index,
            "enabled": self.enabled,
            "vol": self.vol
        }

    def _resolve_device(self):
        if self.device_index is not None:
            return self.device_index
            
        if self.device_name:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if self.device_name.lower() in dev['name'].lower() and dev['max_output_channels'] > 0:
                    return i
                    
        return None # Default Windows sound device

    def _playback_worker(self):
        """Фоновый поток для воспроизведения аудио через sounddevice."""
        current_sr = None
        current_ch = None
        
        device = self._resolve_device()
        
        while not self._stop_event.is_set():
            try:
                # Ждем новых чанков с небольшим таймаутом, чтобы можно было проверить stop_event
                audio_bytes, sr, sw, ch = self.q.get(timeout=0.1)
                
                # Если сменилась частота или каналы — пересоздаем стрим
                if self._stream is None or sr != current_sr or ch != current_ch:
                    if self._stream:
                        self._stream.stop()
                        self._stream.close()
                    current_sr = sr
                    current_ch = ch
                    try:
                        self._stream = sd.OutputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype='float32',
                            device=device
                        )
                        self._stream.start()
                    except Exception as e:
                        logging.error(f"SpeakerOutputFilter: Ошибка открытия аудиопотока: {e}")
                        self._stream = None
                        continue

                # Пишем float32 PCM фреймы в стрим
                if self._stream and self.enabled:
                    try:
                        pcm_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        if self.vol != 1.0:
                            pcm_array = pcm_array * self.vol
                        pcm_array = pcm_array.reshape(-1, ch) # OutputStream требует 2D массив
                        self._stream.write(pcm_array)
                    except sd.PortAudioError as e:
                        logging.error(f"SpeakerOutputFilter: Ошибка воспроизведения (underflow/overflow): {e}")
                        
                self.q.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"SpeakerOutputFilter: Непредвиденная ошибка в вокере: {e}")

    async def _generate(self, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        audio_bytes = kwargs.get('audio_bytes')
        sr = kwargs.get('sr')
        sw = kwargs.get('sw')
        ch = kwargs.get('ch')

        if self.enabled and audio_bytes is not None:
            # Кладем чанк в очередь на воспроизведение в фоновом потоке
            try:
                self.q.put_nowait((audio_bytes, sr, sw, ch))
            except queue.Full:
                logging.warning("SpeakerOutputFilter: Очередь переполнена, пропускаем чанк")
                
        # Пропускаем данные дальше по пайплайну (чтобы веб-дашборд тоже получал их)
        yield kwargs
