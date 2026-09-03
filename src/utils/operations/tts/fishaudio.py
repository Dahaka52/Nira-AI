"""
Fish Audio TTS Operation — WebSocket streaming (низкая задержка).

Использует официальный WebSocket API Fish Audio (wss://api.fish.audio/v1/tts/live)
с сериализацией MessagePack. Аудио запрашивается в формате PCM (сырые байты),
что исключает необходимость в дополнительном декодировании.

Вывод (_generate):
  audio_bytes : bytes  — сырые PCM байты (16-bit signed, mono)
  sr          : int    — частота дискретизации (по умолчанию 44100)
  sw          : int    — ширина сэмпла в байтах (2 = 16-bit)
  ch          : int    — каналы (1 = mono)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Dict, Any

import msgpack
import websockets
import websockets.asyncio.client
from websockets.exceptions import ConnectionClosed

from .base import TTSOperation


# ─── Константы ────────────────────────────────────────────────────────────────

WS_URL = "wss://api.fish.audio/v1/tts/live"

# PCM-параметры, которые Fish Audio возвращает при format=pcm
_PCM_SR = 44100   # Hz (дефолт Fish Audio для PCM)
_PCM_SW = 2       # байт (16-bit)
_PCM_CH = 1       # моно


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _pack(obj: dict) -> bytes:
    """Сериализация dict → MessagePack bytes."""
    return msgpack.packb(obj, use_bin_type=True)


def _unpack(data: bytes) -> dict:
    """Десериализация MessagePack bytes → dict."""
    return msgpack.unpackb(data, raw=False)


# ─── Операция TTS ────────────────────────────────────────────────────────────

class FishAudioTTS(TTSOperation):
    """TTS через Fish Audio WebSocket API (streaming, MessagePack)."""

    def __init__(self):
        super().__init__("fish_audio")

        # Обязательные поля
        self.api_key: str = ""
        self.reference_id: str = ""

        # Модель Fish Audio
        self.model: str = "s2.1-pro"          # s1 | s2-pro | s2.1-pro | s2.1-pro-free

        # Параметры генерации
        self.latency: str = "balanced"         # low | balanced | normal
        self.format: str = "pcm"               # всегда pcm для прямой передачи
        self.sample_rate: int = _PCM_SR        # 44100 Hz
        self.temperature: float = 0.7
        self.top_p: float = 0.7
        self.repetition_penalty: float = 1.2
        self.chunk_length: int = 200           # 100–300 символов на чанк
        self.min_chunk_length: int = 50
        self.max_new_tokens: int = 1024
        self.speed: float = 1.0               # просодия: скорость речи 0.5–2.0
        self.volume: float = 0.0              # просодия: громкость в dB

    # ─── Жизненный цикл ───────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()
        logging.info("FishAudioTTS: операция запущена (WebSocket streaming, format=pcm)")

    async def close(self) -> None:
        await super().close()
        logging.info("FishAudioTTS: операция остановлена")

    # ─── Конфигурация ─────────────────────────────────────────────────────────

    async def configure(self, config_d: Dict[str, Any]) -> None:
        """Считать и проверить параметры из config.yaml."""
        # Ключ: из конфига или из переменной окружения
        self.api_key = str(
            config_d.get("api_key") or os.environ.get("FISH_API_KEY") or ""
        ).strip()
        if not self.api_key:
            raise ValueError(
                "FishAudioTTS: api_key не задан. "
                "Укажи 'api_key' в config.yaml или переменную окружения FISH_API_KEY."
            )

        self.reference_id = str(config_d.get("reference_id") or "").strip()
        if not self.reference_id:
            raise ValueError(
                "FishAudioTTS: reference_id не задан. "
                "Укажи UUID клона голоса из fish.audio → My Voices."
            )

        if "model" in config_d:
            m = str(config_d["model"]).strip()
            if m not in ("s1", "s2-pro", "s2.1-pro", "s2.1-pro-free"):
                logging.warning("FishAudioTTS: неизвестная модель '%s', используем s2.1-pro", m)
            else:
                self.model = m

        if "latency" in config_d:
            lv = str(config_d["latency"]).strip()
            if lv not in ("low", "balanced", "normal"):
                logging.warning("FishAudioTTS: неизвестный latency '%s', используем balanced", lv)
            else:
                self.latency = lv

        if "temperature" in config_d:
            self.temperature = float(config_d["temperature"])
            assert 0.0 <= self.temperature <= 1.0, "temperature должен быть в диапазоне [0, 1]"

        if "top_p" in config_d:
            self.top_p = float(config_d["top_p"])
            assert 0.0 <= self.top_p <= 1.0, "top_p должен быть в диапазоне [0, 1]"

        if "repetition_penalty" in config_d:
            self.repetition_penalty = float(config_d["repetition_penalty"])

        if "chunk_length" in config_d:
            self.chunk_length = int(config_d["chunk_length"])
            assert 100 <= self.chunk_length <= 300, "chunk_length должен быть в диапазоне [100, 300]"

        if "min_chunk_length" in config_d:
            self.min_chunk_length = int(config_d["min_chunk_length"])
            assert 0 <= self.min_chunk_length <= 100, "min_chunk_length должен быть в диапазоне [0, 100]"

        if "max_new_tokens" in config_d:
            self.max_new_tokens = int(config_d["max_new_tokens"])

        if "speed" in config_d:
            self.speed = float(config_d["speed"])
            assert 0.5 <= self.speed <= 2.0, "speed должен быть в диапазоне [0.5, 2.0]"

        if "volume" in config_d:
            self.volume = float(config_d["volume"])
            assert -20.0 <= self.volume <= 20.0, "volume должен быть в диапазоне [-20, 20]"

        if "sample_rate" in config_d and config_d["sample_rate"]:
            self.sample_rate = int(config_d["sample_rate"])

        logging.info(
            "FishAudioTTS: настроен — model=%s, reference_id=%s..., latency=%s, "
            "sample_rate=%s, speed=%.1f",
            self.model,
            self.reference_id[:8] if len(self.reference_id) > 8 else self.reference_id,
            self.latency,
            self.sample_rate,
            self.speed,
        )

    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "reference_id": self.reference_id,
            "latency": self.latency,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "chunk_length": self.chunk_length,
            "min_chunk_length": self.min_chunk_length,
            "max_new_tokens": self.max_new_tokens,
            "speed": self.speed,
            "volume": self.volume,
        }

    # ─── Генерация ────────────────────────────────────────────────────────────

    async def _generate(
        self,
        content: str = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Подключиться к Fish Audio WebSocket, стримить PCM-чанки.

        Протокол:
          1. → StartEvent   (конфиг сессии, text="")
          2. → TextEvent    (текст для синтеза)
          3. → FlushEvent   (принудительный синтез буфера)
          4. → CloseEvent   (event="stop")
          5. ← AudioEvent*  (чанки PCM bytes)
          6. ← FinishEvent  (reason="stop" или "error")
        """
        if not content:
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "model": self.model,
        }

        start_event = {
            "event": "start",
            "request": {
                "text": "",
                "format": "pcm",
                "sample_rate": self.sample_rate,
                "reference_id": self.reference_id,
                "latency": self.latency,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "chunk_length": self.chunk_length,
                "min_chunk_length": self.min_chunk_length,
                "max_new_tokens": self.max_new_tokens,
                "condition_on_previous_chunks": True,
                "normalize": True,
                "prosody": {
                    "speed": self.speed,
                    "volume": self.volume,
                    "normalize_loudness": True,
                },
            },
        }

        text_event = {"event": "text", "text": content}
        flush_event = {"event": "flush"}
        stop_event = {"event": "stop"}

        try:
            async with websockets.asyncio.client.connect(
                WS_URL,
                additional_headers=headers,
                open_timeout=10,       # таймаут подключения (сек)
                ping_interval=None,    # отключить ping — Fish Audio не поддерживает
            ) as ws:
                # 1. Отправить конфиг сессии
                await ws.send(_pack(start_event))
                # 2. Отправить текст
                await ws.send(_pack(text_event))
                # 3. Принудительно запустить синтез
                await ws.send(_pack(flush_event))
                # 4. Сигнал конца потока
                await ws.send(_pack(stop_event))

                # 5. Принимать аудио чанки
                async for raw_msg in ws:
                    if isinstance(raw_msg, bytes):
                        msg = _unpack(raw_msg)
                    elif isinstance(raw_msg, str):
                        msg = json.loads(raw_msg)
                    else:
                        continue

                    event_type = msg.get("event")

                    if event_type == "audio":
                        audio_bytes = msg.get("audio")
                        if audio_bytes and isinstance(audio_bytes, (bytes, bytearray)):
                            yield {
                                "audio_bytes": bytes(audio_bytes),
                                "sr": self.sample_rate,
                                "sw": _PCM_SW,
                                "ch": _PCM_CH,
                            }

                    elif event_type == "finish":
                        reason = msg.get("reason", "")
                        if reason == "error":
                            raise RuntimeError(
                                "FishAudioTTS: сервер вернул finish с reason='error'"
                            )
                        # reason == "stop" — нормальное завершение
                        break

                    else:
                        # Игнорируем неизвестные события (future-proof)
                        logging.debug("FishAudioTTS: неизвестное событие: %s", event_type)

        except ConnectionClosed as e:
            logging.error("FishAudioTTS: соединение закрыто неожиданно: %s", e)
            raise
        except Exception as e:
            logging.error("FishAudioTTS: ошибка генерации: %s", e, exc_info=True)
            raise
