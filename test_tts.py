"""
Быстрый тест Fish Audio TTS.
Синтезирует тестовую фразу и воспроизводит через sounddevice (наушники/колонки).

Запуск:
    python test_tts.py
"""

import asyncio
import sys
import os
import numpy as np

# Добавляем src в путь, чтобы импортировать модули проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import msgpack
import websockets.asyncio.client
import sounddevice as sd

# ─── Настройки (копируй из config.yaml) ───────────────────────────────────────
API_KEY      = "sk-fish-9BgqjIpX07tVQhhadF2vetrgVsSMUG-QxpVRCpIl1kU"
REFERENCE_ID = "8579f19895d8412d9f9e6d515872d455"
MODEL        = "s2.1-pro-free"   # Бесплатный tier для теста
LATENCY      = "balanced"
SAMPLE_RATE  = 44100
TEXT         = "Привет! Я Нира, твой виртуальный ассистент. Голосовой тест прошёл успешно!"

WS_URL = "wss://api.fish.audio/v1/tts/live"


def _pack(obj):
    return msgpack.packb(obj, use_bin_type=True)

def _unpack(data):
    return msgpack.unpackb(data, raw=False)


async def test_tts():
    print(f"[*] Подключаемся к Fish Audio...")
    print(f"[*] Текст: {TEXT}")
    print(f"[*] Модель: {MODEL}, Latency: {LATENCY}")
    print()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "model": MODEL,
    }

    start_event = {
        "event": "start",
        "request": {
            "text": "",
            "format": "pcm",
            "sample_rate": SAMPLE_RATE,
            "reference_id": REFERENCE_ID,
            "latency": LATENCY,
            "temperature": 0.7,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "chunk_length": 200,
            "min_chunk_length": 50,
            "normalize": True,
            "condition_on_previous_chunks": True,
            "prosody": {"speed": 1.0, "volume": 0.0, "normalize_loudness": True},
        },
    }

    all_audio = bytearray()
    chunk_count = 0

    try:
        async with websockets.asyncio.client.connect(
            WS_URL,
            additional_headers=headers,
            open_timeout=10,
            ping_interval=None,
        ) as ws:
            await ws.send(_pack(start_event))
            await ws.send(_pack({"event": "text", "text": TEXT}))
            await ws.send(_pack({"event": "flush"}))
            await ws.send(_pack({"event": "stop"}))

            print("[*] Получаем аудио чанки...")
            async for raw in ws:
                if not isinstance(raw, bytes):
                    continue
                msg = _unpack(raw)
                event = msg.get("event")

                if event == "audio":
                    audio_bytes = msg.get("audio", b"")
                    if audio_bytes:
                        all_audio.extend(audio_bytes)
                        chunk_count += 1
                        print(f"    Чанк {chunk_count}: {len(audio_bytes)} байт (итого: {len(all_audio)} байт)")

                elif event == "finish":
                    reason = msg.get("reason", "")
                    if reason == "error":
                        print(f"[!] Fish Audio вернул ошибку!")
                        return
                    print(f"[✓] Синтез завершён. Всего чанков: {chunk_count}, байт: {len(all_audio)}")
                    break

    except Exception as e:
        print(f"[!] Ошибка соединения: {e}")
        raise

    if not all_audio:
        print("[!] Аудио не получено — проверь api_key и reference_id")
        return

    # Конвертируем PCM (int16, mono) в float32 для sounddevice
    pcm_array = np.frombuffer(bytes(all_audio), dtype=np.int16).astype(np.float32) / 32768.0

    print(f"\n[*] Воспроизведение через sounddevice (sr={SAMPLE_RATE})...")
    print("[*] Если слышишь голос — TTS работает!")
    sd.play(pcm_array, samplerate=SAMPLE_RATE)
    sd.wait()
    print("[✓] Воспроизведение завершено.")


if __name__ == "__main__":
    asyncio.run(test_tts())
