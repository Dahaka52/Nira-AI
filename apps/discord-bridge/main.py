"""Nira's Discord voice bridge (Pycord + DAVE).

Использует py-cord PR#3159 + davey для расшифровки DAVE E2EE пакетов.
Discord сделал DAVE обязательным с 1 марта 2026 для всех голосовых каналов.
Передаёт декодированный PCM от каждого пользователя в STT endpoint,
и воспроизводит TTS из JAIson WebSocket обратно в голосовой канал.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Optional

import discord
import httpx
import websockets


SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
FRAME_BYTES = 3_840  # 20 ms @ 48 kHz, 16-bit stereo PCM

# STT target format: 16 kHz, mono, 16-bit PCM
STT_SAMPLE_RATE = 16_000
STT_CHANNELS = 1
STT_SAMPLE_WIDTH = 2


# ---------------------------------------------------------------------------
# TTS playback source
# ---------------------------------------------------------------------------

class StreamingPCMSource(discord.AudioSource):
    """Живой PCM-буфер, потребляемый аудио-плеером discord.py."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._condition = threading.Condition()
        self._closed = False
        self._ratecv_state = None

    def add_pcm(self, audio: bytes, sr: int, sw: int, ch: int) -> None:
        if not audio or sw != SAMPLE_WIDTH or ch not in (1, 2):
            return
        try:
            mono = audioop.tomono(audio, SAMPLE_WIDTH, 0.5, 0.5) if ch == 2 else audio
            if sr != SAMPLE_RATE:
                mono, self._ratecv_state = audioop.ratecv(
                    mono, SAMPLE_WIDTH, 1, sr, SAMPLE_RATE, self._ratecv_state
                )
            stereo = audioop.tostereo(mono, SAMPLE_WIDTH, 1.0, 1.0)
        except audioop.error:
            logging.exception("Could not convert TTS audio for Discord")
            return

        with self._condition:
            if not self._closed:
                self._buffer.extend(stereo)
                self._condition.notify_all()

    def has_frame(self) -> bool:
        with self._condition:
            return len(self._buffer) >= FRAME_BYTES

    def close_input(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def read(self) -> bytes:
        with self._condition:
            while len(self._buffer) < FRAME_BYTES and not self._closed:
                self._condition.wait(timeout=0.08)
                if len(self._buffer) < FRAME_BYTES and not self._closed:
                    return b"\0" * FRAME_BYTES
            if not self._buffer and self._closed:
                return b""
            if len(self._buffer) < FRAME_BYTES:
                frame = bytes(self._buffer).ljust(FRAME_BYTES, b"\0")
                self._buffer.clear()
                return frame
            frame = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
            return frame


# ---------------------------------------------------------------------------
# Per-user speech accumulator
# ---------------------------------------------------------------------------

class SpeakerBuffer:
    """Накапливает PCM от одного пользователя и сбрасывает по тишине."""

    def __init__(self, member, loop, silence_s: float, min_bytes: int, callback):
        self.member = member
        self.loop = loop
        self.silence_s = silence_s
        self.min_bytes = min_bytes
        self.callback = callback
        self.audio = bytearray()
        self.started_at = 0.0
        self.last_packet_time = 0.0
        self.lock = threading.Lock()
        self._closing = False

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def append(self, pcm: bytes) -> None:
        with self.lock:
            if not self.audio:
                self.started_at = time.time()
            if len(pcm) % 4 != 0:
                logging.warning(
                    "[SINK] PCM length %d is not a multiple of 4! Byte alignment corrupted!", len(pcm)
                )
            self.audio.extend(pcm)
            self.last_packet_time = time.time()

    def _monitor_loop(self) -> None:
        while not self._closing:
            time.sleep(0.1)
            with self.lock:
                if not self.audio:
                    continue
                if time.time() - self.last_packet_time >= self.silence_s:
                    self._do_flush()

    def _do_flush(self) -> None:
        total = len(self.audio)
        if total >= self.min_bytes:
            audio, started_at = bytes(self.audio), self.started_at
            self.loop.call_soon_threadsafe(self.callback, self.member, audio, started_at)
        self.audio.clear()

    def close(self) -> None:
        self._closing = True
        with self.lock:
            self._do_flush()


# ---------------------------------------------------------------------------
# Pycord Sink — принимает декодированный PCM из DAVE-потока
# ---------------------------------------------------------------------------

class NiraRawSink(discord.sinks.Sink):
    """
    Кастомный Pycord Sink с поддержкой DAVE.

    Pycord PR#3159 + davey расшифровывают DAVE-пакеты перед вызовом write(),
    поэтому мы получаем чистый стерео 48 kHz PCM, а не зашифрованный мусор.

    Сигнатура write(data, user) — стандартный Pycord Sink API.
    data может быть VoiceData (с полем .pcm) или bytes напрямую.
    """

    def __init__(self, bot: "NiraDiscordBot") -> None:
        super().__init__()
        self.bot = bot
        self.buffers: dict[int, SpeakerBuffer] = {}
        self.lock = threading.Lock()
        self._write_call_count = 0
        self._last_write_time = time.time()

    def write(self, data, user) -> None:  # type: ignore[override]
        self._last_write_time = time.time()
        self._write_call_count += 1

        # Извлекаем PCM из VoiceData или bytes
        if hasattr(data, "pcm"):
            pcm = bytes(data.pcm)
        elif isinstance(data, (bytes, bytearray)):
            pcm = bytes(data)
        else:
            return

        if not pcm:
            return

        # user — это Member | User | None в Pycord
        user_id = getattr(user, "id", None)
        if user_id is None:
            if self._write_call_count <= 5:
                logging.warning("[SINK] user is None — SSRC→Member resolve failed")
            return

        # Диагностика: первые 5 и каждые 200 вызовов
        if self._write_call_count <= 5 or self._write_call_count % 200 == 0:
            logging.info(
                "[SINK] write() #%d: user_id=%d, bytes=%d",
                self._write_call_count,
                user_id,
                len(pcm),
            )

        # Игнорируем собственный голос бота
        if self.bot.user and user_id == self.bot.user.id:
            return

        with self.lock:
            buffer = self.buffers.get(user_id)
            if buffer is None:
                minimum = int(
                    SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * self.bot.min_speech_ms / 1000
                )
                logging.info(
                    "[SINK] Создаём буфер для user_id=%d (min_bytes=%d)", user_id, minimum
                )
                buffer = SpeakerBuffer(
                    user,
                    self.bot.loop,
                    self.bot.silence_ms / 1000,
                    minimum,
                    self.bot.submit_speech_from_thread,
                )
                self.buffers[user_id] = buffer

        buffer.append(pcm)

    def format_audio(self, audio) -> None:
        """Нам не нужна пост-обработка — данные уже переданы в STT."""
        pass

    def cleanup(self) -> None:
        with self.lock:
            for buf in self.buffers.values():
                buf.close()
            self.buffers.clear()


# ---------------------------------------------------------------------------
# Discord Bot
# ---------------------------------------------------------------------------

class NiraDiscordBot(discord.Bot):
    """Основной Discord-бот Ниры — принимает голос через DAVE, воспроизводит TTS."""

    def __init__(self, args: argparse.Namespace) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.members = True  # нужен для разрешения user_id → Member
        super().__init__(intents=intents)

        self.guild_id = int(args.guild_id)
        self.channel_id = int(args.voice_channel_id)
        self.api_url = args.api_url
        self.ws_url = args.ws_url
        self.source_id = args.source_id
        self.silence_ms = max(100, args.silence_ms)
        self.min_speech_ms = max(100, args.min_speech_ms)
        self.auto_join = args.auto_join

        self.vc: Optional[discord.VoiceClient] = None
        self.sink: Optional[NiraRawSink] = None
        self.ws_task: Optional[asyncio.Task] = None
        self.status_heartbeat_task: Optional[asyncio.Task] = None
        self.audio_source: Optional[StreamingPCMSource] = None
        self.audio_job_id: Optional[str] = None
        self.ready_once = False
        self._voice_lock = asyncio.Lock()

    async def on_ready(self) -> None:
        if self.ready_once:
            return
        self.ready_once = True
        logging.info("Authenticated as %s (Pycord %s)", self.user, discord.__version__)
        self.ws_task = asyncio.create_task(self.event_listener())
        self.watchdog_task = asyncio.create_task(self._recording_watchdog())
        self.status_heartbeat_task = asyncio.create_task(self._status_heartbeat())
        if self.auto_join:
            await self.join_configured_channel()

    async def _recording_watchdog(self) -> None:
        """Отключено: вызывало баги со спамом пакетов из-за ложных срабатываний в моменты долгой тишины."""
        pass

    async def _status_heartbeat(self) -> None:
        """Регулярная отправка телеметрии моста в JAIson."""
        while not self.is_closed():
            try:
                await self._send_status_update()
            except Exception:
                pass
            await asyncio.sleep(3.0)

    async def _send_status_update(self) -> None:
        connected = bool(self.vc and self.vc.is_connected())
        ch = getattr(self.vc, "channel", None) if self.vc else None
        status = {
            "online": True,
            "connected_to_voice": connected,
            "channel_name": getattr(ch, "name", None),
            "channel_id": getattr(ch, "id", None),
            "guild_id": self.guild_id,
            "is_playing": bool(self.vc and self.vc.is_playing()),
            "voice_ping_ms": round(self.vc.latency * 1000, 1) if self.vc and hasattr(self.vc, "latency") else None,
            "gateway_ping_ms": round(self.latency * 1000, 1) if hasattr(self, "latency") else None,
        }
        base_api = self.api_url.rsplit("/api", 1)[0]
        status_url = f"{base_api}/api/bridge/discord/status"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(status_url, json=status)
        except Exception:
            pass

    async def join_configured_channel(self) -> None:
        async with self._voice_lock:
            channel = self.get_channel(self.channel_id) or await self.fetch_channel(self.channel_id)
            if not isinstance(channel, discord.VoiceChannel):
                raise RuntimeError(f"Channel {self.channel_id} is not a voice channel")
            if channel.guild.id != self.guild_id:
                raise RuntimeError("Configured channel belongs to another server")

            # Если уже подключены к нужному каналу — ничего не делаем
            if self.vc and self.vc.is_connected():
                if self.vc.channel.id != channel.id:
                    await self.vc.move_to(channel)
                await self._send_status_update()
                return

            self.vc = await channel.connect()
            guild = self.get_guild(self.guild_id)
            if guild:
                await guild.change_voice_state(channel=channel, self_deaf=False, self_mute=False)
            logging.info(
                "Joined voice channel %s (%s) with DAVE support",
                channel.name, channel.id,
            )

            # Запускаем запись с NiraRawSink
            self.sink = NiraRawSink(self)
            self.vc.start_recording(
                self.sink,
                self._on_recording_finished,
            )
            logging.info("[SINK] Recording started — Нира слушает голосовой канал")
            await self._send_status_update()

    async def leave_channel(self) -> None:
        async with self._voice_lock:
            self.stop_audio()
            if self.vc and self.vc.is_connected():
                try:
                    self.vc.stop_recording()
                except Exception:
                    pass
                if self.sink:
                    self.sink.cleanup()
                    self.sink = None
                try:
                    await self.vc.disconnect(force=True)
                except Exception:
                    pass
                self.vc = None
                logging.info("Disconnected from Discord voice channel (Local mode active)")
            await self._send_status_update()

    def _on_recording_finished(self, sink: NiraRawSink, *args) -> None:
        """Вызывается когда stop_recording() завершён (не используем активно)."""
        logging.info("[SINK] Recording finished")

    # ------------------------------------------------------------------
    # Speech submission
    # ------------------------------------------------------------------

    def submit_speech_from_thread(self, member, audio: bytes, timestamp: float) -> None:
        asyncio.create_task(self.submit_speech(member, audio, timestamp))

    async def submit_speech(self, member, audio: bytes, timestamp: float) -> None:
        if not self.vc or not self.vc.is_connected():
            return

        # Диагностика: сохраняем WAV
        try:
            import wave
            wav_path = r"C:\Nirmita\scratch\discord_bridge_raw.wav"
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio)
        except Exception:
            logging.exception("[SINK] Could not save diagnostic WAV")

        payload = {
            "user": getattr(member, "display_name", None) or getattr(member, "name", f"user_{member.id}"),
            "speaker_id": str(member.id),
            "timestamp": timestamp,
            "audio_bytes": base64.b64encode(audio).decode("ascii"),
            "sr": SAMPLE_RATE,
            "sw": SAMPLE_WIDTH,
            "ch": CHANNELS,
            "source_id": self.source_id,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
            logging.info(
                "Submitted Discord speech from %s (%d bytes raw 48 kHz stereo)",
                payload["user"],
                len(audio),
            )
        except Exception:
            logging.exception("Could not send Discord speech to JAIson")

    # ------------------------------------------------------------------
    # TTS playback
    # ------------------------------------------------------------------

    def start_audio_if_ready(self) -> None:
        if not self.vc or not self.vc.is_connected() or not self.audio_source:
            return
        if not self.vc.is_playing() and self.audio_source.has_frame():
            self.vc.play(self.audio_source, after=self.audio_finished)

    def audio_finished(self, error: Exception | None) -> None:
        if error:
            logging.error("Discord playback failed: %s", error)
        self.audio_source = None
        self.audio_job_id = None

    def stop_audio(self) -> None:
        if self.audio_source:
            self.audio_source.close_input()
        if self.vc and self.vc.is_playing():
            self.vc.stop()
        self.audio_source = None
        self.audio_job_id = None

    # ------------------------------------------------------------------
    # JAIson WebSocket listener (TTS events & Control)
    # ------------------------------------------------------------------

    async def event_listener(self) -> None:
        while not self.is_closed():
            try:
                async with websockets.connect(self.ws_url, max_size=8 * 1024 * 1024) as ws:
                    logging.info("Connected to JAIson WebSocket")
                    async for raw_event in ws:
                        await self.handle_event(raw_event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("JAIson WebSocket disconnected; retrying in 3 seconds")
                await asyncio.sleep(3)

    async def handle_event(self, raw_event: str) -> None:
        try:
            event, _status = json.loads(raw_event)
        except (TypeError, ValueError):
            return

        msg = event.get("message")
        if msg == "discord_control":
            payload = event.get("response") or {}
            action = payload.get("action")
            if action == "leave":
                await self.leave_channel()
            elif action == "join":
                await self.join_configured_channel()
            return

        if msg == "audio_output_mode":
            payload = event.get("response") or {}
            mode = payload.get("mode")
            if mode == "local":
                await self.leave_channel()
            elif mode == "discord":
                await self.join_configured_channel()
            return

        if msg != "response":
            return

        # Воспроизводим звук только если подключены к голосовому каналу
        if not self.vc or not self.vc.is_connected():
            return

        payload = event.get("response") or {}
        result = payload.get("result") or {}
        job_id = payload.get("job_id")

        if result.get("event") == "stop_audio":
            self.stop_audio()
            return

        encoded_audio = result.get("audio_bytes")
        if encoded_audio and job_id:
            if self.audio_job_id and self.audio_job_id != job_id:
                self.stop_audio()
            if self.audio_source is None:
                self.audio_source = StreamingPCMSource()
                self.audio_job_id = job_id
            try:
                self.audio_source.add_pcm(
                    base64.b64decode(encoded_audio),
                    int(result.get("sr", 44100)),
                    int(result.get("sw", 2)),
                    int(result.get("ch", 1)),
                )
            except (TypeError, ValueError):
                logging.warning("Received invalid TTS audio event")
                return
            self.start_audio_if_ready()

        if payload.get("finished") and job_id == self.audio_job_id and self.audio_source:
            self.audio_source.close_input()
            self.start_audio_if_ready()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        self.stop_audio()
        if self.status_heartbeat_task:
            self.status_heartbeat_task.cancel()
        if self.vc and self.vc.is_connected():
            try:
                self.vc.stop_recording()
            except Exception:
                pass
        if self.sink:
            self.sink.cleanup()
        if self.vc and self.vc.is_connected():
            await self.vc.disconnect(force=True)
        if self.ws_task:
            self.ws_task.cancel()
        await super().close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nira Discord voice bridge (Pycord + DAVE)")
    parser.add_argument("--guild-id", required=True)
    parser.add_argument("--voice-channel-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--source-id", default="discord")
    parser.add_argument("--silence-ms", type=int, default=800)
    parser.add_argument("--min-speech-ms", type=int, default=220)
    parser.add_argument("--no-auto-join", dest="auto_join", action="store_false")
    parser.set_defaults(auto_join=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Фильтруем спам от Pycord про неизвестные RTCP пакеты
    class NoRTCPSpamFilter(logging.Filter):
        def filter(self, record):
            return "unexpected rtcp packet type=200" not in record.getMessage()
            
    for handler in logging.root.handlers:
        handler.addFilter(NoRTCPSpamFilter())
        
    # Глушим спам от низкоуровневых модулей
    logging.getLogger("discord.gateway").setLevel(logging.WARNING)
    logging.getLogger("discord.voice_client").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN is not set")

    NiraDiscordBot(parse_args()).run(token)


if __name__ == "__main__":
    main()
