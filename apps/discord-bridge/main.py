"""Nira's Discord voice bridge.

Discord voice runs in a separate process. It sends per-user PCM speech to the
existing immediate STT endpoint and plays TTS PCM from JAIson's WebSocket.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import json
import logging
import os
from pathlib import Path
import threading
import time

import discord
from discord.ext import voice_recv
import httpx
import websockets


SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
FRAME_BYTES = 3_840  # 20 ms of 48 kHz, 16-bit stereo PCM

# STT target format: 16kHz mono 16-bit PCM
STT_SAMPLE_RATE = 16_000
STT_CHANNELS = 1
STT_SAMPLE_WIDTH = 2


class StreamingPCMSource(discord.AudioSource):
    """A live PCM buffer consumed by discord.py's audio-player thread."""

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


class SpeakerBuffer:
    def __init__(self, member, loop, silence_s, min_bytes, callback):
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
        
        # Запускаем один фоновый поток для проверки таймаута
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def append(self, pcm: bytes) -> None:
        with self.lock:
            if not self.audio:
                self.started_at = time.time()
            if len(pcm) % 4 != 0:
                import logging
                logging.warning(f"PCM length {len(pcm)} is not a multiple of 4! Byte alignment corrupted!")
            self.audio.extend(pcm)
            self.last_packet_time = time.time()

    def _monitor_loop(self):
        while not self._closing:
            time.sleep(0.1)  # Проверяем каждые 100мс
            with self.lock:
                if not self.audio:
                    continue
                # Если прошло больше silence_s с последнего пакета
                if time.time() - self.last_packet_time >= self.silence_s:
                    self._do_flush()

    def _do_flush(self) -> None:
        # Внутренний метод, вызывается под локом
        total = len(self.audio)
        if total >= self.min_bytes:
            audio, started_at = bytes(self.audio), self.started_at
            self.loop.call_soon_threadsafe(self.callback, self.member, audio, started_at)
        self.audio.clear()

    def close(self) -> None:
        self._closing = True
        with self.lock:
            self._do_flush()


class DiscordInputSink(voice_recv.AudioSink):
    def __init__(self, bot: "NiraDiscordBot") -> None:
        super().__init__()
        self.bot = bot
        self.buffers: dict[int, SpeakerBuffer] = {}
        self.lock = threading.Lock()
        self._write_call_count = 0

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData) -> None:
        # === ДИАГНОСТИКА: трассируем каждый вызов write() ===
        self._write_call_count += 1
        if self._write_call_count <= 5 or self._write_call_count % 100 == 0:
            logging.info(
                "[SINK] write() call #%d: user=%s bot=%s",
                self._write_call_count,
                repr(user),
                repr(self.bot.user),
            )

        if user is None:
            if self._write_call_count <= 5:
                logging.warning("[SINK] user is None — SSRC→Member resolve failed! Нужен intents.members")
            return
        if self.bot.user is None:
            if self._write_call_count <= 5:
                logging.warning("[SINK] bot.user is None — бот ещё не аутентифицирован")
            return
        if user.id == self.bot.user.id:
            return  # игнорируем собственный голос

        pcm = bytes(data.pcm)
        if not pcm:
            logging.debug("[SINK] пустой PCM от %s", user)
            return

        logging.debug("[SINK] PCM от %s: %d bytes", user, len(pcm))
        with self.lock:
            buffer = self.buffers.get(user.id)
            if buffer is None:
                minimum = int(SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * self.bot.min_speech_ms / 1000)
                logging.info("[SINK] Создаём буфер для %s (min_bytes=%d)", user, minimum)
                buffer = SpeakerBuffer(
                    user,
                    self.bot.loop,
                    self.bot.silence_ms / 1000,
                    minimum,
                    self.bot.submit_speech_from_thread,
                )
                self.buffers[user.id] = buffer
        buffer.append(pcm)

    def cleanup(self) -> None:
        with self.lock:
            for buffer in self.buffers.values():
                buffer.close()
            self.buffers.clear()


class NiraDiscordBot(discord.Client):
    def __init__(self, args: argparse.Namespace) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.members = True  # нужен для разрешения SSRC→Member в voice_recv
        super().__init__(intents=intents)
        self.guild_id = int(args.guild_id)
        self.channel_id = int(args.voice_channel_id)
        self.api_url = args.api_url
        self.ws_url = args.ws_url
        self.source_id = args.source_id
        self.silence_ms = max(100, args.silence_ms)
        self.min_speech_ms = max(100, args.min_speech_ms)
        self.auto_join = args.auto_join
        self.vc: voice_recv.VoiceRecvClient | None = None
        self.sink: DiscordInputSink | None = None
        self.ws_task: asyncio.Task | None = None
        self.audio_source: StreamingPCMSource | None = None
        self.audio_job_id: str | None = None
        self.ready_once = False
        self._load_opus()

    def _load_opus(self) -> None:
        if discord.opus.is_loaded():
            return
        opus_path = Path(discord.__file__).resolve().parent / "bin" / "libopus-0.x64.dll"
        if not opus_path.is_file():
            raise RuntimeError(f"Opus DLL is missing: {opus_path}")
        discord.opus.load_opus(str(opus_path))
        if not discord.opus.is_loaded():
            raise RuntimeError("Discord Opus DLL could not be loaded")

    async def on_ready(self) -> None:
        if self.ready_once:
            return
        self.ready_once = True
        logging.info("Authenticated as %s", self.user)
        self.ws_task = asyncio.create_task(self.event_listener())
        if self.auto_join:
            await self.join_configured_channel()

    async def join_configured_channel(self) -> None:
        channel = self.get_channel(self.channel_id) or await self.fetch_channel(self.channel_id)
        if not isinstance(channel, discord.VoiceChannel):
            raise RuntimeError(f"Channel {self.channel_id} is not a voice channel")
        if channel.guild.id != self.guild_id:
            raise RuntimeError("Configured channel belongs to another server")
        if self.vc and self.vc.is_connected():
            if self.vc.channel.id != channel.id:
                await self.vc.move_to(channel)
            return
        # Отключаем прием аудио через voice_recv, так как он несовместим с DAVE.
        # Оставляем только передачу TTS (self_deaf=True)
        self.vc = await channel.connect(reconnect=True, self_deaf=True)
        # self.sink = DiscordInputSink(self)
        # self.vc.listen(self.sink)
        logging.info("Joined voice channel %s (%s)", channel.name, channel.id)

    def submit_speech_from_thread(self, member, audio: bytes, timestamp: float) -> None:
        asyncio.create_task(self.submit_speech(member, audio, timestamp))

    async def submit_speech(self, member, audio: bytes, timestamp: float) -> None:
        import wave
        with wave.open(f"C:\\Nirmita\\scratch\\discord_bridge_raw.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio)

        payload = {
            "user": member.display_name or member.name,
            "speaker_id": str(member.id),
            "timestamp": timestamp,
            "audio_bytes": base64.b64encode(audio).decode('ascii'),
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
                "Submitted Discord speech from %s (%d bytes raw 48kHz stereo)",
                payload["user"], len(audio)
            )
        except Exception:
            logging.exception("Could not send Discord speech to JAIson")

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
        if event.get("message") != "response":
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
                self.audio_source, self.audio_job_id = StreamingPCMSource(), job_id
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

    async def close(self) -> None:
        self.stop_audio()
        if self.sink:
            self.sink.cleanup()
        if self.vc and self.vc.is_connected():
            await self.vc.disconnect(force=True)
        if self.ws_task:
            self.ws_task.cancel()
        await super().close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nira Discord voice bridge")
    parser.add_argument("--guild-id", required=True)
    parser.add_argument("--voice-channel-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--source-id", default="discord")
    parser.add_argument("--silence-ms", type=int, default=500)
    parser.add_argument("--min-speech-ms", type=int, default=220)
    parser.add_argument("--no-auto-join", dest="auto_join", action="store_false")
    parser.set_defaults(auto_join=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Отключаем спам RTCP пакетов
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.rtp").setLevel(logging.WARNING)

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN is not set")
    NiraDiscordBot(parse_args()).run(token, log_handler=None)


if __name__ == "__main__":
    main()
