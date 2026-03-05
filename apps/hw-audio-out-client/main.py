import argparse
import asyncio
import audioop
import base64
import json
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets


def _get_hostapi_name(hostapi_index: int) -> str:
    try:
        hostapis = sd.query_hostapis()
        if 0 <= hostapi_index < len(hostapis):
            return str(hostapis[hostapi_index]["name"])
    except Exception:
        pass
    return "unknown"


def get_output_devices() -> list[dict]:
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_output_channels", 0)) <= 0:
            continue
        hostapi_idx = int(dev.get("hostapi", -1))
        devices.append(
            {
                "index": idx,
                "name": str(dev.get("name", "")),
                "hostapi": _get_hostapi_name(hostapi_idx),
                "max_output_channels": int(dev.get("max_output_channels", 0)),
                "default_samplerate": int(dev.get("default_samplerate", 0)),
            }
        )
    return devices


def print_output_devices() -> None:
    output_devices = get_output_devices()
    if not output_devices:
        print("[AUDIO_OUT] Output devices not found.")
        return

    print("[AUDIO_OUT] Available output devices:")
    for dev in output_devices:
        print(
            f"  [{dev['index']}] {dev['name']} | hostapi={dev['hostapi']} | "
            f"channels={dev['max_output_channels']} | default_sr={dev['default_samplerate']}"
        )


def resolve_output_device_index(
    preferred_index: Optional[int],
    preferred_name: Optional[str],
    preferred_hostapi: Optional[str],
) -> int:
    output_devices = get_output_devices()
    if not output_devices:
        raise RuntimeError("No output devices available")

    if preferred_name:
        needle = preferred_name.strip().lower()
        hostapi_needle = (preferred_hostapi or "").strip().lower()
        matches = [d for d in output_devices if needle in d["name"].lower()]
        if hostapi_needle:
            hostapi_matches = [d for d in matches if hostapi_needle in d["hostapi"].lower()]
            if hostapi_matches:
                matches = hostapi_matches
        if matches:
            exact = [d for d in matches if d["name"].lower() == needle]
            selected = exact[0] if exact else matches[0]
            if len(matches) > 1:
                print(
                    f"[AUDIO_OUT] WARNING: {len(matches)} devices matched '{preferred_name}'. "
                    f"Using index {selected['index']}."
                )
            print(
                f"[AUDIO_OUT] Selected by name: [{selected['index']}] "
                f"{selected['name']} ({selected['hostapi']})"
            )
            return selected["index"]
        print(f"[AUDIO_OUT] WARNING: device_name '{preferred_name}' not found. Falling back.")

    if preferred_index is not None:
        for dev in output_devices:
            if dev["index"] == preferred_index:
                print(f"[AUDIO_OUT] Selected by index: [{dev['index']}] {dev['name']} ({dev['hostapi']})")
                return preferred_index
        print(f"[AUDIO_OUT] WARNING: device_index={preferred_index} is not a valid output device. Falling back.")

    default_idx = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    if isinstance(default_idx, int) and default_idx >= 0:
        for dev in output_devices:
            if dev["index"] == default_idx:
                print(
                    f"[AUDIO_OUT] Selected by system default: [{dev['index']}] "
                    f"{dev['name']} ({dev['hostapi']})"
                )
                return default_idx

    selected = output_devices[0]
    print(
        f"[AUDIO_OUT] Selected first available output: [{selected['index']}] "
        f"{selected['name']} ({selected['hostapi']})"
    )
    return selected["index"]


class PCM16OutputPlayer:
    def __init__(
        self,
        device_index: int,
        sample_rate: int,
        channels: int,
        blocksize: int,
        max_buffer_ms: int,
        prebuffer_ms: int,
        rebuffer_ms: int,
        rebuffer_max_wait_ms: int,
        chunk_fade_ms: int,
        start_fade_ms: int,
        reset_resampler_on_stream_start: bool,
        clear_on_stop: bool,
        output_gain_db: float,
    ):
        if channels != 1:
            raise ValueError("Only mono output is supported right now.")
        self.device_index = int(device_index)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_width = 2
        self.blocksize = int(max(64, blocksize))
        self.clear_on_stop = bool(clear_on_stop)
        self.frame_bytes = self.sample_width * self.channels
        self.max_buffer_ms = int(max(100, max_buffer_ms))
        self.max_buffer_bytes = int(self.sample_rate * self.frame_bytes * self.max_buffer_ms / 1000.0)
        self.prebuffer_ms = int(max(0, prebuffer_ms))
        self.rebuffer_ms = int(max(0, rebuffer_ms))
        self.rebuffer_max_wait_ms = int(max(0, rebuffer_max_wait_ms))
        self.chunk_fade_ms = int(max(0, chunk_fade_ms))
        self.start_fade_ms = int(max(0, start_fade_ms))
        self.reset_resampler_on_stream_start = bool(reset_resampler_on_stream_start)
        self.prebuffer_bytes = int(self.sample_rate * self.frame_bytes * self.prebuffer_ms / 1000.0)
        self.rebuffer_bytes = int(self.sample_rate * self.frame_bytes * self.rebuffer_ms / 1000.0)
        self.chunk_fade_samples = int(self.sample_rate * self.chunk_fade_ms / 1000.0)
        self.start_fade_samples = int(self.sample_rate * self.start_fade_ms / 1000.0)
        self.output_gain = float(10.0 ** (output_gain_db / 20.0))

        self._stream = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._last_underflow_print = 0.0
        self._last_rebuffer_print = 0.0
        self._last_overflow_print = 0.0
        self._debug_chunks_seen = 0
        self._ratecv_state = None
        self._ratecv_src_sr = None
        self._primed = False
        self._need_rebuffer = False
        self._need_rebuffer_since = 0.0
        self._stream_ended = False
        self._pending_stream_start_fade = False
        self._active_job_id: Optional[str] = None

    async def start(self) -> None:
        if self._stream is not None:
            return

        def callback(outdata, frames, _time_info, status):
            if status:
                now = time.time()
                if (now - self._last_underflow_print) > 2.0:
                    self._last_underflow_print = now
                    print(f"[AUDIO_OUT] Stream status: {status}")

            requested = int(frames) * self.frame_bytes
            chunk = b""
            if requested > 0:
                with self._lock:
                    if not self._primed and self.prebuffer_bytes > 0:
                        if len(self._buffer) < self.prebuffer_bytes:
                            outdata[:] = b"\x00" * requested
                            return
                        self._primed = True

                    if self._need_rebuffer and self.rebuffer_bytes > 0:
                        now = time.time()
                        if self._stream_ended and 0 < len(self._buffer) < requested:
                            # Final tail for this response: play once padded and finish.
                            tail = bytes(self._buffer)
                            self._buffer.clear()
                            self._need_rebuffer = False
                            self._need_rebuffer_since = 0.0
                            tail = tail + (b"\x00" * (requested - len(tail)))
                            outdata[:] = tail
                            return
                        # Prefer a true jitter-buffer resume target. The previous logic
                        # resumed too early (tiny buffer), which produced micro-bursts.
                        # Here we wait for rebuffer_bytes, and only relax target after
                        # rebuffer_max_wait_ms to avoid hard lock on very slow streams.
                        dynamic_resume_bytes = int(self.rebuffer_bytes)
                        waited_ms = (
                            (now - self._need_rebuffer_since) * 1000.0
                            if self._need_rebuffer_since > 0.0
                            else 0.0
                        )
                        stall_flush_ms = max(1200.0, float(self.rebuffer_max_wait_ms) * 3.0)
                        if len(self._buffer) > 0 and waited_ms >= stall_flush_ms:
                            # Hard-stall guard: if stream-end signal was missed, do not keep
                            # tiny residual tails in rebuffer forever.
                            tail = bytes(self._buffer)
                            self._buffer.clear()
                            self._need_rebuffer = False
                            self._need_rebuffer_since = 0.0
                            if len(tail) < requested:
                                tail = tail + (b"\x00" * (requested - len(tail)))
                            else:
                                tail = tail[:requested]
                            print(
                                f"[AUDIO_OUT] Rebuffer stall guard flushed tail: "
                                f"{len(tail)} bytes after {int(waited_ms)}ms wait"
                            )
                            outdata[:] = tail
                            return
                        if self.rebuffer_max_wait_ms > 0 and waited_ms >= self.rebuffer_max_wait_ms:
                            dynamic_resume_bytes = max(
                                int(self.rebuffer_bytes * 0.35),
                                max(requested * 8, self.frame_bytes * 1024),
                            )
                        if self._stream_ended and len(self._buffer) > 0:
                            # End-of-stream: do not wait for full rebuffer target.
                            dynamic_resume_bytes = min(dynamic_resume_bytes, requested)
                        if len(self._buffer) < dynamic_resume_bytes and (
                            self.rebuffer_max_wait_ms <= 0 or waited_ms < self.rebuffer_max_wait_ms
                        ):
                            if self._stream_ended and len(self._buffer) > 0:
                                # Flush tail on stream end to avoid getting stuck forever.
                                tail = bytes(self._buffer)
                                self._buffer.clear()
                                self._need_rebuffer = False
                                self._need_rebuffer_since = 0.0
                                if len(tail) < requested:
                                    tail = tail + (b"\x00" * (requested - len(tail)))
                                else:
                                    tail = tail[:requested]
                                outdata[:] = tail
                                return
                            now = time.time()
                            if (now - self._last_rebuffer_print) > 2.0:
                                self._last_rebuffer_print = now
                                print(
                                    f"[AUDIO_OUT] Rebuffering: "
                                    f"{len(self._buffer)}/{dynamic_resume_bytes} bytes"
                                )
                            outdata[:] = b"\x00" * requested
                            return
                        self._need_rebuffer = False
                        self._need_rebuffer_since = 0.0

                    if len(self._buffer) >= requested:
                        chunk = bytes(self._buffer[:requested])
                        del self._buffer[:requested]
                    elif len(self._buffer) > 0:
                        # If tail is almost a full callback block, play it once padded.
                        # This avoids sticky near-full tails like 1022/1024 bytes that can
                        # otherwise keep the player in repeated rebuffer state.
                        if len(self._buffer) >= int(requested * 0.75):
                            tail = bytes(self._buffer)
                            self._buffer.clear()
                            if len(tail) < requested:
                                tail = tail + (b"\x00" * (requested - len(tail)))
                            else:
                                tail = tail[:requested]
                            chunk = tail
                            self._need_rebuffer = True
                            if self._need_rebuffer_since <= 0.0:
                                self._need_rebuffer_since = time.time()
                        else:
                            # Do not flush tiny tail + zeros (it produces audible clicks).
                            # Keep buffered tail and resume only when rebuffer target is reached.
                            chunk = b"\x00" * requested
                            self._need_rebuffer = True
                            if self._need_rebuffer_since <= 0.0:
                                self._need_rebuffer_since = time.time()
                    else:
                        chunk = b"\x00" * requested
                        self._need_rebuffer = True
                        if self._need_rebuffer_since <= 0.0:
                            self._need_rebuffer_since = time.time()
                        if self._stream_ended:
                            # Nothing else is expected for this stream; avoid sticky rebuffer.
                            self._need_rebuffer = False
                            self._need_rebuffer_since = 0.0
            outdata[:] = chunk

        def _open_stream(sample_rate: int):
            return sd.RawOutputStream(
                device=self.device_index,
                samplerate=int(sample_rate),
                channels=self.channels,
                dtype="int16",
                blocksize=self.blocksize,
                latency="low",
                callback=callback,
            )

        try:
            self._stream = _open_stream(self.sample_rate)
            self._stream.start()
        except Exception as primary_err:
            fallback_sr = 0
            try:
                info = sd.query_devices(self.device_index, "output")
                fallback_sr = int(float(info.get("default_samplerate", 0)))
            except Exception:
                fallback_sr = 0

            if fallback_sr > 0 and fallback_sr != self.sample_rate:
                print(
                    f"[AUDIO_OUT] WARNING: sample_rate {self.sample_rate} not accepted by device. "
                    f"Falling back to {fallback_sr}."
                )
                self.sample_rate = fallback_sr
                self.max_buffer_bytes = int(self.sample_rate * self.frame_bytes * self.max_buffer_ms / 1000.0)
                self.prebuffer_bytes = int(self.sample_rate * self.frame_bytes * self.prebuffer_ms / 1000.0)
                self.rebuffer_bytes = int(self.sample_rate * self.frame_bytes * self.rebuffer_ms / 1000.0)
                self._stream = _open_stream(self.sample_rate)
                self._stream.start()
            else:
                raise primary_err

    async def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        await self.clear()

    async def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._primed = False
            self._need_rebuffer = False
            self._need_rebuffer_since = 0.0
            self._stream_ended = False
            self._pending_stream_start_fade = False
            self._active_job_id = None
        self._ratecv_state = None
        self._ratecv_src_sr = None

    async def begin_stream(self, job_id: Optional[str]) -> None:
        jid = str(job_id or "").strip() or None
        dropped_bytes = 0
        with self._lock:
            if jid and self._active_job_id and self._active_job_id != jid:
                dropped_bytes = len(self._buffer)
                if dropped_bytes > 0:
                    self._buffer.clear()
                self._primed = False
                self._need_rebuffer = False
                self._need_rebuffer_since = 0.0
            if jid:
                self._active_job_id = jid
            self._stream_ended = False
            self._pending_stream_start_fade = True
        if dropped_bytes > 0:
            print(f"[AUDIO_OUT] Switched response stream, dropped {dropped_bytes} pending bytes.")
        if self.reset_resampler_on_stream_start:
            self._ratecv_state = None
            self._ratecv_src_sr = None

    def is_job_active(self, job_id: Optional[str]) -> bool:
        jid = str(job_id or "").strip()
        if not jid:
            return True
        with self._lock:
            if self._active_job_id is None:
                self._active_job_id = jid
                return True
            return self._active_job_id == jid

    async def mark_stream_end(self, job_id: Optional[str] = None) -> None:
        jid = str(job_id or "").strip()
        with self._lock:
            if jid and self._active_job_id and self._active_job_id != jid:
                return
            self._stream_ended = True
            if len(self._buffer) <= 0:
                self._need_rebuffer = False
                self._need_rebuffer_since = 0.0

    def _apply_gain(self, pcm_bytes: bytes) -> bytes:
        if abs(self.output_gain - 1.0) < 1e-3:
            return pcm_bytes
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        arr *= self.output_gain
        np.clip(arr, -32768, 32767, out=arr)
        return arr.astype(np.int16).tobytes()

    def _apply_chunk_fade(self, pcm_bytes: bytes) -> bytes:
        """Apply short fade-in/out to reduce clicks when stream is sparse."""
        if self.chunk_fade_samples <= 0:
            return pcm_bytes
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n = int(arr.size)
        if n < 4:
            return pcm_bytes
        fade = min(self.chunk_fade_samples, n // 4)
        if fade <= 0:
            return pcm_bytes
        ramp = np.linspace(0.0, 1.0, num=fade, endpoint=True, dtype=np.float32)
        arr[:fade] *= ramp
        arr[-fade:] *= ramp[::-1]
        np.clip(arr, -32768, 32767, out=arr)
        return arr.astype(np.int16).tobytes()

    def _apply_start_fade(self, pcm_bytes: bytes) -> bytes:
        """One-shot fade-in for new phrase start to suppress leading click/pop."""
        if self.start_fade_samples <= 0:
            return pcm_bytes
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n = int(arr.size)
        if n <= 0:
            return pcm_bytes
        fade = min(self.start_fade_samples, n)
        if fade <= 0:
            return pcm_bytes
        ramp = np.linspace(0.0, 1.0, num=fade, endpoint=True, dtype=np.float32)
        arr[:fade] *= ramp
        np.clip(arr, -32768, 32767, out=arr)
        return arr.astype(np.int16).tobytes()

    @staticmethod
    def _resample_pcm16_mono(pcm_bytes: bytes, src_sr: int, dst_sr: int) -> bytes:
        if src_sr == dst_sr:
            return pcm_bytes
        if src_sr <= 0 or dst_sr <= 0:
            return b""
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if arr.size < 2:
            return pcm_bytes
        out_len = int(round(arr.size * float(dst_sr) / float(src_sr)))
        if out_len <= 0:
            return b""
        x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
        resampled = np.interp(x_new, x_old, arr)
        np.clip(resampled, -32768, 32767, out=resampled)
        return resampled.astype(np.int16).tobytes()

    async def push(self, pcm_bytes: bytes, sr: int, sw: int, ch: int) -> None:
        if not pcm_bytes:
            return
        if int(sw) != 2:
            return
        if int(ch) != 1:
            return

        src_sr = int(sr)
        if src_sr != self.sample_rate:
            if self._ratecv_src_sr != src_sr:
                self._ratecv_state = None
                self._ratecv_src_sr = src_sr
            try:
                pcm_bytes, self._ratecv_state = audioop.ratecv(
                    pcm_bytes,
                    2,  # sample width
                    1,  # mono
                    src_sr,
                    self.sample_rate,
                    self._ratecv_state,
                )
            except Exception:
                # Fallback to stateless resample if ratecv fails for any reason.
                pcm_bytes = self._resample_pcm16_mono(pcm_bytes, src_sr, self.sample_rate)
            if not pcm_bytes:
                return
        else:
            self._ratecv_state = None
            self._ratecv_src_sr = None

        pcm_bytes = self._apply_gain(pcm_bytes)
        apply_start_fade = False
        with self._lock:
            if self._pending_stream_start_fade:
                self._pending_stream_start_fade = False
                apply_start_fade = True
        if apply_start_fade:
            pcm_bytes = self._apply_start_fade(pcm_bytes)
        pcm_bytes = self._apply_chunk_fade(pcm_bytes)
        if not pcm_bytes:
            return

        if self._debug_chunks_seen < 3:
            try:
                arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
                peak = float(np.max(np.abs(arr))) if arr.size else 0.0
                print(
                    f"[AUDIO_OUT] PCM chunk #{self._debug_chunks_seen + 1}: "
                    f"bytes={len(pcm_bytes)} sr={self.sample_rate} rms={rms:.4f} peak={peak:.4f}"
                )
            except Exception:
                print(f"[AUDIO_OUT] PCM chunk #{self._debug_chunks_seen + 1}: bytes={len(pcm_bytes)} sr={self.sample_rate}")
            self._debug_chunks_seen += 1

        with self._lock:
            self._stream_ended = False
            free_bytes = self.max_buffer_bytes - len(self._buffer)
            if free_bytes <= 0:
                now = time.time()
                if (now - self._last_overflow_print) > 2.0:
                    self._last_overflow_print = now
                    print(
                        f"[AUDIO_OUT] Buffer overflow: queue full ({len(self._buffer)}/{self.max_buffer_bytes} bytes), "
                        "dropping incoming chunk."
                    )
                return

            aligned_free = free_bytes - (free_bytes % self.frame_bytes)
            if aligned_free <= 0:
                return

            if len(pcm_bytes) > aligned_free:
                dropped = len(pcm_bytes) - aligned_free
                pcm_bytes = pcm_bytes[:aligned_free]
                now = time.time()
                if (now - self._last_overflow_print) > 2.0:
                    self._last_overflow_print = now
                    print(
                        f"[AUDIO_OUT] Buffer overflow: dropped {dropped} incoming bytes "
                        f"(queue={len(self._buffer)}/{self.max_buffer_bytes})."
                    )

            self._buffer.extend(pcm_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws_url", type=str, default="ws://127.0.0.1:7272/")
    parser.add_argument("--device_index", type=int, default=None)
    parser.add_argument("--device_name", type=str, default=None)
    parser.add_argument("--device_hostapi", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--blocksize", type=int, default=256)
    parser.add_argument("--max_buffer_ms", type=int, default=1800)
    parser.add_argument("--prebuffer_ms", type=int, default=220)
    parser.add_argument("--rebuffer_ms", type=int, default=420)
    parser.add_argument("--rebuffer_max_wait_ms", type=int, default=350)
    parser.add_argument("--chunk_fade_ms", type=int, default=0)
    parser.add_argument("--start_fade_ms", type=int, default=2)
    parser.add_argument("--reset_resampler_on_stream_start", type=int, default=1)
    parser.add_argument("--clear_on_stop", type=int, default=1)
    parser.add_argument("--output_gain_db", type=float, default=0.0)
    parser.add_argument("--reconnect_delay_ms", type=int, default=1200)
    parser.add_argument("--list_devices", action="store_true")
    return parser.parse_args()


async def handle_ws_message(message: str, player: PCM16OutputPlayer) -> None:
    try:
        payload_raw = json.loads(message)
    except Exception:
        return

    # Quart WS server emits create_response(...) tuple -> JSON array [body, status]
    if isinstance(payload_raw, list):
        payload = payload_raw[0] if payload_raw else {}
    else:
        payload = payload_raw
    if not isinstance(payload, dict):
        return

    event_name = payload.get("message")
    if event_name != "response":
        return
    body = payload.get("response")
    if not isinstance(body, dict):
        return
    job_id = str(body.get("job_id", "")).strip() or None
    if body.get("start") is not None:
        await player.begin_stream(job_id)
        return
    # JAIson WS envelope for jobs uses {finished, success, result}. It does not expose
    # a nested "status" field, so treat finished=true as end-of-stream for response jobs.
    finished = bool(body.get("finished", False))
    if finished:
        await player.mark_stream_end(job_id=job_id)
        return
    status = str(body.get("status", "")).strip().lower()
    if status in ("success", "cancelled", "error"):
        await player.mark_stream_end(job_id=job_id)
        return
    result = body.get("result")
    if not isinstance(result, dict):
        return

    event_type = str(result.get("event", "")).strip().lower()
    if event_type == "stop_audio":
        if player.clear_on_stop:
            await player.clear()
        else:
            await player.mark_stream_end(job_id=job_id)
        return
    if event_type != "audio_chunk":
        return
    if not player.is_job_active(job_id):
        return

    # Backend emits audio payload as "audio_bytes". Keep "content" as legacy fallback.
    b64_chunk = result.get("audio_bytes")
    if not b64_chunk:
        b64_chunk = result.get("content")
    if not b64_chunk:
        return

    try:
        pcm_bytes = base64.b64decode(str(b64_chunk), validate=True)
    except Exception:
        return

    sr = int(result.get("sr", player.sample_rate) or player.sample_rate)
    sw = int(result.get("sw", 2) or 2)
    ch = int(result.get("ch", 1) or 1)
    await player.push(pcm_bytes, sr=sr, sw=sw, ch=ch)


async def run() -> None:
    args = parse_args()
    if args.list_devices:
        print_output_devices()
        return

    device_index = resolve_output_device_index(
        preferred_index=args.device_index,
        preferred_name=args.device_name,
        preferred_hostapi=args.device_hostapi,
    )
    player = PCM16OutputPlayer(
        device_index=device_index,
        sample_rate=args.sample_rate,
        channels=args.channels,
        blocksize=args.blocksize,
        max_buffer_ms=args.max_buffer_ms,
        prebuffer_ms=args.prebuffer_ms,
        rebuffer_ms=args.rebuffer_ms,
        rebuffer_max_wait_ms=args.rebuffer_max_wait_ms,
        chunk_fade_ms=args.chunk_fade_ms,
        start_fade_ms=args.start_fade_ms,
        reset_resampler_on_stream_start=bool(int(args.reset_resampler_on_stream_start)),
        clear_on_stop=bool(int(args.clear_on_stop)),
        output_gain_db=args.output_gain_db,
    )
    await player.start()

    try:
        dev_info = sd.query_devices(device_index, "output")
        hostapi_name = _get_hostapi_name(int(dev_info.get("hostapi", -1)))
        print(
            f"[AUDIO_OUT] Listening WS={args.ws_url} and streaming to "
            f"{dev_info['name']} (ID: {device_index}, hostapi: {hostapi_name}, sr={player.sample_rate})"
        )
    except Exception:
        print(f"[AUDIO_OUT] Listening WS={args.ws_url} and streaming to device index {device_index}")
    print("========================================================")

    reconnect_delay_s = max(0.2, float(args.reconnect_delay_ms) / 1000.0)
    try:
        while True:
            try:
                async with websockets.connect(
                    args.ws_url,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=2,
                ) as ws:
                    print("[AUDIO_OUT] WebSocket connected.")
                    async for msg in ws:
                        await handle_ws_message(msg, player)
            except KeyboardInterrupt:
                raise
            except Exception as err:
                # If WS drops mid-stream, mark end so callback can drain/flush tail
                # instead of staying in sticky rebuffer state.
                await player.mark_stream_end()
                print(f"[AUDIO_OUT] WS disconnected: {err}. Reconnect in {reconnect_delay_s:.1f}s.")
                await asyncio.sleep(reconnect_delay_s)
    finally:
        await player.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[AUDIO_OUT] Stopped by user.")
