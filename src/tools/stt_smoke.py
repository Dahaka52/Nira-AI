import argparse
import base64
import sys
import time
import uuid
import wave
from pathlib import Path

import requests


def read_wav_bytes(path: Path):
    with wave.open(str(path), "rb") as wav:
        ch = int(wav.getnchannels())
        sr = int(wav.getframerate())
        sw = int(wav.getsampwidth())
        frames = wav.readframes(wav.getnframes())
    return frames, sr, sw, ch


def main():
    parser = argparse.ArgumentParser(description="Smoke/Burst test for immediate STT HTTP path.")
    parser.add_argument("--host", default="http://127.0.0.1:7272", help="Server host")
    parser.add_argument("--wav", required=True, help="Path to WAV PCM file")
    parser.add_argument("--user", default="Creator")
    parser.add_argument("--source-id", default="mic")
    parser.add_argument("--burst", type=int, default=1, help="How many audio requests to send")
    parser.add_argument("--interval-ms", type=int, default=40, help="Delay between requests")
    parser.add_argument("--speech-start", action="store_true", help="Send speech_start before each audio segment")
    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    if not wav_path.exists():
        print(f"[ERROR] WAV file not found: {wav_path}")
        return 2

    audio_bytes, sr, sw, ch = read_wav_bytes(wav_path)
    if sw != 2:
        print(f"[WARN] sample width is {sw} bytes; backend expects 16-bit PCM for Sherpa.")
    if ch < 1:
        print("[ERROR] invalid channel count")
        return 2

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_url = args.host.rstrip("/") + "/api/context/conversation/audio"
    speech_start_url = args.host.rstrip("/") + "/api/context/conversation/speech_start"

    accepted = 0
    dropped = 0
    failures = 0
    started = time.time()

    with requests.Session() as s:
        for idx in range(args.burst):
            turn_id = str(uuid.uuid4())
            utterance_id = str(uuid.uuid4())
            payload = {
                "user": args.user,
                "timestamp": time.time(),
                "audio_bytes": audio_b64,
                "sr": sr,
                "sw": sw,
                "ch": ch,
                "source_id": args.source_id,
                "turn_id": turn_id,
                "utterance_id": utterance_id,
            }

            if args.speech_start:
                try:
                    s.post(
                        speech_start_url,
                        json={"timestamp": time.time(), "source_id": args.source_id, "turn_id": turn_id},
                        timeout=1.0,
                    )
                except Exception as exc:
                    print(f"[WARN] speech_start failed for #{idx + 1}: {exc}")

            try:
                res = s.post(audio_url, json=payload, timeout=5.0)
                if res.status_code != 200:
                    failures += 1
                    print(f"[FAIL] #{idx + 1}: HTTP {res.status_code}")
                else:
                    body = res.json()
                    is_accepted = bool((body or {}).get("response", {}).get("accepted", True))
                    if is_accepted:
                        accepted += 1
                        print(f"[OK]   #{idx + 1}: accepted")
                    else:
                        dropped += 1
                        reason = (body or {}).get("response", {}).get("drop_reason", "backpressure")
                        print(f"[DROP] #{idx + 1}: {reason}")
            except Exception as exc:
                failures += 1
                print(f"[FAIL] #{idx + 1}: {exc}")

            if idx + 1 < args.burst:
                time.sleep(max(0.0, args.interval_ms / 1000.0))

    elapsed = max(0.001, time.time() - started)
    print("")
    print("[SUMMARY]")
    print(f"sent={args.burst} accepted={accepted} dropped={dropped} failures={failures} elapsed_s={elapsed:.3f}")
    print(f"rps={args.burst / elapsed:.2f}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
