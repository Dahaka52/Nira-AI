import argparse
import statistics
import time
from typing import Dict, Optional

import requests


def _run_once(session: requests.Session, url: str, payload: Dict, read_chunk: int) -> Dict[str, Optional[float]]:
    started = time.perf_counter()
    first_chunk_at = None
    total_bytes = 0
    sample_rate = int(payload.get("sample_rate", 24000) or 24000)
    sample_width = int(payload.get("sample_width", 2) or 2)
    channels = int(payload.get("channels", 1) or 1)

    with session.post(url, json=payload, stream=True, timeout=(10.0, 90.0)) as resp:
        resp.raise_for_status()
        try:
            hdr_sr = int(resp.headers.get("x-sample-rate", "0") or "0")
            if hdr_sr > 0:
                sample_rate = hdr_sr
        except Exception:
            pass

        for raw in resp.iter_content(chunk_size=read_chunk):
            if not raw:
                continue
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            total_bytes += len(raw)

    ended = time.perf_counter()
    total_s = max(0.0, ended - started)
    audio_s = 0.0
    if sample_rate > 0 and sample_width > 0 and channels > 0:
        audio_s = float(total_bytes) / float(sample_rate * sample_width * channels)

    ttfb_ms = None
    if first_chunk_at is not None:
        ttfb_ms = (first_chunk_at - started) * 1000.0

    rtf = None
    if audio_s > 0:
        rtf = total_s / audio_s

    return {
        "ttfb_ms": ttfb_ms,
        "total_ms": total_s * 1000.0,
        "audio_s": audio_s,
        "rtf": rtf,
        "bytes": float(total_bytes),
        "sr": float(sample_rate),
    }


def _fmt(v: Optional[float], ndigits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{ndigits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick TTS sidecar smoke benchmark (TTFB/RTF).")
    parser.add_argument("--url", default="http://127.0.0.1:6116/v1/tts/stream", help="TTS stream endpoint")
    parser.add_argument("--text", default="Привет, как дела?", help="Input text")
    parser.add_argument("--language", default="russian")
    parser.add_argument("--voice-mode", dest="voice_mode", default="voice_clone", help="voice_clone")
    parser.add_argument("--ref-audio-path", dest="ref_audio_path", default="C:\\Nirmita\\Nira_voice.wav", help="Path to reference WAV")
    parser.add_argument("--ref-text", dest="ref_text", default="", help="Optional transcript for reference WAV")
    parser.add_argument("--x-vector-only-mode", dest="x_vector_only_mode", type=int, default=1, help="0|1")
    parser.add_argument("--model-id", dest="model_id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--provider", default="cuda", help="cpu|cuda")
    parser.add_argument("--gpu-id", dest="gpu_id", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", dest="attn_implementation", default="sdpa")
    parser.add_argument("--sample-rate", dest="sample_rate", type=int, default=24000)
    parser.add_argument("--sample-width", dest="sample_width", type=int, default=2)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--emit-every-frames", dest="emit_every_frames", type=int, default=8)
    parser.add_argument("--decode-window-frames", dest="decode_window_frames", type=int, default=64)
    parser.add_argument("--overlap-samples", dest="overlap_samples", type=int, default=512)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=56)
    parser.add_argument("--do-sample", dest="do_sample", type=int, default=0, help="0|1")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--top-k", dest="top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--runs", type=int, default=3, help="How many runs")
    parser.add_argument("--discard-first", dest="discard_first", type=int, default=1, help="Ignore first run in AVG")
    parser.add_argument("--read-chunk", dest="read_chunk", type=int, default=8192)
    args = parser.parse_args()

    payload = {
        "text": args.text,
        "language": args.language,
        "voice_mode": args.voice_mode,
        "ref_audio_path": args.ref_audio_path,
        "ref_text": args.ref_text,
        "x_vector_only_mode": bool(int(args.x_vector_only_mode)),
        "provider": args.provider,
        "gpu_id": args.gpu_id,
        "model_id": args.model_id,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "sample_rate": args.sample_rate,
        "sample_width": args.sample_width,
        "channels": args.channels,
        "emit_every_frames": args.emit_every_frames,
        "decode_window_frames": args.decode_window_frames,
        "overlap_samples": args.overlap_samples,
        "max_frames": args.max_frames,
        "do_sample": bool(int(args.do_sample)),
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
    }
    if str(args.voice_mode).strip().lower() == "voice_clone" and not str(args.ref_audio_path).strip():
        raise SystemExit("--ref-audio-path is required when --voice-mode=voice_clone")

    print(f"[TTS_SMOKE] endpoint={args.url}")
    print(f"[TTS_SMOKE] runs={args.runs} discard_first={args.discard_first}")
    print(
        "[TTS_SMOKE] params: "
        f"emit={args.emit_every_frames} decode={args.decode_window_frames} "
        f"overlap={args.overlap_samples} max_frames={args.max_frames}"
    )

    results = []
    with requests.Session() as session:
        for i in range(1, args.runs + 1):
            try:
                run = _run_once(session, args.url, payload, args.read_chunk)
                results.append(run)
                print(
                    f"[RUN {i}] ttfb_ms={_fmt(run['ttfb_ms'])} "
                    f"total_ms={_fmt(run['total_ms'])} audio_s={_fmt(run['audio_s'], 3)} "
                    f"rtf={_fmt(run['rtf'], 3)} bytes={int(run['bytes'] or 0)} sr={int(run['sr'] or 0)}"
                )
            except Exception as exc:
                print(f"[RUN {i}] FAIL: {exc}")
                return 1

    start_idx = min(max(0, args.discard_first), len(results))
    usable = results[start_idx:]
    if not usable:
        usable = results

    def collect(key: str):
        vals = [float(r[key]) for r in usable if r.get(key) is not None]
        return vals

    ttfb_vals = collect("ttfb_ms")
    rtf_vals = collect("rtf")
    total_vals = collect("total_ms")

    print("")
    print("[SUMMARY]")
    if ttfb_vals:
        print(f"avg_ttfb_ms={statistics.mean(ttfb_vals):.2f} p95_ttfb_ms={max(ttfb_vals):.2f}")
    else:
        print("avg_ttfb_ms=n/a")
    if total_vals:
        print(f"avg_total_ms={statistics.mean(total_vals):.2f}")
    else:
        print("avg_total_ms=n/a")
    if rtf_vals:
        print(f"avg_rtf={statistics.mean(rtf_vals):.3f} best_rtf={min(rtf_vals):.3f} worst_rtf={max(rtf_vals):.3f}")
    else:
        print("avg_rtf=n/a")

    print("")
    print("Interpretation:")
    print("- RTF < 1.0: faster-than-realtime (target).")
    print("- RTF 1.0-1.5: usable with small pauses.")
    print("- RTF > 2.0: noticeable stutter/gaps in live dialogue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
