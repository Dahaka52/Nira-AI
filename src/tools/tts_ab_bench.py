import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    error: str
    ttfb_ms: float
    max_gap_ms: float
    total_ms: float
    audio_s: float
    rtf: float
    bytes_total: int
    sample_rate: int
    emit_every_frames: int
    decode_window_frames: int
    first_chunk_emit_every: int
    first_chunk_decode_window: int
    first_chunk_frames: int
    max_new_tokens: int
    max_frames: int
    overlap_samples: int
    use_optimized_decode: bool
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int


def _run_stream_once(session: requests.Session, url: str, payload: Dict[str, Any], read_chunk: int) -> Dict[str, Any]:
    started = time.perf_counter()
    first_chunk_at: Optional[float] = None
    last_chunk_at: Optional[float] = None
    max_gap_ms = 0.0
    total_bytes = 0
    sample_rate = int(payload.get("sample_rate", 24000) or 24000)
    sample_width = int(payload.get("sample_width", 2) or 2)
    channels = int(payload.get("channels", 1) or 1)

    with session.post(url, json=payload, stream=True, timeout=(10.0, 120.0)) as resp:
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
            now = time.perf_counter()
            if first_chunk_at is None:
                first_chunk_at = now
            if last_chunk_at is not None:
                gap_ms = (now - last_chunk_at) * 1000.0
                if gap_ms > max_gap_ms:
                    max_gap_ms = gap_ms
            last_chunk_at = now
            total_bytes += len(raw)

    ended = time.perf_counter()
    total_s = max(0.0, ended - started)
    audio_s = 0.0
    if sample_rate > 0 and sample_width > 0 and channels > 0:
        audio_s = float(total_bytes) / float(sample_rate * sample_width * channels)

    ttfb_ms = -1.0
    if first_chunk_at is not None:
        ttfb_ms = (first_chunk_at - started) * 1000.0

    rtf = -1.0
    if audio_s > 0:
        rtf = total_s / audio_s

    return {
        "ttfb_ms": ttfb_ms,
        "max_gap_ms": max_gap_ms,
        "total_ms": total_s * 1000.0,
        "audio_s": audio_s,
        "rtf": rtf,
        "bytes_total": int(total_bytes),
        "sample_rate": int(sample_rate),
    }


def _default_matrix() -> List[Dict[str, Any]]:
    # Conservative matrix for current voice_clone profile.
    # Keep overlap=0 to avoid boundary artifacts until throughput is fixed.
    return [
        {
            "case_id": "baseline_e4_d48_f1_24_24",
            "emit_every_frames": 4,
            "decode_window_frames": 48,
            "first_chunk_emit_every": 1,
            "first_chunk_decode_window": 24,
            "first_chunk_frames": 24,
            "max_new_tokens": 112,
            "max_frames": 80,
            "overlap_samples": 0,
            "use_optimized_decode": True,
            "do_sample": False,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
        },
        {
            "case_id": "e6_d64_f2_32_32",
            "emit_every_frames": 6,
            "decode_window_frames": 64,
            "first_chunk_emit_every": 2,
            "first_chunk_decode_window": 32,
            "first_chunk_frames": 32,
            "max_new_tokens": 112,
            "max_frames": 80,
            "overlap_samples": 0,
            "use_optimized_decode": True,
            "do_sample": False,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
        },
        {
            "case_id": "e8_d80_f2_48_48",
            "emit_every_frames": 8,
            "decode_window_frames": 80,
            "first_chunk_emit_every": 2,
            "first_chunk_decode_window": 48,
            "first_chunk_frames": 48,
            "max_new_tokens": 112,
            "max_frames": 80,
            "overlap_samples": 0,
            "use_optimized_decode": True,
            "do_sample": False,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
        },
        {
            "case_id": "e4_d64_f2_32_32",
            "emit_every_frames": 4,
            "decode_window_frames": 64,
            "first_chunk_emit_every": 2,
            "first_chunk_decode_window": 32,
            "first_chunk_frames": 32,
            "max_new_tokens": 112,
            "max_frames": 80,
            "overlap_samples": 0,
            "use_optimized_decode": True,
            "do_sample": False,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
        },
        {
            "case_id": "e6_d80_f3_48_48",
            "emit_every_frames": 6,
            "decode_window_frames": 80,
            "first_chunk_emit_every": 3,
            "first_chunk_decode_window": 48,
            "first_chunk_frames": 48,
            "max_new_tokens": 112,
            "max_frames": 80,
            "overlap_samples": 0,
            "use_optimized_decode": True,
            "do_sample": False,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark for Qwen3 TTS sidecar.")
    parser.add_argument("--url", default="http://127.0.0.1:6116/v1/tts/stream")
    parser.add_argument("--text", default="Привет! Какой у тебя любимый компьютерный персонаж?")
    parser.add_argument("--speaker", default="serena")
    parser.add_argument("--language", default="russian")
    parser.add_argument("--voice-mode", dest="voice_mode", default="voice_clone")
    parser.add_argument("--ref-audio-path", dest="ref_audio_path", default="C:\\Nirmita\\Nira_voice.wav")
    parser.add_argument("--ref-text", dest="ref_text", default="")
    parser.add_argument("--x-vector-only-mode", dest="x_vector_only_mode", type=int, default=1)
    parser.add_argument("--model-id", dest="model_id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--provider", default="cuda")
    parser.add_argument("--gpu-id", dest="gpu_id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", dest="attn_implementation", default="flash_attention_2")
    parser.add_argument("--sample-rate", dest="sample_rate", type=int, default=24000)
    parser.add_argument("--sample-width", dest="sample_width", type=int, default=2)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--read-chunk", dest="read_chunk", type=int, default=16384)
    parser.add_argument("--runs-per-case", dest="runs_per_case", type=int, default=3)
    parser.add_argument("--discard-first", dest="discard_first", type=int, default=1)
    parser.add_argument("--matrix-json", dest="matrix_json", default="")
    parser.add_argument("--out-prefix", dest="out_prefix", default="tts_ab_round3")
    args = parser.parse_args()

    if args.matrix_json:
        matrix = json.loads(Path(args.matrix_json).read_text(encoding="utf-8"))
    else:
        matrix = _default_matrix()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = Path(f"{args.out_prefix}_{ts}.json").resolve()
    out_csv = Path(f"{args.out_prefix}_{ts}.csv").resolve()

    results: List[CaseResult] = []
    with requests.Session() as session:
        for case in matrix:
            case_id = str(case["case_id"])
            print(f"[CASE] {case_id}")
            run_metrics: List[Dict[str, Any]] = []
            err = ""
            ok = True
            for i in range(1, int(args.runs_per_case) + 1):
                payload = {
                    "text": args.text,
                    "speaker": args.speaker,
                    "language": args.language,
                    "voice_mode": args.voice_mode,
                    "ref_audio_path": args.ref_audio_path,
                    "ref_text": args.ref_text,
                    "x_vector_only_mode": bool(int(args.x_vector_only_mode)),
                    "provider": args.provider,
                    "gpu_id": int(args.gpu_id),
                    "model_id": args.model_id,
                    "device": args.device,
                    "dtype": args.dtype,
                    "attn_implementation": args.attn_implementation,
                    "sample_rate": int(args.sample_rate),
                    "sample_width": int(args.sample_width),
                    "channels": int(args.channels),
                    "emit_every_frames": int(case["emit_every_frames"]),
                    "decode_window_frames": int(case["decode_window_frames"]),
                    "first_chunk_emit_every": int(case["first_chunk_emit_every"]),
                    "first_chunk_decode_window": int(case["first_chunk_decode_window"]),
                    "first_chunk_frames": int(case["first_chunk_frames"]),
                    "overlap_samples": int(case["overlap_samples"]),
                    "max_new_tokens": int(case["max_new_tokens"]),
                    "max_frames": int(case["max_frames"]),
                    "use_optimized_decode": bool(case["use_optimized_decode"]),
                    "do_sample": bool(case["do_sample"]),
                    "temperature": float(case["temperature"]),
                    "top_p": float(case["top_p"]),
                    "top_k": int(case["top_k"]),
                }
                try:
                    m = _run_stream_once(session, args.url, payload, int(args.read_chunk))
                    run_metrics.append(m)
                    print(
                        f"  [RUN {i}] ttfb={m['ttfb_ms']:.1f}ms gap={m['max_gap_ms']:.1f}ms "
                        f"total={m['total_ms']:.1f}ms audio={m['audio_s']:.3f}s rtf={m['rtf']:.3f}"
                    )
                except Exception as e:
                    ok = False
                    err = str(e)
                    print(f"  [RUN {i}] FAIL: {e}")
                    break

            if not run_metrics:
                results.append(
                    CaseResult(
                        case_id=case_id,
                        ok=False,
                        error=err or "no-runs",
                        ttfb_ms=-1.0,
                        max_gap_ms=-1.0,
                        total_ms=-1.0,
                        audio_s=-1.0,
                        rtf=-1.0,
                        bytes_total=0,
                        sample_rate=0,
                        emit_every_frames=int(case["emit_every_frames"]),
                        decode_window_frames=int(case["decode_window_frames"]),
                        first_chunk_emit_every=int(case["first_chunk_emit_every"]),
                        first_chunk_decode_window=int(case["first_chunk_decode_window"]),
                        first_chunk_frames=int(case["first_chunk_frames"]),
                        max_new_tokens=int(case["max_new_tokens"]),
                        max_frames=int(case["max_frames"]),
                        overlap_samples=int(case["overlap_samples"]),
                        use_optimized_decode=bool(case["use_optimized_decode"]),
                        do_sample=bool(case["do_sample"]),
                        temperature=float(case["temperature"]),
                        top_p=float(case["top_p"]),
                        top_k=int(case["top_k"]),
                    )
                )
                continue

            start_idx = min(max(0, int(args.discard_first)), len(run_metrics))
            usable = run_metrics[start_idx:] or run_metrics

            def avg(k: str) -> float:
                return float(sum(float(x[k]) for x in usable) / len(usable))

            results.append(
                CaseResult(
                    case_id=case_id,
                    ok=ok,
                    error=err,
                    ttfb_ms=avg("ttfb_ms"),
                    max_gap_ms=avg("max_gap_ms"),
                    total_ms=avg("total_ms"),
                    audio_s=avg("audio_s"),
                    rtf=avg("rtf"),
                    bytes_total=int(avg("bytes_total")),
                    sample_rate=int(avg("sample_rate")),
                    emit_every_frames=int(case["emit_every_frames"]),
                    decode_window_frames=int(case["decode_window_frames"]),
                    first_chunk_emit_every=int(case["first_chunk_emit_every"]),
                    first_chunk_decode_window=int(case["first_chunk_decode_window"]),
                    first_chunk_frames=int(case["first_chunk_frames"]),
                    max_new_tokens=int(case["max_new_tokens"]),
                    max_frames=int(case["max_frames"]),
                    overlap_samples=int(case["overlap_samples"]),
                    use_optimized_decode=bool(case["use_optimized_decode"]),
                    do_sample=bool(case["do_sample"]),
                    temperature=float(case["temperature"]),
                    top_p=float(case["top_p"]),
                    top_k=int(case["top_k"]),
                )
            )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    headers = list(asdict(results[0]).keys()) if results else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    print("")
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_csv}")

    ok_rows = [r for r in results if r.ok and r.rtf > 0]
    if ok_rows:
        ranked = sorted(ok_rows, key=lambda x: (x.rtf, x.max_gap_ms, x.ttfb_ms))
        print("")
        print("[TOP by rtf/max_gap/ttfb]")
        for r in ranked[:5]:
            print(
                f"  {r.case_id}: rtf={r.rtf:.3f} gap={r.max_gap_ms:.1f}ms "
                f"ttfb={r.ttfb_ms:.1f}ms total={r.total_ms:.1f}ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

