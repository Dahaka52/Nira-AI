import argparse
import asyncio
import json
import re
import sys
import uuid
import wave
from pathlib import Path


def read_wav_bytes(path: Path):
    with wave.open(str(path), "rb") as wav:
        ch = int(wav.getnchannels())
        sr = int(wav.getframerate())
        sw = int(wav.getsampwidth())
        frames = wav.readframes(wav.getnframes())
    return frames, sr, sw, ch


def evaluate_case(case: dict, text: str) -> tuple[bool, str]:
    text_norm = text.lower().strip()
    if not text_norm:
        return False, "empty_transcript"

    expected_any = [str(x).lower() for x in (case.get("expected_any") or [])]
    expected_all = [str(x).lower() for x in (case.get("expected_all") or [])]
    expected_not = [str(x).lower() for x in (case.get("expected_not") or [])]
    expected_regex = case.get("expected_regex")

    if expected_any and not any(token in text_norm for token in expected_any):
        return False, f"expected_any not found: {expected_any}"
    if expected_all and not all(token in text_norm for token in expected_all):
        return False, f"expected_all not satisfied: {expected_all}"
    if expected_not and any(token in text_norm for token in expected_not):
        return False, f"expected_not violated: {expected_not}"
    if expected_regex:
        if not re.search(str(expected_regex), text, flags=re.IGNORECASE):
            return False, f"expected_regex mismatch: {expected_regex}"
    return True, "ok"


async def run_regression(config_name: str, manifest_path: Path, reuse_running_sidecar: bool) -> int:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # utils.args parses argv at import time; keep only script name to avoid unknown-arg exit.
    sys.argv = [sys.argv[0]]

    from utils.config import Config
    from utils.operations.manager import load_op, OpTypes

    cfg = Config()
    cfg.load_from_name(config_name)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    if not cases:
        print("[ERROR] manifest has no cases")
        return 2

    stt_ops = [op for op in (cfg.operations or []) if str(op.get("role", "")).lower() == "stt"]
    if not stt_ops:
        print("[ERROR] config has no STT operations")
        return 2

    active_id = cfg.stt_active_id
    selected = None
    if active_id:
        selected = next((op for op in stt_ops if str(op.get("id", "")) == str(active_id)), None)
    if selected is None:
        selected = stt_ops[0]
    stt_id = str(selected.get("id"))
    selected_cfg = dict(selected)

    # Useful when backend is already running: avoid launching a second Sherpa sidecar
    # on the same port and just reuse existing ws_url.
    if reuse_running_sidecar:
        selected_cfg["process_autostart"] = False

    stt_op = load_op(OpTypes.STT, stt_id, op_details=selected_cfg)
    await stt_op.configure(selected_cfg)
    await stt_op.start()

    failed = 0
    try:
        for idx, case in enumerate(cases, start=1):
            case_id = str(case.get("id", f"case_{idx}"))
            wav_path = Path(case["wav"])
            if not wav_path.is_absolute():
                wav_path = (manifest_path.parent / wav_path).resolve()
            if not wav_path.exists():
                failed += 1
                print(f"[FAIL] {case_id}: wav not found ({wav_path})")
                continue

            audio_bytes, sr, sw, ch = read_wav_bytes(wav_path)
            payload = {
                "prompt": "",
                "audio_bytes": audio_bytes,
                "sr": sr,
                "sw": sw,
                "ch": ch,
                "source_id": str(case.get("source_id", "regression")),
                "turn_id": str(case.get("turn_id") or uuid.uuid4()),
                "utterance_id": str(case.get("utterance_id") or uuid.uuid4()),
                "speaker_id": case.get("speaker_id"),
            }

            text = ""
            latency_ms = None
            provider = None
            async for out_d in stt_op(payload):
                if not bool(out_d.get("is_final", True)):
                    continue
                provider = out_d.get("provider")
                latency_ms = out_d.get("stt_latency_ms")
                chunk_text = str(out_d.get("text") or out_d.get("transcription") or "").strip()
                if chunk_text:
                    if text:
                        text += " "
                    text += chunk_text

            ok, reason = evaluate_case(case, text)
            if ok:
                print(f"[PASS] {case_id}: '{text}' (provider={provider}, latency_ms={latency_ms})")
            else:
                failed += 1
                print(f"[FAIL] {case_id}: {reason} | got='{text}'")
    finally:
        await stt_op.close()

    total = len(cases)
    print("")
    print(f"[SUMMARY] total={total} failed={failed} passed={total - failed}")
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Run STT regression cases against active STT adapter.")
    parser.add_argument("--config", default="config", help="Config name/file from configs/")
    parser.add_argument("--manifest", required=True, help="Path to regression manifest JSON")
    parser.add_argument(
        "--reuse-running-sidecar",
        type=int,
        default=1,
        help="1 = do not autostart STT sidecar, reuse existing ws endpoint",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}")
        return 2

    return asyncio.run(run_regression(args.config, manifest_path, bool(args.reuse_running_sidecar)))


if __name__ == "__main__":
    raise SystemExit(main())
