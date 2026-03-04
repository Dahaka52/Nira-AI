from __future__ import annotations

from typing import Any, Dict


async def apply_pre_stt_hooks(
    request_data: Dict[str, Any],
    audio_bytes: bytes,
    sr: int,
    sw: int,
    ch: int,
) -> Dict[str, Any]:
    """
    Extension point for pre-STT audio analysis.
    Current implementation is intentionally lightweight and non-breaking:
    it forwards optional speaker_id if supplied externally.
    """
    result: Dict[str, Any] = {}
    speaker_id = request_data.get("speaker_id")
    if isinstance(speaker_id, str) and speaker_id.strip():
        result["speaker_id"] = speaker_id.strip()
    return result

