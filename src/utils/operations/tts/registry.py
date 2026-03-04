import importlib
from typing import Any, Dict, Type

from ..error import UnknownOpID
from .base import TTSOperation


_BUILTIN_TTS = {
    "qwen3": "utils.operations.tts.qwen3:Qwen3TTS",
}


def register_tts_operation(op_id: str, entrypoint: str) -> None:
    key = str(op_id or "").strip()
    if not key:
        raise ValueError("TTS op_id must be non-empty.")
    _BUILTIN_TTS[key] = str(entrypoint)


def _load_entrypoint(entrypoint: str) -> Type[TTSOperation]:
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise ValueError(f"Invalid TTS entrypoint '{entrypoint}'. Expected 'module.path:ClassName'.")
    module_name, class_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, TTSOperation):
        raise TypeError(f"Entrypoint '{entrypoint}' is not a TTSOperation subclass.")
    return cls


def load_tts_operation(op_id: str, op_details: Dict[str, Any] | None = None) -> TTSOperation:
    details = op_details or {}
    entrypoint = details.get("entrypoint")
    if entrypoint:
        cls = _load_entrypoint(str(entrypoint))
        op = cls()
        if op_id:
            op.op_id = str(op_id)
        return op

    builtin = _BUILTIN_TTS.get(str(op_id))
    if not builtin:
        raise UnknownOpID("TTS", op_id)
    cls = _load_entrypoint(builtin)
    op = cls()
    if op_id:
        op.op_id = str(op_id)
    return op

