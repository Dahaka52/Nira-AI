"""
Рабочая память — буфер текущего разговора.

Обогащённая версия текущего диалога: включает
извлечённые воспоминания, активный контекст,
незакрытые темы текущей сессии.

TODO:
- clear() — сброс рабочей памяти (новая сессия)
- add_retrieved(memories) — добавить воспоминания из LTM
- get_context() — получить текущий контекст для промпта
"""
from typing import List, Dict, Any


class WorkingMemory:
    """Заготовка. Не активна."""

    def __init__(self):
        self._retrieved: List[Dict[str, Any]] = []
        self._session_notes: List[str] = []

    def clear(self):
        """Сброс рабочей памяти в начале новой сессии."""
        self._retrieved.clear()
        self._session_notes.clear()

    def add_retrieved(self, memories: List[Dict[str, Any]]):
        """Добавить воспоминания, извлечённые из LTM."""
        self._retrieved.extend(memories)

    def get_context(self) -> str:
        """Сформировать текстовый контекст для вставки в промпт."""
        raise NotImplementedError
