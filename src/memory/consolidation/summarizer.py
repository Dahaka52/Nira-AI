"""
Консолидация памяти через LLM.

Периодически сжимает старые эпизодические воспоминания:
сохраняет суть, выбрасывает избыточные детали.
Запускается фоново по объёму или таймеру.

TODO:
- summarize(episodes: List[Dict]) → str
  Отдаёт список реплик → LLM → краткое резюме сессии
- consolidate_old(older_than_days: int)
  Запускает консолидацию записей старше N дней
"""
from typing import List, Dict, Any


class MemorySummarizer:
    """Заготовка. Не активна."""

    async def summarize(self, episodes: List[Dict[str, Any]]) -> str:
        """Создать краткое резюме набора эпизодов через LLM."""
        raise NotImplementedError

    async def consolidate_old(self, older_than_days: int = 7):
        """Консолидировать воспоминания старше N дней."""
        raise NotImplementedError
