"""
Поиск воспоминаний по давности и важности.

Простой фильтр: берёт N последних записей,
взвешивает по временнóму затуханию и метке важности.

TODO:
- get_recent(n: int) → List[Dict] — последние N записей
- get_important(min_importance: float) → List[Dict] — по важности
"""
from typing import List, Dict, Any


class RecencyRetrieval:
    """Заготовка. Не активна."""

    async def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Получить N последних воспоминаний."""
        raise NotImplementedError

    async def get_important(self, min_importance: float = 0.7) -> List[Dict[str, Any]]:
        """Получить воспоминания выше порога важности."""
        raise NotImplementedError
