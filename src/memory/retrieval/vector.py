"""
Векторный поиск воспоминаний.

Использует embedding/openai.py для векторизации запросов
и сравнивает с сохранёнными векторами воспоминаний.

TODO:
- search(query: str, top_k: int) → List[Dict]
  Принимает текстовый запрос → векторизует → ищет ближайшие
- threshold — настраиваемый порог сходства
"""
from typing import List, Dict, Any


class VectorRetrieval:
    """Заготовка. Не активна."""

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Найти top_k воспоминаний, семантически близких к запросу."""
        raise NotImplementedError
