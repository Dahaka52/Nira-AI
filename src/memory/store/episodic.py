"""
Эпизодическая память — история диалогов.

Хранит: кто говорил, что, когда — с временными метками.
Записи постепенно консолидируются через consolidation/summarizer.py.

TODO:
- Реализовать хранение в SQLite (через sqlite-vec)
- Метод save(turn) — сохранение реплики
- Метод query(n=50) — получить последние N записей
- Метод query_by_speaker(speaker_id) — история по участнику
"""


class EpisodicStore:
    """Заготовка. Не активна."""

    def __init__(self):
        pass

    async def save(self, user: str, content: str, timestamp: float, **kwargs):
        """Сохранить реплику в эпизодическую память."""
        raise NotImplementedError

    async def query_recent(self, n: int = 50):
        """Получить N последних записей."""
        raise NotImplementedError
