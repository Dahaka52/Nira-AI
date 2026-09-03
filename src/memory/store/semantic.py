"""
Семантическая память — факты о мире и людях.

Хранит: знания о пользователях (имена, интересы), о мире,
о самой Нире. Обновляется через диалоги.

TODO:
- Векторное хранилище фактов (sqlite-vec)
- Метод add_fact(subject, predicate, value) — добавить факт
- Метод get_facts(subject) — получить факты об объекте
- Метод search(query_embedding) — поиск похожих фактов
"""


class SemanticStore:
    """Заготовка. Не активна."""

    def __init__(self):
        pass

    async def add_fact(self, subject: str, predicate: str, value: str, **kwargs):
        """Сохранить факт."""
        raise NotImplementedError

    async def get_facts(self, subject: str):
        """Получить все факты об объекте."""
        raise NotImplementedError
