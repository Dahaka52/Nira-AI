import re

from .base import FilterTextOperation

class ResponseCleaningFilter(FilterTextOperation):
    def __init__(self):
        super().__init__("filter_clean")
        self.pattern = None
        
    async def start(self):
        await super().start()
        # Компилируем регулярки для фильтрации
        self.patterns = [
            re.compile(r"\[[^\[\]]+\]:\s*"),  # Старый паттерн (убирает только лейблы спикеров типа [Dahaka]:)
            re.compile(r'\(.*?\)'),           # Круглые скобки (например, (Смех от души...))
            re.compile(r'\*.*?\*'),           # Звездочки (действия, например *улыбается*)
            re.compile(r'[\U00010000-\U0010ffff]') # Базовые эмодзи
        ]
        
    async def close(self):
        await super().close()
    
    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        return
    
    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {}

    async def _generate(self, content: str = None, **kwargs):
        '''Generate a output stream'''
        if content:
            # Применяем все фильтры
            for pattern in self.patterns:
                content = pattern.sub('', content)
            
            # Убираем двойные пробелы, которые могли остаться после удаления скобок
            content = re.sub(r'\s+', ' ', content).strip()
            
        yield {
            "content": content
        }

