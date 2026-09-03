import os
from openai import AsyncOpenAI
import logging

from .base import T2TOperation
from utils.prompter.message import ChatMessage
from utils.prompter import Prompter
from utils.config import Config

class UniversalApiT2T(T2TOperation):
    def __init__(self):
        super().__init__("universal_api")
        self.client = None
        
        # Настройки провайдера
        self.base_url = "https://api.openai.com/v1/"
        self.api_key = ""
        self.model = "gpt-4o"
        
        # Параметры генерации
        self.max_length: int = 500
        self.temperature: float = 0.7
        self.top_p: float = 0.95
        self.stream: bool = True
        
    async def start(self):
        await super().start()
        # Ищем ключ API (из конфига или переменных окружения)
        api_key = self.api_key
        if not api_key:
            api_key = (
                os.environ.get("OPENAI_API_KEY") or
                os.environ.get("GEMINI_API_KEY") or
                os.environ.get("GROQ_API_KEY") or
                os.environ.get("OPENROUTER_API_KEY") or
                os.environ.get("CEREBRAS_API_KEY") or
                ""
            )
            
        if not api_key and "api.openai.com" in self.base_url:
            logging.warning("API key is not set. OpenAI API requests will likely fail.")
        
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
        
    async def close(self):
        await super().close()
        if self.client:
            await self.client.close()
            self.client = None
        
    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        # Провайдер-специфичные настройки
        if "base_url" in config_d: 
            self.base_url = str(config_d['base_url'])
        elif config_d.get("id") == "google_ai":
            # Фоллбэк для старого id
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            
        if "model" in config_d: self.model = str(config_d['model'])
        if "api_key" in config_d: self.api_key = str(config_d['api_key'])
        
        # Глобальные настройки генерации (фоллбэк на Config().t2t)
        global_t2t = getattr(Config(), "t2t", {}) or {}
        
        self.max_length = int(config_d.get("max_length", global_t2t.get("max_length", self.max_length)))
        self.temperature = float(config_d.get("temperature", global_t2t.get("temperature", self.temperature)))
        self.top_p = float(config_d.get("top_p", global_t2t.get("top_p", self.top_p)))
        self.stream = bool(config_d.get("stream", global_t2t.get("stream", self.stream)))
        
        assert self.model is not None and len(self.model) > 0
        assert self.max_length > 0
        assert self.temperature >= 0
        assert 0 <= self.top_p <= 1
        
    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {
            "base_url": self.base_url,
            "model": self.model,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream
        }

    async def _generate(self, instruction_prompt: str = None, messages: list = None, **kwargs):
        history = [{ "role": "system", "content": instruction_prompt }]
        for msg in messages:
            next_hist = None
            if isinstance(msg, ChatMessage) and msg.user == Prompter().character_name:
                next_hist = { "role": "assistant", "content": msg.message }
            else:
                next_hist = { "role": "user", "content": msg.to_line() }
            history.append(next_hist)

        if self.stream:
            stream = await self.client.chat.completions.create(
                messages=history,
                model=self.model,
                stream=True,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            async for chunk in stream:
                content_chunk = chunk.choices[0].delta.content or ""
                if content_chunk:
                    yield {"content": content_chunk}
        else:
            response = await self.client.chat.completions.create(
                messages=history,
                model=self.model,
                stream=False,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            result = response.choices[0].message.content or ""
            yield {"content": result}
