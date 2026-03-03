import httpx
import json
import logging
import traceback
import sys
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Any
from utils.args import args

from utils.processes import ProcessManager, ProcessType
from .base import T2TOperation
from utils.prompter.message import ChatMessage
from utils.prompter import Prompter

class LlamaCppT2T(T2TOperation):
    LLAMACPP_LINK_ID = "llamacpp_t2t"
    
    def __init__(self):
        super().__init__("llamacpp")
        self.uri = None
        
        # Sampler settings
        self.max_length: int = 200
        self.temperature: float = 0.8
        self.top_p: float = 0.95
        self.top_k: int = 40
        self.stream: bool = True
        
    async def start(self) -> None:
        '''Setup and link to managed process'''
        await super().start()
        await ProcessManager().link(self.LLAMACPP_LINK_ID, ProcessType.LLAMACPP)
        # URI не кэшируем — берём актуальный порт при каждом запросе
        # (llama-server может перезапуститься на другом порту)
    
    async def close(self) -> None:
        '''Unlink process'''
        await super().close()
        await ProcessManager().unlink(self.LLAMACPP_LINK_ID, ProcessType.LLAMACPP)
    
    async def configure(self, config_d: Dict[str, Any]):
        '''Configure sampler overrides'''
        if "max_length" in config_d: self.max_length = int(config_d["max_length"])
        if "temperature" in config_d: self.temperature = float(config_d["temperature"])
        if "top_p" in config_d: self.top_p = float(config_d["top_p"])
        if "top_k" in config_d: self.top_k = int(config_d["top_k"])
        if "stream" in config_d: self.stream = bool(config_d["stream"])
        
        assert self.max_length > 0
        assert self.temperature >= 0
        assert 0 <= self.top_p <= 1
        assert self.top_k >= 0

    async def get_configuration(self):
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": self.stream
        }

    async def _generate(self, instruction_prompt: str = None, messages: list = None, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        # Format messages for OpenAI-compatible Chat Completions API
        raw_history = []
        for msg in messages:
            if isinstance(msg, ChatMessage) and msg.user == Prompter().character_name:
                role = "assistant"
                content = msg.message
            else:
                role = "user"
                # Use pure message for ChatMessage, to_line for others (MCP, Request)
                content = msg.message if hasattr(msg, 'message') else msg.to_line()
            raw_history.append({"role": role, "content": content})

        # Merge consecutive roles to satisfy strict alternation requirements (e.g. Mistral-Nemo)
        history = [{ "role": "system", "content": instruction_prompt }]
        if raw_history:
            last_role = None
            for entry in raw_history:
                if last_role == entry["role"]:
                    history[-1]["content"] += "\n" + entry["content"]
                else:
                    history.append(entry)
                    last_role = entry["role"]

        payload = {
            "messages": history,
            "max_tokens": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": self.stream
        }


        # Всегда берём актуальный порт — защита от рестарта llama-server
        uri = "http://127.0.0.1:{}".format(ProcessManager().get_process(ProcessType.LLAMACPP).port)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                if self.stream:
                    async with client.stream("POST", f"{uri}/v1/chat/completions", json=payload) as response:

                        if response.status_code != 200:
                            error_text = await response.aread()
                            raise Exception(f"Llama.cpp error: {response.status_code} - {error_text.decode()}")
                        
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        yield {"content": content}
                                except json.JSONDecodeError:
                                    logging.warning(f"Failed to decode SSE line: {line}")
                else:
                    response = await client.post(f"{uri}/v1/chat/completions", json=payload)
                    if response.status_code == 200:
                        result = response.json()['choices'][0]['message']['content']
                        yield {"content": result}
                    else:
                        raise Exception(f"Llama.cpp error: {response.status_code} - {response.text}")
        except Exception as e:
            raise
