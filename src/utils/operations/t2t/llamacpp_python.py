import logging
from typing import AsyncGenerator, Dict, Any
import asyncio

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

import os
from .base import T2TOperation
from utils.prompter.message import ChatMessage
from utils.prompter import Prompter

class LlamaCppPythonT2T(T2TOperation):
    def __init__(self):
        super().__init__("llamacpp_python")
        self.model = None
        
        self.model_path: str = None
        self.n_ctx: int = 4096
        self.n_gpu_layers: int = 99
        self.max_tokens: int = 200
        self.temperature: float = 0.8
        self.top_p: float = 0.95
        self.stream: bool = True
        self.cache_type_k: str = "f16"
        self.cache_type_v: str = "f16"
        self.gpu_id: int = 0
        
    async def start(self) -> None:
        await super().start()
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Please follow INSTALL_LLAMA_PYTHON.md")
        
        logging.info(f"Loading Llama model from {self.model_path}...")
        # Llama class is synchronous and heavy, running in thread pool
        loop = asyncio.get_event_loop()
        # Mapping for llama-cpp-python cache types
        type_map = {
            "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8
        }
        tk = type_map.get(self.cache_type_k, 1) # Default f16
        tv = type_map.get(self.cache_type_v, 1)

        # Hard bind to specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.model = await loop.run_in_executor(None, lambda: Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            type_k=tk,
            type_v=tv,
            verbose=False
        ))
        logging.info("Llama model loaded successfully.")
    
    async def close(self) -> None:
        await super().close()
        self.model = None
    
    async def configure(self, config_d: Dict[str, Any]):
        if "model_path" in config_d: self.model_path = config_d["model_path"]
        if "n_ctx" in config_d: self.n_ctx = int(config_d["n_ctx"])
        if "n_gpu_layers" in config_d: self.n_gpu_layers = int(config_d["n_gpu_layers"])
        if "max_tokens" in config_d: self.max_tokens = int(config_d["max_tokens"])
        if "temperature" in config_d: self.temperature = float(config_d["temperature"])
        if "top_p" in config_d: self.top_p = float(config_d["top_p"])
        if "stream" in config_d: self.stream = bool(config_d["stream"])
        if "cache_type_k" in config_d: self.cache_type_k = str(config_d["cache_type_k"])
        if "cache_type_v" in config_d: self.cache_type_v = str(config_d["cache_type_v"])
        if "gpu_id" in config_d: self.gpu_id = int(config_d["gpu_id"])
        
        assert self.model_path is not None

    async def get_configuration(self):
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream
        }

    async def _generate(self, instruction_prompt: str = None, messages: list = None, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        history = [{ "role": "system", "content": instruction_prompt }]
        for msg in messages:
            if isinstance(msg, ChatMessage) and msg.user == Prompter().character_name:
                role = "assistant"
                content = msg.message
            else:
                role = "user"
                content = msg.to_line()
            history.append({ "role": role, "content": content })

        loop = asyncio.get_event_loop()
        
        if self.stream:
            # handle streaming
            def run_stream():
                return self.model.create_chat_completion(
                    messages=history,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=True
                )
            
            stream_iterator = await loop.run_in_executor(None, run_stream)
            
            for chunk in stream_iterator:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        yield {"content": delta['content']}
                await asyncio.sleep(0) # Yield control
        else:
            def run_sync():
                return self.model.create_chat_completion(
                    messages=history,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False
                )
            
            response = await loop.run_in_executor(None, run_sync)
            yield {"content": response['choices'][0]['message']['content']}
