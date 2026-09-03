import logging
import subprocess
from subprocess import DEVNULL
import socket
import os
import shlex
import asyncio
import httpx
from utils.config import Config
from utils.args import args
from utils.helpers.singleton import Singleton
from ..base import BaseProcess

class LlamaCPPProcess(BaseProcess, metaclass=Singleton):
    def __init__(self):
        super().__init__("llamacpp")
        self.reload_signal = True
        self._log_file = None

    def _close_log_file(self):
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    async def unload(self):
        try:
            await super().unload()
        finally:
            self._close_log_file()
        
    async def reload(self):
        # Close any existing servers
        if self.process is not None:
            await self.unload()
        
        await super().reload()
        
        # Find open port
        config = Config()
        sock = socket.socket()
        sock.bind(('', 0))
        self.port = sock.getsockname()[1]
        sock.close()
        
        # Prepare command as a list for shell=False
        cmd_list = [
            config.llamacpp_filepath,
            "-m", config.llamacpp_model_filepath,
            "--port", str(self.port),
            "-c", str(config.llamacpp_ctx),
            "-ngl", str(config.llamacpp_ngl),
            "-ctk", config.llamacpp_cache_type_k,
            "-ctv", config.llamacpp_cache_type_v
        ]
        
        if config.llamacpp_extra_args:
            cmd_list.extend(shlex.split(config.llamacpp_extra_args))
            
        logging.debug(f"Running Llama.cpp server using command: {' '.join(cmd_list)}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(config.llamacpp_gpu_id)
        
        # Используем абсолютный путь к лог-файлу (важно при запуске через Start-Job)
        log_path = os.path.join(args.log_dir, "llama_server.log")
        os.makedirs(args.log_dir, exist_ok=True)
        self._close_log_file()
        self._log_file = open(log_path, "w", encoding="utf-8")
        try:
            # Try to start the process
            self.process = subprocess.Popen(
                cmd_list, 
                stdin=DEVNULL,
                stdout=self._log_file, 
                stderr=self._log_file, 
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
            )
            logging.info(f"Opened Llama.cpp server (PID: {self.process.pid}) on port {self.port} (GPU: {config.llamacpp_gpu_id})")
        except Exception as e:
            logging.error(f"Failed to start llama-server: {e}")
            self._close_log_file()
            raise e
        
        # Ждём готовности llama-server (опрашиваем /health до 120 секунд)
        health_url = f"http://127.0.0.1:{self.port}/health"
        logging.info(f"Waiting for llama-server to be ready at {health_url} ...")
        deadline = 120  # секунд максимум
        for attempt in range(deadline * 2):  # проверяем каждые 0.5 сек
            await asyncio.sleep(0.5)
            # Проверяем не упал ли процесс
            if self.process.poll() is not None:
                logging.error(f"llama-server process died (exit code {self.process.returncode}). Check {log_path}")
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}. Check logs: {log_path}")
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        logging.info(f"llama-server is ready! (attempt {attempt+1}, port {self.port})")
                        return
            except Exception:
                pass  # Сервер ещё не готов, продолжаем ждать
        
        raise RuntimeError(f"llama-server did not become ready in {deadline} seconds. Check logs: {log_path}")
