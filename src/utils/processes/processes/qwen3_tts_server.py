import asyncio
import logging
import os
import socket
import subprocess
import glob
from subprocess import DEVNULL

import httpx

from utils.args import args
from utils.config import Config
from utils.processes.base import BaseProcess


class Qwen3TTSProcess(BaseProcess):
    def __init__(self):
        super().__init__("qwen3_tts")
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

    def _find_open_port(self) -> int:
        sock = socket.socket()
        sock.bind(("", 0))
        port = int(sock.getsockname()[1])
        sock.close()
        return port

    def _resolve_sox_bin_dir(self, runtime_cfg: dict) -> str:
        explicit = str(runtime_cfg.get("sox_bin_dir", "") or "").strip()
        if explicit and os.path.isfile(os.path.join(explicit, "sox.exe")):
            return explicit

        env_hint = str(os.getenv("QWEN3_SOX_BIN_DIR", "") or "").strip()
        if env_hint and os.path.isfile(os.path.join(env_hint, "sox.exe")):
            return env_hint

        repo_default = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../../apps/tts-qwen3-server/third_party/sox/sox-14.4.2-win32/sox-14.4.2",
            )
        )
        if os.path.isfile(os.path.join(repo_default, "sox.exe")):
            return repo_default

        wildcard = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../../apps/tts-qwen3-server/third_party/sox/**/sox.exe",
            )
        )
        matches = glob.glob(wildcard, recursive=True)
        if matches:
            return os.path.dirname(matches[0])

        return ""

    async def reload(self):
        if self.process is not None:
            await self.unload()
        await super().reload()

        _ = Config()  # Keep parity with other process implementations.
        runtime_cfg = dict(self.runtime_config or {})

        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../apps/tts-qwen3-server/start_server.py")
        )
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Qwen3 TTS sidecar script missing: {script_path}")

        import sys

        default_sidecar_python = ""
        for candidate in (
            "../../../../apps/tts-qwen3-server/.venv312/Scripts/python.exe",
            "../../../../apps/tts-qwen3-server/.venv/Scripts/python.exe",
        ):
            candidate_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), candidate))
            if os.path.exists(candidate_abs):
                default_sidecar_python = candidate_abs
                break
        python_executable = str(runtime_cfg.get("python_executable", "")).strip()
        if not python_executable:
            python_executable = default_sidecar_python if default_sidecar_python else sys.executable
        if not os.path.exists(python_executable):
            raise FileNotFoundError(f"Qwen3 TTS python_executable not found: {python_executable}")
        host = str(runtime_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        health_host = str(runtime_cfg.get("health_host", "127.0.0.1")).strip() or "127.0.0.1"
        provider = str(runtime_cfg.get("provider", "cuda")).strip().lower() or "cuda"
        try:
            gpu_id = int(runtime_cfg.get("gpu_id", 1))
        except Exception:
            gpu_id = 1
        try:
            port = int(runtime_cfg.get("port", 6116))
        except Exception:
            port = 6116
        if port <= 0:
            port = self._find_open_port()
        self.port = port

        default_device = "cpu" if provider.startswith("cpu") else "cuda:0"
        device = str(runtime_cfg.get("device", "")).strip() or default_device
        runtime_cfg.setdefault("runtime_flavor", "dffdeeq")
        runtime_cfg.setdefault("voice_mode", "voice_clone")

        cmd = [
            python_executable,
            "-u",
            script_path,
            "--host",
            host,
            "--port",
            str(port),
            "--provider",
            provider,
            "--gpu_id",
            str(gpu_id),
            "--device",
            device,
        ]

        # Optional runtime tuning forwarded to sidecar.
        for key, arg_name in (
            ("model_id", "--model_id"),
            ("language", "--language"),
            ("dtype", "--dtype"),
            ("attn_implementation", "--attn_implementation"),
            ("use_compile", "--use_compile"),
            ("sample_rate", "--sample_rate"),
            ("sample_width", "--sample_width"),
            ("channels", "--channels"),
            ("voice_mode", "--voice_mode"),
            ("runtime_flavor", "--runtime_flavor"),
            ("qwen_tts_repo_path", "--qwen_tts_repo_path"),
            ("ref_audio_path", "--ref_audio_path"),
            ("ref_text", "--ref_text"),
            ("x_vector_only_mode", "--x_vector_only_mode"),
            ("emit_every_frames", "--emit_every_frames"),
            ("decode_window_frames", "--decode_window_frames"),
            ("overlap_samples", "--overlap_samples"),
            ("max_frames", "--max_frames"),
            ("use_optimized_decode", "--use_optimized_decode"),
            ("do_sample", "--do_sample"),
            ("top_p", "--top_p"),
            ("top_k", "--top_k"),
            ("temperature", "--temperature"),
            ("dynamic_max_frames", "--dynamic_max_frames"),
            ("dynamic_chars_per_second", "--dynamic_chars_per_second"),
            ("dynamic_frame_budget_mul", "--dynamic_frame_budget_mul"),
            ("dynamic_min_frames", "--dynamic_min_frames"),
            ("dynamic_max_frames_cap", "--dynamic_max_frames_cap"),
            ("max_concurrent", "--max_concurrent"),
            ("preload_on_start", "--preload_on_start"),
            ("warmup_on_start", "--warmup_on_start"),
            ("warmup_text", "--warmup_text"),
            ("warmup_chunks", "--warmup_chunks"),
            ("first_chunk_timeout_s", "--first_chunk_timeout_s"),
            ("stream_idle_timeout_s", "--stream_idle_timeout_s"),
            ("compile_mode", "--compile_mode"),
            ("compile_use_cuda_graphs", "--compile_use_cuda_graphs"),
            ("compile_codebook_predictor", "--compile_codebook_predictor"),
            ("compile_talker", "--compile_talker"),
            ("compile_cudagraph_skip_dynamic_graphs", "--compile_cudagraph_skip_dynamic_graphs"),
        ):
            if key in runtime_cfg and runtime_cfg[key] is not None:
                value = runtime_cfg[key]
                if isinstance(value, bool):
                    value = int(value)
                cmd.extend([arg_name, str(value)])

        log_path = os.path.join(args.log_dir, "qwen3_tts_server.log")
        os.makedirs(args.log_dir, exist_ok=True)
        self._close_log_file()
        self._log_file = open(log_path, "w", encoding="utf-8")

        env = os.environ.copy()
        if provider.startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        sox_bin_dir = self._resolve_sox_bin_dir(runtime_cfg)
        if sox_bin_dir:
            env["PATH"] = f"{sox_bin_dir}{os.pathsep}{env.get('PATH', '')}"
            logging.info("Qwen3 TTS SoX bin detected: %s", sox_bin_dir)
        else:
            logging.warning("Qwen3 TTS SoX bin not found. Install sox.exe or set process.sox_bin_dir.")

        logging.info("Starting Qwen3 TTS Server with command: %s", " ".join(cmd))
        logging.info(
            "Qwen3 TTS GPU pin: provider=%s gpu_id=%s device=%s CUDA_VISIBLE_DEVICES=%s",
            provider,
            gpu_id,
            device,
            env.get("CUDA_VISIBLE_DEVICES", ""),
        )
        self.process = subprocess.Popen(
            cmd,
            stdin=DEVNULL,
            stdout=self._log_file,
            stderr=self._log_file,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0,
        )

        health_url = f"http://{health_host}:{self.port}/health"
        try:
            deadline_s = int(runtime_cfg.get("startup_health_timeout_s", 90))
        except Exception:
            deadline_s = 90
        deadline_s = max(30, deadline_s)
        for _ in range(deadline_s * 2):
            await asyncio.sleep(0.5)
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Qwen3 TTS sidecar exited with code {self.process.returncode}. Check logs: {log_path}"
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        logging.info("Qwen3 TTS sidecar is ready at %s", health_url)
                        return
            except Exception:
                pass

        raise RuntimeError(f"Qwen3 TTS sidecar health timeout after {deadline_s}s. Check logs: {log_path}")
