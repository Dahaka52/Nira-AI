from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from .manager import ProcessManager, ProcessType


class TTSProcessRunner:
    """
    Helper around ProcessManager for TTS sidecar lifecycle.
    """

    def __init__(
        self,
        link_id: str,
        process_type: ProcessType,
        process_config: Dict[str, Any] | None = None,
        startup_retries: int = 3,
        startup_backoff_s: float = 0.5,
    ):
        self.link_id = link_id
        self.process_type = process_type
        self.process_config = dict(process_config or {})
        self._linked = False
        self.startup_retries = max(1, int(startup_retries))
        self.startup_backoff_s = max(0.05, float(startup_backoff_s))

    def set_process_config(self, process_config: Dict[str, Any] | None) -> None:
        self.process_config = dict(process_config or {})

    async def ensure_running(self) -> None:
        await ProcessManager().link(self.link_id, self.process_type, process_config=self.process_config)
        self._linked = True

    async def close(self) -> None:
        if not self._linked:
            return
        try:
            await ProcessManager().unlink(self.link_id, self.process_type)
        finally:
            self._linked = False

    async def restart(self) -> None:
        pm = ProcessManager()
        try:
            pm.signal_reload(self.process_type)
            await pm.reload()
        except Exception:
            # Process may be unloaded already; ensure_running below will recreate it.
            pass
        await self.ensure_running()

    async def health(self) -> Dict[str, Any]:
        try:
            proc = ProcessManager().get_process(self.process_type)
            running = bool(proc.process is not None and proc.process.poll() is None)
            return {
                "running": running,
                "pid": proc.process.pid if running else None,
                "port": getattr(proc, "port", None),
            }
        except Exception:
            return {"running": False, "pid": None, "port": None}

    async def ensure_healthy(self) -> None:
        last_exc = None
        for attempt in range(1, self.startup_retries + 1):
            health = await self.health()
            if health["running"]:
                return
            try:
                await self.ensure_running()
                await asyncio.sleep(min(0.3, self.startup_backoff_s / 2.0))
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "TTSProcessRunner ensure_healthy attempt %s/%s failed: %s",
                    attempt,
                    self.startup_retries,
                    exc,
                )
            health = await self.health()
            if health["running"]:
                return
            if attempt < self.startup_retries:
                await asyncio.sleep(self.startup_backoff_s * attempt)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"TTS process {self.process_type.value} failed health checks after retries.")

