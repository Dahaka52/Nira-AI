import logging
import asyncio
import uuid
import base64
import datetime
import time
import os
import re
import json
from collections import deque
from typing import Dict, Coroutine, List, Any, Tuple
from enum import Enum
from utils.args import args
from utils.console import (
    print_stt, print_stt_done, print_stage,
    print_llm_start, print_llm_done,
    print_tts_phrase, print_tts_done,
    print_interrupt,
)

from utils.helpers.singleton import Singleton
from utils.helpers.iterable import chunk_buffer
from utils.helpers.observer import ObserverServer

from utils.config import Config, UnknownField, UnknownFile
from utils.prompter import Prompter
from utils.prompter.message import (
    RawMessage,
    RequestMessage,
    ChatMessage,
    MCPMessage,
    CustomMessage
)
from utils.processes import ProcessManager
from utils.operations import (
    OperationManager,
    OpRoles,
    Operation,
    UnknownOpType,
    UnknownOpRole,
    UnknownOpID,
    DuplicateFilter,
    OperationUnloaded,
    StartActiveError,
    CloseInactiveError,
    UsedInactiveError
)
from utils.operations.stt.hooks import apply_pre_stt_hooks
from utils.mcp import MCPManager

# ── Санитайзер текста перед синтезом речи (TTS) ──
_RE_TTS_BRACKETS = re.compile(r'\[.*?\]')
_RE_TTS_ASTERISKS = re.compile(r'\*.*?\*')
_RE_TTS_EMOJIS = re.compile(
    r'[\U00010000-\U0010ffff'
    r'\u2600-\u27bf'
    r'\u2300-\u23ff'
    r'\u2b50-\u2b55'
    r'\u200d\ufe0e\ufe0f]'
)
_RE_TTS_EMOTICONS = re.compile(r'(?i)(?:[:;=8][-o*]?[)DdpP/\\}\]|[({\[]|XD|xd|\^_\^|>_<|<3)')

def clean_phrase_for_tts(text: str) -> str:
    """Удаляет из текста для TTS эмодзи, смайлики, псевдо-теги в квадратных скобках, звездочки и REQUEST-мусор."""
    if not text:
        return ""
    # Отрезаем скобочные конструкции с REQUEST: (например (REQUEST: ВОЛЕЙБОЛ) или [REQUEST: ...])
    text = re.sub(r'\s*\([^)]*REQUEST[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\[[^\]]*REQUEST[^\]]*\]', '', text, flags=re.IGNORECASE)
    # Отрезаем любые мета-инструкции REQUEST
    if "REQUEST:" in text:
        text = text.split("REQUEST:")[0]
    cleaned = _RE_TTS_BRACKETS.sub('', text)
    cleaned = _RE_TTS_ASTERISKS.sub('', cleaned)
    cleaned = _RE_TTS_EMOJIS.sub('', cleaned)
    cleaned = _RE_TTS_EMOTICONS.sub('', cleaned)
    # Подчищаем висячие открывающие скобки в конце
    cleaned = re.sub(r'[\s(\[{]+$', '', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()
    # Если не осталось ни одной буквы или цифры — фраза не содержит речи
    if not re.search(r'[a-zA-Zа-яА-Я0-9]', cleaned):
        return ""
    return cleaned

class NonexistantJobException(Exception):
    pass

class UnknownJobType(Exception):
    pass

class JobType(Enum):
    RESPONSE = 'response'
    CONTEXT_CLEAR = 'context_clear'
    CONTEXT_CONFIGURE = "context_configure"
    CONTEXT_REQUEST_ADD = 'context_request_add'
    CONTEXT_CONVERSATION_ADD_TEXT = 'context_conversation_add_text'
    CONTEXT_CONVERSATION_ADD_AUDIO = 'context_conversation_add_audio'
    CONTEXT_CUSTOM_REGISTER = 'context_custom_register'
    CONTEXT_CUSTOM_REMOVE = 'context_custom_remove'
    CONTEXT_CUSTOM_ADD = 'context_custom_add'
    OPERATION_LOAD = 'operation_load'
    OPERATION_CONFIG_RELOAD = "operation_reload_from_config"
    OPERATION_UNLOAD = 'operation_unload'
    OPERATION_CONFIGURE = 'operation_configure'
    OPERATION_USE = 'operation_use'
    CONFIG_LOAD = 'config_load'
    CONFIG_UPDATE = 'config_update'
    CONFIG_SAVE = 'config_save'
    
class JAIson(metaclass=Singleton):
    def __init__(self): # attribute stubs
        self.job_loop: asyncio.Task = None
        self.job_queue: asyncio.Queue = None
        self.job_map: Dict[str, Tuple[JobType, Coroutine]] = None
        self.job_current_id: str = None
        self.job_current: asyncio.Task = None
        self.job_skips: dict = None
        
        # Any asyncio.Tasks in this list will be cancelled before the next job runs
        self.tasks_to_clean: List = list()
        
        self.event_server: ObserverServer = None
        
        self.prompter: Prompter = None
        self.process_manager: ProcessManager = None
        self.op_manager: OperationManager = None
        self.mcp_manager: MCPManager = None
        self._pending_voice_response_task: asyncio.Task = None
        self._pending_voice_response_seq: int = 0
        self._pending_voice_turn: Dict[str, Any] = None
        self._last_speech_start_ts: float = 0.0
        self._assistant_live_job_id: str = None
        self._assistant_live_reply: str = ""
        self._assistant_last_full_reply: str = ""
        self._assistant_last_partial_reply: str = ""
        self._response_job_speakers: Dict[str, str] = dict()

        # Immediate STT path backpressure/runtime
        self._immediate_audio_lock: asyncio.Lock | None = None
        self._immediate_audio_active: int = 0
        self._immediate_audio_pending = deque()
        self._immediate_audio_tasks = set()

        # STT observability
        self._stt_window = deque(maxlen=200)
        self._stt_events_path = os.path.join(args.log_dir, "stt_events.jsonl")
        self._stt_last_status = {"key": None, "ts": 0.0}

        # Pipeline telemetry and Discord bridge observability
        self._last_telemetry: Dict[str, Any] = None
        self._telemetry_history: deque = deque(maxlen=20)
        self._discord_bridge_status: Dict[str, Any] = {
            "online": False,
            "connected_to_voice": False,
            "gateway_ping_ms": None,
            "voice_ping_ms": None,
            "channel_name": None,
            "guild_id": None,
            "channel_id": None,
            "is_playing": False,
            "updated_at": 0.0,
            "members": [],
        }
        self._discord_channel_members: Dict[str, str] = {}
        self._prev_discord_channel_id: Optional[int] = None
        self._prev_discord_connected: bool = False
        self._audio_output_mode: str = "local"
    
    async def start(self):
        logging.info("Starting JAIson application layer.")
        print_stage("CORE", "Инициализация JAIson (очереди, события)…", "boot")
        self.job_queue = asyncio.Queue()
        self.job_map = dict()
        self.job_skips = dict()
        self.job_loop = asyncio.create_task(self._process_job_loop())
        self.manager_loop = asyncio.create_task(self._process_manager_loop())
        
        self.event_server = ObserverServer()
        
        self.prompter = Prompter()
        await self.prompter.configure(Config().prompter)
        print_stage("PROMPT", f"Промпт загружен: {getattr(Config().prompter, 'character_name', '?') if hasattr(Config().prompter, 'character_name') else Config().prompter.get('character_name', '?') if isinstance(Config().prompter, dict) else '?'}", "ok")
        
        self.process_manager = ProcessManager()
        self.op_manager = OperationManager()
        self.mcp_manager = MCPManager()
        await self.mcp_manager.start()
        self.prompter.add_mcp_usage_prompt(self.mcp_manager.get_tooling_prompt(), self.mcp_manager.get_response_prompt())
        print_stage("OPS", "Загрузка операций из конфига…", "boot")
        await self.op_manager.load_operations_from_config()
        print_stage("OPS", "Операции загружены (STT/T2T/TTS)", "ok")
        self._set_speaker_filter_enabled(self.get_audio_output_mode() == "local")
        await self.process_manager.reload()
        self._immediate_audio_lock = asyncio.Lock()
        
        # Start microphone if enabled
        mic_cfg = Config().microphone or {}
        if isinstance(mic_cfg, dict) and mic_cfg.get("enabled", False):
            from utils.processes.manager import ProcessType
            try:
                await self.process_manager.link("core_hw_mic", ProcessType.HW_MIC)
                print_stage("MIC", "Микрофон подключён", "ok")
            except Exception as e:
                logging.error(f"Could not start HW_MIC process: {e}")
                print_stage("MIC", f"Ошибка микрофона: {e}", "error")

        discord_cfg = Config().discord or {}
        if isinstance(discord_cfg, dict) and discord_cfg.get("enabled", False):
            try:
                await self.process_manager.link(
                    "core_discord",
                    ProcessType.DISCORD,
                    process_config=discord_cfg,
                )
                print_stage("DISCORD", "Discord Bridge запущен", "ok")
            except Exception as e:
                logging.error(f"Could not start Discord bridge: {e}")
                print_stage("DISCORD", f"Ошибка Discord Bridge: {e}", "error")

        logging.info("JAIson application layer has started.")
        print_stage("READY", "✨ Нира готова к работе!", "ok")
        
    async def stop(self):
        logging.info("Shutting down JAIson application layer")
        for task in list(self._immediate_audio_tasks):
            task.cancel("shutdown")
        self._immediate_audio_tasks.clear()
        self._immediate_audio_pending.clear()
        self._immediate_audio_active = 0
        if getattr(self, "op_manager", None):
            await self.op_manager.close_operation_all()
        if getattr(self, "mcp_manager", None):
            await self.mcp_manager.close()
        if getattr(self, "process_manager", None):
            await self.process_manager.unload()
        logging.info("JAIson application layer has been shut down")

    def _get_microphone_config(self) -> Dict[str, Any]:
        try:
            cfg = Config().microphone or {}
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
        return {}

    def _get_audio_backpressure_config(self) -> Dict[str, Any]:
        cfg = self._get_microphone_config()
        max_active = int(cfg.get("stt_immediate_max_active", 2) or 2)
        max_pending = int(cfg.get("stt_immediate_max_pending", 8) or 8)
        policy = str(cfg.get("stt_backpressure_policy", "drop_oldest") or "drop_oldest").strip().lower()
        if policy not in {"drop_oldest", "drop_latest", "merge_latest"}:
            policy = "drop_oldest"
        return {
            "max_active": max(1, max_active),
            "max_pending": max(1, max_pending),
            "policy": policy,
        }

    def _safe_source_id(self, value: Any) -> str:
        source = str(value or "mic").strip()
        return source if source else "mic"

    async def _append_stt_event_log(self, event_d: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self._stt_events_path), exist_ok=True)
            with open(self._stt_events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_d, ensure_ascii=False))
                f.write("\n")
        except Exception:
            logging.debug("Failed to append STT event log", exc_info=True)

    async def _emit_stt_status(self, state: str, **payload) -> None:
        if not self.event_server:
            return

        cfg = self._get_microphone_config()
        cooldown_ms = int(cfg.get("stt_status_cooldown_ms", 750) or 750)
        now = time.time()
        key = f"{state}:{payload.get('reason', '')}:{payload.get('source_id', '')}"
        if self._stt_last_status["key"] == key and ((now - self._stt_last_status["ts"]) * 1000.0) < cooldown_ms:
            return

        self._stt_last_status = {"key": key, "ts": now}
        status_payload = {
            "event": "stt_status",
            "state": state,
            "timestamp": now,
        }
        status_payload.update(payload)
        await self.event_server.broadcast_event("stt_status", status_payload)

    async def _record_stt_metrics(
        self,
        source_id: str,
        turn_id: str,
        utterance_id: str,
        provider: str,
        latency_ms: int | None,
        text: str,
        detected_stop_cmd: bool,
        expected_stop_cmd: bool | None = None,
    ) -> None:
        self._stt_window.append({
            "latency_ms": latency_ms,
            "empty": not bool((text or "").strip()),
            "stop_detected": bool(detected_stop_cmd),
            "stop_expected": expected_stop_cmd,
        })

        # Keep all turns traceable for offline analysis.
        await self._append_stt_event_log({
            "timestamp": time.time(),
            "source_id": source_id,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "provider": provider,
            "latency_ms": latency_ms,
            "text": text,
            "empty": not bool((text or "").strip()),
            "stop_detected": bool(detected_stop_cmd),
            "stop_expected": expected_stop_cmd,
        })

        if len(self._stt_window) < 20:
            return
        if len(self._stt_window) % 20 != 0:
            return

        latencies = [x["latency_ms"] for x in self._stt_window if isinstance(x["latency_ms"], int)]
        empty_rate = sum(1 for x in self._stt_window if x["empty"]) / max(1, len(self._stt_window))
        avg_latency = int(sum(latencies) / len(latencies)) if latencies else -1

        expected = [x for x in self._stt_window if x["stop_expected"] is True]
        if expected:
            recall = sum(1 for x in expected if x["stop_detected"]) / max(1, len(expected))
        else:
            # Proxy metric until labeled stop-command dataset is wired.
            recall = sum(1 for x in self._stt_window if x["stop_detected"]) / max(1, len(self._stt_window))

        logging.info(
            "STT metrics(window=%s): avg_latency_ms=%s empty_rate=%.3f stop_cmd_recall=%.3f",
            len(self._stt_window),
            avg_latency,
            empty_rate,
            recall,
        )

    async def _run_immediate_audio_task(self, request_data: Dict[str, Any]) -> None:
        try:
            await self.process_audio_immediate(request_data)
        except Exception:
            logging.error("Unhandled error in immediate STT worker", exc_info=True)
            await self._emit_stt_status(
                "unavailable",
                reason="worker_exception",
                source_id=self._safe_source_id(request_data.get("source_id")),
            )
        finally:
            next_payload = None
            if self._immediate_audio_lock is None:
                self._immediate_audio_lock = asyncio.Lock()
            async with self._immediate_audio_lock:
                self._immediate_audio_active = max(0, self._immediate_audio_active - 1)
                if self._immediate_audio_pending:
                    next_payload = self._immediate_audio_pending.popleft()
                    self._immediate_audio_active += 1
            if next_payload is not None:
                self._start_immediate_audio_task(next_payload)

    def _start_immediate_audio_task(self, request_data: Dict[str, Any]) -> None:
        task = asyncio.create_task(self._run_immediate_audio_task(request_data))
        self._immediate_audio_tasks.add(task)
        task.add_done_callback(lambda t: self._immediate_audio_tasks.discard(t))

    async def submit_audio_immediate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request_data, dict) or not request_data.get("audio_bytes"):
            return {
                "accepted": False,
                "queued": False,
                "dropped": True,
                "drop_reason": "invalid_audio_payload",
                "active": self._immediate_audio_active,
                "pending": len(self._immediate_audio_pending),
                "policy": self._get_audio_backpressure_config()["policy"],
            }

        config = self._get_audio_backpressure_config()
        source_id = self._safe_source_id(request_data.get("source_id"))
        
        # Изоляция аудиопотоков: отбрасываем неактивный источник до очереди STT
        current_mode = self.get_audio_output_mode()
        if current_mode == "discord" and source_id == "mic":
            return {
                "accepted": False,
                "queued": False,
                "dropped": True,
                "drop_reason": "mic_ignored_in_discord_mode",
                "active": self._immediate_audio_active,
                "pending": len(self._immediate_audio_pending),
                "policy": config["policy"],
            }
        if current_mode == "local" and source_id == "discord":
            return {
                "accepted": False,
                "queued": False,
                "dropped": True,
                "drop_reason": "discord_ignored_in_local_mode",
                "active": self._immediate_audio_active,
                "pending": len(self._immediate_audio_pending),
                "policy": config["policy"],
            }

        to_emit = None

        if self._immediate_audio_lock is None:
            self._immediate_audio_lock = asyncio.Lock()
        async with self._immediate_audio_lock:
            if self._immediate_audio_active < config["max_active"]:
                self._immediate_audio_active += 1
                self._start_immediate_audio_task(request_data)
                return {
                    "accepted": True,
                    "queued": False,
                    "active": self._immediate_audio_active,
                    "pending": len(self._immediate_audio_pending),
                    "policy": config["policy"],
                }

            pending = self._immediate_audio_pending
            if len(pending) >= config["max_pending"]:
                if config["policy"] == "drop_latest":
                    to_emit = ("backpressure_drop", "drop_latest")
                    result = {
                        "accepted": False,
                        "queued": False,
                        "dropped": True,
                        "drop_reason": "backpressure_drop_latest",
                        "active": self._immediate_audio_active,
                        "pending": len(self._immediate_audio_pending),
                        "policy": config["policy"],
                    }
                    # Return early while still under lock for accurate counters.
                    if to_emit:
                        # Emit after releasing lock.
                        pass
                elif config["policy"] == "merge_latest":
                    merged = False
                    for idx in range(len(pending) - 1, -1, -1):
                        candidate = pending[idx]
                        if self._safe_source_id(candidate.get("source_id")) == source_id:
                            pending[idx] = request_data
                            merged = True
                            break
                    if not merged:
                        pending.popleft()
                        pending.append(request_data)
                    to_emit = ("backpressure_merge", "merge_latest")
                    result = {
                        "accepted": True,
                        "queued": True,
                        "merged": True,
                        "active": self._immediate_audio_active,
                        "pending": len(self._immediate_audio_pending),
                        "policy": config["policy"],
                    }
                else:
                    # drop_oldest
                    pending.popleft()
                    pending.append(request_data)
                    to_emit = ("backpressure_drop", "drop_oldest")
                    result = {
                        "accepted": True,
                        "queued": True,
                        "dropped_oldest": True,
                        "active": self._immediate_audio_active,
                        "pending": len(self._immediate_audio_pending),
                        "policy": config["policy"],
                    }
            else:
                pending.append(request_data)
                result = {
                    "accepted": True,
                    "queued": True,
                    "active": self._immediate_audio_active,
                    "pending": len(self._immediate_audio_pending),
                    "policy": config["policy"],
                }

        if to_emit:
            state, reason = to_emit
            await self._emit_stt_status(
                state,
                reason=reason,
                source_id=source_id,
                active=self._immediate_audio_active,
                pending=len(self._immediate_audio_pending),
            )

        return result

    def get_stt_runtime_stats(self) -> Dict[str, Any]:
        return {
            "immediate_active": self._immediate_audio_active,
            "immediate_pending": len(self._immediate_audio_pending),
            "immediate_workers": len(self._immediate_audio_tasks),
            "stt_window_size": len(self._stt_window),
        }
    
    ## Job Queueing #########################
    
    # Add async task to Queue to be ran in the order it was requested
    async def create_job(self, job_type: Enum, **kwargs):
        new_job_id = str(uuid.uuid4())
        
        job_type_enum = JobType(job_type)
        
        coro = None
        if job_type_enum == JobType.RESPONSE:
            # Очередь не должна накапливать устаревшие RESPONSE задачи.
            # Если в очереди уже ждут неотвеченные RESPONSE, отменяем их в пользу самого свежего ответа.
            for queued_id, (q_type, _) in list(self.job_map.items()):
                if q_type == JobType.RESPONSE and queued_id != self.job_current_id and queued_id not in self.job_skips:
                    self.job_skips[queued_id] = "superceded_by_newer_response"
                    logging.info(f"[PREEMPTION] Отменяем устаревший RESPONSE job {queued_id} в пользу свежего ответа {new_job_id}")

            coro = self.response_pipeline(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_REQUEST_ADD: coro = self.append_request_context(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_CONVERSATION_ADD_TEXT: coro = self.append_conversation_context_text(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_CONVERSATION_ADD_AUDIO: coro = self.append_conversation_context_audio(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_CLEAR: coro = self.clear_context(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_CONFIGURE: coro = self.configure_context(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_CUSTOM_REGISTER: coro = self.register_custom_context(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_CUSTOM_REMOVE: coro = self.remove_custom_context(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONTEXT_CUSTOM_ADD: coro = self.add_custom_context(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.OPERATION_LOAD: coro = self.load_operations(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.OPERATION_CONFIG_RELOAD: coro = self.load_operations_from_config(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.OPERATION_UNLOAD: coro = self.unload_operations(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.OPERATION_CONFIGURE: coro = self.configure_operations(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.OPERATION_USE: coro = self.use_operation(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONFIG_LOAD: coro = self.load_config(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONFIG_UPDATE: coro = self.update_config(new_job_id, job_type_enum, **kwargs)
        elif job_type_enum == JobType.CONFIG_SAVE: coro = self.save_config(new_job_id, job_type_enum, **kwargs)
        self.job_map[new_job_id] = (job_type_enum, coro)
        
        await self.job_queue.put(new_job_id)
        
        logging.info("Queued new {} job {}".format(job_type_enum.value, new_job_id))
        return new_job_id
    
    async def cancel_job(self, job_id: str, reason: str = None):
        if job_id not in self.job_map: raise NonexistantJobException(f"Job {job_id} does not exist or already finished")
        
        cancel_message = f"Setting job {job_id} to cancel"
        if reason: cancel_message += f" because {reason}"
        logging.info(cancel_message)

        if job_id == self.job_current_id:
            # If job is already running
            self._clear_current_job(reason=cancel_message)
        else: 
            # If job is still in Queue
            # Simply flag to skip. Unzipping queue can potentially process a job out of order 
            self.job_skips[job_id] = cancel_message
            
    def _clear_current_job(self, reason: str = None):
        job_id = self.job_current_id
        job_type, _ = self.job_map.get(job_id, (None, None))
        
        self.job_map.pop(job_id, None)
        self.job_skips.pop(job_id, None)
        self._response_job_speakers.pop(job_id, None)
        self.job_current_id = None
        
        for task in self.tasks_to_clean:
            task.cancel(reason)
        self.tasks_to_clean.clear()
        
        if self.job_current is not None:
            if reason:
                logging.info(f"Job {job_id} ({job_type.value if job_type else 'unknown'}) is being cancelled due to: {reason}")
            self.job_current.cancel(reason)
            self.job_current = None

    def _stop_local_speakers(self):
        """Мгновенный сброс воспроизведения звука на локальных колонках."""
        try:
            filters = self.op_manager.get_operation(OpRoles.FILTER_AUDIO)
            if not isinstance(filters, list):
                filters = [filters]
            for f in filters:
                if hasattr(f, "stop_audio"):
                    f.stop_audio()
        except Exception:
            pass

    def _can_interrupt_assistant(self, speaker_id: str = None, reason: str = "barge_in") -> bool:
        """
        Проверяет, разрешено ли прервать речь Ниры.
        Оставляет возможность проигнорировать прерывание при определенном условии
        (например, в будущем при эмоциональном аффекте, упрямстве или режиме монолога).
        """
        if getattr(self, "suppress_interruptions", False):
            logging.info("Interruption suppressed by assistant condition (suppress_interruptions=True).")
            return False
        return True

    def _interrupt_allowed_for_speaker(self, speaker_id: str = None) -> bool:
        """Совместимость: проверяет права спикера через _can_interrupt_assistant."""
        return self._can_interrupt_assistant(speaker_id=speaker_id)

    def _interrupt_jobs(self, reason: str = "user_interruption"):
        """Экстренное прерывание: очистка устаревших ответов и текущей задачи, с сохранением контекста"""
        logging.info(f"Smart Barge-in: Interrupting and clearing queue due to: {reason}")
        print_interrupt(reason=reason)
        self._cancel_pending_voice_response()
        
        # 1. Извлекаем задачи из очереди: отменяем ТОЛЬКО устаревшие RESPONSE,
        # а все контекстные задачи (добавление реплик пользователей, системные события) сохраняем!
        preserved_jobs = []
        while True:
            try:
                queued_job_id = self.job_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            job_type_coro = self.job_map.get(queued_job_id)
            if job_type_coro:
                job_type, coro = job_type_coro
                if job_type == JobType.RESPONSE:
                    self.job_map.pop(queued_job_id, None)
                    self.job_skips.pop(queued_job_id, None)
                    self._response_job_speakers.pop(queued_job_id, None)
                    try:
                        coro.close()
                    except Exception:
                        pass
                else:
                    preserved_jobs.append(queued_job_id)

            try:
                self.job_queue.task_done()
            except Exception:
                pass

        # Возвращаем все контекстные задачи обратно в очередь
        for preserved_id in preserved_jobs:
            self.job_queue.put_nowait(preserved_id)
        
        logging.debug(f"Interrupting current job for Barge-in")
        # 2. Прерываем текущую задачу через стандартный метод
        self._clear_current_job(reason=reason)

        # 3. Мгновенно глушим локальные колонки
        self._stop_local_speakers()

    def _cancel_pending_voice_response(self):
        task = self._pending_voice_response_task
        if task is not None and not task.done():
            task.cancel("new_voice_activity")
        self._pending_voice_response_task = None

    def _is_continue_intent(self, text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        continue_markers = (
            "продолжай",
            "продолжи",
            "договори",
            "не перебивай",
            "рассказывай дальше",
            "дальше продолжай",
            "продолжение",
            "продолжай историю",
            "продолжи историю",
            "ну давай дальше",
            "продолжай мысль",
        )
        return any(marker in low for marker in continue_markers)

    async def _commit_pending_voice_turn(self):
        turn = self._pending_voice_turn
        self._pending_voice_turn = None
        if not turn:
            return

        content = str(turn.get("content", "")).strip()
        if not content:
            return

        user = turn.get("user", "user")
        timestamp = turn.get("timestamp", time.time())
        source_id = self._safe_source_id(turn.get("source_id"))
        turn_id = str(turn.get("turn_id") or uuid.uuid4())
        utterance_ids = list(turn.get("utterance_ids") or [])
        utterance_id = str(utterance_ids[0]) if utterance_ids else str(uuid.uuid4())
        speaker_id = turn.get("speaker_id")
        stt_provider = turn.get("stt_provider")
        stt_confidence = turn.get("stt_confidence")
        stt_latency_ms = turn.get("stt_latency_ms")
        continue_intent = bool(turn.get("continue_intent", False))
        
        # Проверяем, не состоит ли весь собранный буфер из одного "ага"
        import re
        words = re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ-]+", content.lower().strip())
        fillers = {"угу", "ага", "понятно", "ясно", "да", "так", "хорошо", "ок", "слышу", "мгм", "ладно", "понял", "ого", "ммм", "эмм", "хмм", "интересно"}
        non_filler_words = [w for w in words if w not in fillers]
        is_backchannel = len(words) <= 2 and len(non_filler_words) == 0

        should_respond = bool(turn.get("should_respond", True))
        if is_backchannel:
            should_respond = False
            logging.info("Entire buffer was just a backchannel. Disabling response.")

        continue_from_text = None
        if continue_intent:
            continue_from_text = (self._assistant_last_partial_reply or self._assistant_last_full_reply or "").strip()
            if continue_from_text:
                logging.info("Continue-intent detected: next response will continue previous thought.")

        # [Barge-in Rework] LLM Filter
        try:
            filter_mode = str(getattr(Config(), "stt_filter_mode", "algorithm")).strip().lower()
            if filter_mode == "llm" and content and not is_backchannel:
                logging.info(f"Running LLM STT filter on: '{content}'")
                prompt = (
                    f"Исправь опечатки STT, убери мусорные слова (кашель, шум, субтитры) и верни ТОЛЬКО чистый текст. "
                    f"Если текст - бессмысленный шум, верни слово 'DROP'. Текст: {content}"
                )
                from .messages import RawMessage
                clean_text = ""
                async for chunk in self.op_manager.use_operation(
                    OpRoles.T2T, 
                    {
                        "messages": [RawMessage(prompt)], 
                        "instruction_prompt": "Ты - системный корректор распознавания речи. Отвечай только текстом без пояснений.",
                        "temperature": 0.1,
                        "max_length": 200
                    }
                ):
                    clean_text += chunk.get("content", "")
                
                clean_text = clean_text.strip()
                if "DROP" in clean_text.upper():
                    logging.info("LLM Filter dropped the phrase as garbage.")
                    return  # Полностью отбрасываем
                elif clean_text:
                    logging.info(f"LLM Filter cleaned phrase to: '{clean_text}'")
                    content = clean_text
        except Exception as e:
            logging.error(f"LLM STT filter failed (falling back to raw text): {e}")

        await self.create_job(
            JobType.CONTEXT_CONVERSATION_ADD_TEXT,
            user=user,
            content=content,
            timestamp=timestamp,
            source_id=source_id,
            turn_id=turn_id,
            utterance_id=utterance_id,
            utterance_ids=utterance_ids,
            speaker_id=speaker_id,
            stt_provider=stt_provider,
            stt_confidence=stt_confidence,
            stt_latency_ms=stt_latency_ms,
        )
        if not should_respond:
            logging.info("Voice turn committed as context-only (no RESPONSE job).")
            return

        speech_start_ts = turn.get("speech_start_ts") or turn.get("timestamp") or time.time()
        speech_end_ts = turn.get("speech_end_ts") or time.time()
        response_job_id = await self.create_job(
            JobType.RESPONSE,
            input_timestamp=speech_end_ts,   # ПОСЛЕДНИЙ ПАКЕТ ГОЛОСА ПОЛЬЗОВАТЕЛЯ (КОНЕЦ РЕЧИ)
            speech_start_ts=speech_start_ts, # НАЧАЛО ЗАПИСИ ГОЛОСА В СТТ
            speech_end_ts=speech_end_ts,     # КОНЕЦ РЕЧИ ПОЛЬЗОВАТЕЛЯ
            input_mode="voice",
            continue_from_text=continue_from_text,
            source_id=source_id,
            turn_id=turn_id,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            stt_provider=stt_provider,
            stt_confidence=stt_confidence,
            stt_latency_ms=stt_latency_ms,
            stt_finish_ts=turn.get("stt_finish_ts") or time.time(),
        )
        if response_job_id and speaker_id:
            self._response_job_speakers[response_job_id] = str(speaker_id)

    async def _buffer_voice_turn(
        self,
        user: str,
        timestamp: float,
        content: str,
        continue_intent: bool,
        should_respond: bool,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        speaker_id: str = None,
        stt_provider: str = None,
        stt_confidence: float = None,
        stt_latency_ms: int = None,
        speech_start_ts: float = None,
        speech_end_ts: float = None,
        is_direct_question: bool = False,
    ):
        merge_window_ms = 2500
        try:
            cfg = Config()
            mic_cfg = cfg.microphone or {}
            merge_window_ms = int(getattr(cfg, "turn_merge_window_ms", None) or mic_cfg.get("voice_merge_window_ms", 2500))
        except Exception:
            pass

        pending = self._pending_voice_turn
        if pending is None:
            self._pending_voice_turn = {
                "user": user,
                "timestamp": timestamp,
                "speech_start_ts": speech_start_ts or timestamp,
                "speech_end_ts": speech_end_ts or timestamp,
                "last_timestamp": timestamp,
                "content": content,
                "source_id": self._safe_source_id(source_id),
                "turn_id": str(turn_id or uuid.uuid4()),
                "utterance_ids": [str(utterance_id)] if utterance_id else [],
                "speaker_id": speaker_id,
                "stt_provider": stt_provider,
                "stt_confidence": stt_confidence,
                "stt_latency_ms": stt_latency_ms,
                "stt_finish_ts": time.time(),
                "continue_intent": continue_intent,
                "should_respond": bool(should_respond),
                "is_direct_question": is_direct_question,
            }
        else:
            same_user = pending.get("user") == user or (speaker_id and pending.get("speaker_id") == speaker_id)
            
            # 1. Проверяем аудио-паузу (интервал между окончанием предыдущего фрагмента и началом нового)
            prev_end = float(pending.get("speech_end_ts") or pending.get("last_timestamp") or timestamp)
            cur_start = float(speech_start_ts or timestamp)
            audio_gap_ms = max(0.0, (cur_start - prev_end) * 1000.0)
            
            # 2. Проверяем системное время (сколько прошло с момента завершения STT предыдущего фрагмента)
            last_stt_finish = float(pending.get("stt_finish_ts") or time.time())
            wall_gap_ms = max(0.0, (time.time() - last_stt_finish) * 1000.0)
            
            # 3. Активен ли ещё таймер ответа Ниры (дебаунс тишины ещё не истёк, Нира молчит)
            is_timer_active = (self._pending_voice_response_task is not None and not self._pending_voice_response_task.done())
            within_merge_window = (audio_gap_ms <= merge_window_ms or wall_gap_ms <= merge_window_ms)

            if same_user and (within_merge_window or is_timer_active):
                prev = str(pending.get("content", "")).strip()
                cur = str(content).strip()
                pending["content"] = (prev + " " + cur).strip() if prev else cur
                pending["last_timestamp"] = timestamp
                pending["continue_intent"] = bool(pending.get("continue_intent", False) or continue_intent)
                pending["should_respond"] = bool(pending.get("should_respond", False) or should_respond)
                if is_direct_question:
                    pending["is_direct_question"] = True
                pending["stt_finish_ts"] = time.time()
                if speech_end_ts:
                    pending["speech_end_ts"] = speech_end_ts
                if utterance_id:
                    pending.setdefault("utterance_ids", []).append(str(utterance_id))
                if speaker_id and not pending.get("speaker_id"):
                    pending["speaker_id"] = speaker_id
                if stt_provider:
                    pending["stt_provider"] = stt_provider
                if stt_confidence is not None:
                    pending["stt_confidence"] = stt_confidence
                if stt_latency_ms is not None:
                    pending["stt_latency_ms"] = stt_latency_ms
            else:
                await self._commit_pending_voice_turn()
                self._pending_voice_turn = {
                    "user": user,
                    "timestamp": timestamp,
                    "speech_start_ts": speech_start_ts or timestamp,
                    "speech_end_ts": speech_end_ts or timestamp,
                    "last_timestamp": timestamp,
                    "content": content,
                    "source_id": self._safe_source_id(source_id),
                    "turn_id": str(turn_id or uuid.uuid4()),
                    "utterance_ids": [str(utterance_id)] if utterance_id else [],
                    "speaker_id": speaker_id,
                    "stt_provider": stt_provider,
                    "stt_confidence": stt_confidence,
                    "stt_latency_ms": stt_latency_ms,
                    "stt_finish_ts": time.time(),
                    "continue_intent": continue_intent,
                    "should_respond": bool(should_respond),
                    "is_direct_question": is_direct_question,
                }

        self._schedule_voice_response()

    def _schedule_voice_response(self):
        self._pending_voice_response_seq += 1
        seq = self._pending_voice_response_seq
        self._cancel_pending_voice_response()

        try:
            cfg = Config()
            mic_cfg = cfg.microphone or {}
            base_debounce_ms = int(getattr(cfg, "stt_buffer_timeout_ms", 1400))
            is_direct = bool(self._pending_voice_turn and self._pending_voice_turn.get("is_direct_question"))
            # Если задан прямой вопрос к Нире — отвечаем быстрее (750мс).
            # Если идет связная речь или рассказ — ждем 1400мс тишины, чтобы не перебивать.
            debounce_ms = min(base_debounce_ms, 750) if is_direct else base_debounce_ms
            min_quiet_ms_after_speech_start = int(mic_cfg.get("response_min_quiet_ms_after_speech_start", 350))
        except Exception:
            debounce_ms = 1400
            min_quiet_ms_after_speech_start = 350

        async def _delayed_response():
            try:
                await asyncio.sleep(max(0.0, debounce_ms / 1000.0))
                if seq != self._pending_voice_response_seq:
                    return

                # Coalesce near-adjacent speech chunks: wait until a short quiet window
                while True:
                    if seq != self._pending_voice_response_seq:
                        return
                    if self._last_speech_start_ts <= 0:
                        break

                    quiet_ms = (time.time() - self._last_speech_start_ts) * 1000.0
                    if quiet_ms >= max(0, min_quiet_ms_after_speech_start):
                        break

                    wait_s = min(0.2, max(0.01, (min_quiet_ms_after_speech_start - quiet_ms) / 1000.0))
                    await asyncio.sleep(wait_s)

                await self._commit_pending_voice_turn()
            except asyncio.CancelledError:
                return
            finally:
                if self._pending_voice_response_task is not None and self._pending_voice_response_task.done():
                    self._pending_voice_response_task = None

        self._pending_voice_response_task = asyncio.create_task(_delayed_response())

    def _ensure_pending_voice_response_scheduled(self) -> None:
        """Гарантирует, что если в _pending_voice_turn есть накопленный текст, таймер ответа активен."""
        if self._pending_voice_turn is not None and self._pending_voice_turn.get("content"):
            if self._pending_voice_response_task is None or self._pending_voice_response_task.done():
                self._schedule_voice_response()
        
    async def _process_job_loop(self):
        while True:
            try:
                self.job_current_id = await self.job_queue.get()
                current_job_id = self.job_current_id
                job_type, coro = self.job_map[current_job_id]
                
                if current_job_id in self.job_skips:
                    # Skip cancelled jobs
                    reason = self.job_skips[current_job_id]
                    await self._handle_broadcast_cancelled(current_job_id, job_type, reason)
                    # Cancelled queued jobs were never awaited: close coroutine explicitly
                    try:
                        coro.close()
                    except Exception:
                        pass
                    self._clear_current_job(reason=reason)
                else:
                    # Run and wait for completion
                    self.job_current = asyncio.create_task(coro)
                    try:
                        await self.job_current
                    except asyncio.CancelledError as err:
                        reason = str(err) if str(err) else "cancelled"
                        if reason == "cancelled":
                            if self._last_speech_start_ts > 0 and (time.time() - self._last_speech_start_ts) <= 2.0:
                                reason = "user_voice_start"
                        if self._assistant_live_job_id == current_job_id:
                            partial = self._assistant_live_reply.strip()
                            if partial:
                                clean_partial = partial.replace("\n", " ").rstrip(" .")
                                self._assistant_last_partial_reply = clean_partial[-2000:]
                                # В историю для модели пишем естественное многоточие без скобок, чтобы она не копировала псевдо-теги
                                self.prompter.add_chat(self.prompter.character_name, clean_partial + "...")
                                try:
                                    await self.event_server.broadcast_event(
                                        "context_conversation_add_text",
                                        {
                                            "finished": True,
                                            "job_id": f"interrupted_{int(time.time() * 1000)}",
                                            "result": {
                                                "user": self.prompter.character_name,
                                                "content": clean_partial + "...",
                                                "timestamp": time.time(),
                                                "source_id": "interrupted",
                                            }
                                        }
                                    )
                                except Exception:
                                    pass
                            self._assistant_live_job_id = None
                            self._assistant_live_reply = ""
                        logging.info(f"Job {current_job_id} ({job_type.value}) was cancelled.")
                        await self._handle_broadcast_cancelled(current_job_id, job_type, reason)
                    except Exception as err:
                        if self._assistant_live_job_id == current_job_id:
                            self._assistant_live_job_id = None
                            self._assistant_live_reply = ""
                        logging.error(f"Job {current_job_id} failed with error: {err}", exc_info=True)
                        await self._handle_broadcast_error(current_job_id, job_type, err)
                    
                    # Cleanup
                    self._clear_current_job()
            except Exception as err:
                logging.error("Encountered error in main job processing loop", exc_info=True)
    async def _process_manager_loop(self):
        """Фоновый цикл для обновления статуса процессов (сокращает задержку основного цикла)"""
        while True:
            try:
                await self.process_manager.reload()
                await self.process_manager.unload()
            except Exception as e:
                logging.error(f"Error in process manager loop: {e}")
            await asyncio.sleep(5)  # Проверяем сигналы раз в 5 секунд

    ## Regular Request Handlers ###################
    
    def get_loaded_operations(self):
        if not self.op_manager:
            return {}
        op_d = self.op_manager.get_operation_all()
        for key in op_d:
            if isinstance(op_d[key], Operation):
                op_d[key] = op_d[key].op_id
            elif isinstance(op_d[key], list):
                op_d[key] = list(map(lambda x: x.op_id, op_d[key]))
            else:
                op_d[key] = "unknown"
                
        return op_d
                
    def get_current_config(self):
        return Config().get_config_dict()

    def get_active_providers(self) -> Dict[str, Any]:
        """Возвращает текущие активные провайдеры STT, T2T, TTS с моделями и типами (local/cloud)."""
        cfg = Config().get_config_dict()
        operations_cfg = cfg.get("operations", [])
        
        loaded = self.get_loaded_operations()
        stt_id = loaded.get("stt") or cfg.get("stt_active_id")
        t2t_id = loaded.get("t2t") or cfg.get("t2t_active_id")
        tts_id = loaded.get("tts") or cfg.get("tts_active_id")

        if not tts_id:
            for op in operations_cfg:
                if isinstance(op, dict) and op.get("role") == "tts":
                    tts_id = op.get("id")
                    break
        
        stt_info = {"id": stt_id, "model": None, "type": "local"}
        t2t_info = {"id": t2t_id, "model": None, "type": "cloud"}
        tts_info = {"id": tts_id, "model": None, "type": "cloud"}
        
        for op in operations_cfg:
            if not isinstance(op, dict):
                continue
            role = op.get("role")
            oid = op.get("id")
            if role == "stt" and oid == stt_id:
                stt_info["model"] = op.get("model") or op.get("model_path") or op.get("model_dir") or op.get("model_size")
                ep = str(op.get("entrypoint", "")).lower()
                stt_info["type"] = "local" if ("gigaam" in ep or oid in ("gigaam", "gigaam_ru")) else "cloud"
            elif role == "t2t" and oid == t2t_id:
                t2t_info["model"] = op.get("model") or "local-llm"
                t2t_info["type"] = "local" if oid == "llamacpp" else "cloud"
            elif role == "tts" and oid == tts_id:
                tts_info["model"] = op.get("model") or "fish-speech"
                ep = str(op.get("entrypoint", "")).lower()
                tts_info["type"] = "local" if ("local" in ep or oid in ("local_tts", "piper")) else "cloud"
                
        return {
            "stt": stt_info,
            "t2t": t2t_info,
            "tts": tts_info
        }

    def _format_discord_member_label(self, uid: str, raw_name: str) -> str:
        """Форматирует имя участника Discord с учетом его семейной роли и статуса Создателя."""
        resolved_name = self.resolve_speaker_name(speaker_id=str(uid), raw_user=raw_name, source_id="discord")
        cfg = Config()
        known = dict(getattr(cfg, "known_users", {}))
        dynamic_path = os.path.join(cfg.CONFIG_DIR, "known_users.json")
        role = ""
        user_info = {}
        if os.path.isfile(dynamic_path):
            try:
                with open(dynamic_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and str(uid) in d and isinstance(d[str(uid)], dict):
                        user_info = d[str(uid)]
            except Exception:
                pass
        if not user_info and str(uid) in known and isinstance(known[str(uid)], dict):
            user_info = known[str(uid)]

        role = user_info.get("role", "")
        if not role:
            aliases = user_info.get("aliases", [])
            if "Папа" in aliases or resolved_name == "Вова":
                role = "Папа"
            elif "Мама" in aliases or resolved_name == "Настя":
                role = "Мама"

        if role == "Папа":
            return f"{resolved_name} (Папа, Создатель)"
        elif role == "Мама":
            return f"{resolved_name} (Мама)"
        elif role:
            return f"{resolved_name} ({role})"
        return resolved_name

    def emit_environment_notice(self, notice: str):
        """Вкидывает системное событие окружения в контекст диалога Ниры и на дашборд."""
        if not notice:
            return
        logging.info(f"[DISCORD ENV] {notice}")
        if self.prompter:
            self.prompter.add_chat("Окружение", notice)
        if self.event_server:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.event_server.broadcast_event(
                    "context_conversation_add_text",
                    {
                        "finished": False,
                        "job_id": f"env_{int(time.time() * 1000)}",
                        "result": {
                            "user": "Окружение",
                            "content": notice,
                            "source_id": "discord",
                            "timestamp": time.time(),
                            "type": "environment",
                        }
                    }
                ))
            except RuntimeError:
                pass

    def get_discord_presence_prompt(self) -> str:
        """Динамический блок присутствия участников Discord канала для промпта сцены."""
        bridge = self.get_discord_bridge_status()
        if not bridge.get("connected_to_voice"):
            return ""

        channel_name = bridge.get("channel_name") or "голосовой канал"
        members = getattr(self, "_discord_channel_members", {})
        if not members:
            return (
                f"### Discord Environment ###\n"
                f"You are connected to Discord voice channel \"{channel_name}\".\n"
                f"Currently, no other members are present.\n"
            )

        lines = [f"- {self._format_discord_member_label(uid, name)}" for uid, name in members.items()]
        members_list_str = "\n".join(lines)
        return (
            f"### Discord Environment ###\n"
            f"You are in Discord voice channel \"{channel_name}\".\n"
            f"Present members:\n"
            f"{members_list_str}\n"
            f"All present members hear your voice. Acknowledge them naturally by their names/roles (Папа, Мама, etc.).\n"
        )

    def set_discord_bridge_status(self, status: Dict[str, Any]):
        status["updated_at"] = time.time()

        now_connected = bool(status.get("connected_to_voice", False))
        now_channel_id = status.get("channel_id")
        channel_name = status.get("channel_name") or "голосовой канал"
        members_raw = status.get("members", [])
        new_members: Dict[str, str] = {
            str(m["id"]): str(m.get("name", f"user_{m['id']}"))
            for m in members_raw
            if isinstance(m, dict) and "id" in m
        }

        prev_connected = getattr(self, "_prev_discord_connected", False)
        prev_channel_id = getattr(self, "_prev_discord_channel_id", None)
        prev_members = getattr(self, "_discord_channel_members", {})

        # Случай 1: Подключение к каналу или переход в другой канал
        if now_connected and (not prev_connected or (prev_channel_id is not None and prev_channel_id != now_channel_id)):
            self._prev_discord_connected = True
            self._prev_discord_channel_id = now_channel_id
            self._discord_channel_members = dict(new_members)

            if new_members:
                labels = [self._format_discord_member_label(uid, name) for uid, name in new_members.items()]
                notice = f'Подключилась к голосовому каналу "{channel_name}". В канале находятся: {", ".join(labels)}. Все они тебя слышат.'
            else:
                notice = f'Подключилась к голосовому каналу "{channel_name}". Сейчас в канале никого нет, кроме тебя.'
            self.emit_environment_notice(notice)

        # Случай 2: Уже в канале — отслеживаем вход и выход участников
        elif now_connected and prev_connected and prev_channel_id == now_channel_id:
            joined_ids = set(new_members.keys()) - set(prev_members.keys())
            left_ids = set(prev_members.keys()) - set(new_members.keys())

            for j_id in joined_ids:
                label = self._format_discord_member_label(j_id, new_members[j_id])
                self.emit_environment_notice(f'В голосовой канал зашёл(ла) {label}. Теперь он(а) с вами и слышит тебя.')

            for l_id in left_ids:
                label = self._format_discord_member_label(l_id, prev_members.get(l_id, "Пользователь"))
                self.emit_environment_notice(f'{label} покинул(а) голосовой канал.')

            self._discord_channel_members = dict(new_members)

        # Случай 3: Отключились от канала
        elif not now_connected and prev_connected:
            self._prev_discord_connected = False
            self._prev_discord_channel_id = None
            self._discord_channel_members = {}

        self._discord_bridge_status.update(status)

    def get_discord_bridge_status(self) -> Dict[str, Any]:
        st = dict(self._discord_bridge_status)
        if time.time() - float(st.get("updated_at", 0)) > 7.0:
            st["online"] = False
            st["connected_to_voice"] = False
        return st

    def get_audio_output_mode(self) -> str:
        return getattr(self, "_audio_output_mode", "local")

    def _set_speaker_filter_enabled(self, enabled: bool) -> None:
        if getattr(self, "op_manager", None):
            try:
                ops = self.op_manager.get_operation(OpRoles.FILTER_AUDIO) or []
                for op in ops:
                    if getattr(op, "op_id", "") == "speaker" and hasattr(op, "set_enabled"):
                        op.set_enabled(enabled)
            except Exception as e:
                logging.warning(f"Could not toggle speaker filter: {e}")

    async def set_audio_output_mode(self, mode: str) -> Dict[str, Any]:
        mode = "discord" if str(mode).strip().lower() == "discord" else "local"
        self._audio_output_mode = mode
        
        # Автоматическое переключение сцены
        try:
            cfg = Config()
            cfg.active_scene = mode
        except Exception:
            pass
            
        logging.info(f"Audio output mode & active_scene switched to: {mode}")
        
        # Локальный динамик: активен только в режиме local
        self._set_speaker_filter_enabled(mode == "local")

        if self.event_server:
            await self.event_server.broadcast_event("audio_output_mode", {"mode": mode})
            await self.event_server.broadcast_event("discord_control", {"action": "join" if mode == "discord" else "leave"})
        return {"ok": True, "mode": mode}

    def get_pipeline_telemetry(self) -> Dict[str, Any]:
        return {
            "latest": self._last_telemetry,
            "history": list(self._telemetry_history),
            "discord": self.get_discord_bridge_status(),
            "active_providers": self.get_active_providers(),
            "audio_output_mode": self.get_audio_output_mode(),
        }
        
    def clear_telemetry_history(self):
        self._telemetry_history.clear()
        self._last_telemetry = None
            
    ## Async Job Handlers #########################
    
    '''
    Generate responses from the current contexts.
    This does not take an input. Context for what to repond to must be added prior to running this.
    '''
    async def response_pipeline(
        self,
        job_id: str,
        job_type: JobType,
        include_audio: bool = True,
        input_timestamp: float = None,
        input_mode: str = None,
        continue_from_text: str = None,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        speaker_id: str = None,
        stt_provider: str = None,
        stt_confidence: float = None,
        stt_latency_ms: int = None,
        stt_finish_ts: float = None,
        **kwargs
    ):
        
        # Adjust flags based on loaded ops
        if not self.op_manager.get_operation(OpRoles.TTS): include_audio = False
        
        # Broadcast start conditions
        start_time = time.time()
        token_count = 0
        latency = 0
        first_token_sent = False
        first_audio_sent = False
        self._assistant_live_job_id = job_id
        self._assistant_live_reply = ""
        # ── Цветная метка: начало генерации LLM ──
        print_llm_start(source_id=source_id, mode=input_mode or "text")
        await self._handle_broadcast_start(job_id, job_type, {
            "include_audio": include_audio,
            "input_mode": input_mode,
            "source_id": source_id,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "stt_provider": stt_provider,
            "stt_latency_ms": stt_latency_ms,
        })
    
        # Handle MCP stuff
        if self.op_manager.get_operation(OpRoles.MCP):
            self.prompter.add_mcp_usage_prompt(self.mcp_manager.get_tooling_prompt(), self.mcp_manager.get_response_prompt())
            mcp_sys_prompt, mcp_user_prompt = self.prompter.generate_mcp_system_context(), self.prompter.generate_mcp_user_context()
            tooling_response = ""
            async for chunk in self.op_manager.use_operation(OpRoles.MCP, {"instruction_prompt": mcp_sys_prompt, "messages": [RawMessage(mcp_user_prompt)]}):
                tooling_response += chunk['content']

            ## Perform MCP tool calls
            tool_call_results = await self.mcp_manager.use(tooling_response)
            
            ## Add results and usage prompt to prompter
            self.prompter.add_mcp_results(tool_call_results)

        # Get prompts
        instruction_prompt, history = self.prompter.get_sys_prompt(), self.prompter.get_history()

        # ── Interrupted speech awareness & unfinished thought memory ──
        interrupted_tail = (getattr(self, "_assistant_last_partial_reply", "") or "").strip()
        if continue_from_text and isinstance(continue_from_text, str) and continue_from_text.strip():
            tail = continue_from_text.strip()[-1200:]
            instruction_prompt = (
                f"{instruction_prompt}\n\n"
                "### Request to Continue ###\n"
                "The user explicitly asked you to continue what you were saying before being interrupted.\n"
                f"Your unfinished thought stopped at: «{tail}»\n"
                "Seamlessly resume your narrative from the exact point you stopped, developing the thought further in natural Russian.\n"
            )
        elif interrupted_tail:
            tail = interrupted_tail[-1200:]
            instruction_prompt = (
                f"{instruction_prompt}\n\n"
                "### Interrupted Speech Memory ###\n"
                "In your previous turn you were cut off mid-sentence while saying:\n"
                f"«{tail}»\n"
                "You retain full memory and awareness of your unfinished thought.\n"
                "Dialogue guidance:\n"
                "1. If the user made a brief reaction, question, or if your unfinished thought is valuable/relevant: seamlessly finish it and connect to their new remark.\n"
                "2. If the user shifted to a completely different topic: answer their new input naturally without forcing the old topic, while keeping prior context in mind.\n"
                "3. Stay natural: never apologize for being interrupted or say robotic clichés like 'as I was saying'.\n"
            )
        
        # Apply t2t and stream to TTS
        t2t_result = ""
        full_filtered_text = ""
        first_sentence_ready_ts = None
        first_sentence_ms = None
        first_tts_ttfa_ms = None
        first_audio_ts = None
        first_audio_ms = None
        e2e_tts_start_ms = None
        llm_req_start_ts = None
        first_token_ts = None
        llm_ttft_ms = None
        e2e_ttft_ms = None
        tps = 0.0
        
        tts_queue = asyncio.Queue()
        
        async def tts_worker():
            nonlocal full_filtered_text, first_audio_sent, first_tts_ttfa_ms, first_audio_ts, first_audio_ms, e2e_tts_start_ms
            try:
                while True:
                    phrase = await tts_queue.get()
                    if phrase is None: # Sentinel
                        break
                        
                    clean_phrase = clean_phrase_for_tts(phrase)
                    if not clean_phrase:
                        tts_queue.task_done()
                        continue
                        
                    if full_filtered_text and not full_filtered_text.endswith((" ", "\n")):
                        full_filtered_text += " "
                    full_filtered_text += clean_phrase
                    
                    if include_audio:
                        # ── Цветная метка: фраза передана на TTS (очищенная от смайлов и тегов) ──
                        print_tts_phrase(clean_phrase)
                        tts_t0 = time.time()
                        phrase_first_chunk = True
                        try:
                            async for audio_chunk_out in self.op_manager.use_operation(OpRoles.TTS, {"content": clean_phrase}):
                                async for final_audio_chunk_out in self.op_manager.use_operation(OpRoles.FILTER_AUDIO, audio_chunk_out):
                                    now_audio = time.time()
                                    if phrase_first_chunk:
                                        phrase_first_chunk = False
                                        phrase_ttfa = int((now_audio - tts_t0) * 1000)
                                        if first_tts_ttfa_ms is None:
                                            first_tts_ttfa_ms = phrase_ttfa

                                    for ws_chunk in chunk_buffer(base64.b64encode(final_audio_chunk_out['audio_bytes']).decode('utf-8')):
                                        audio_event = {
                                            "audio_bytes": ws_chunk,
                                            "sr": final_audio_chunk_out['sr'],
                                            "sw": final_audio_chunk_out['sw'],
                                            "ch": final_audio_chunk_out['ch'],
                                            "event": "audio_chunk"
                                        }
                                        if not first_audio_sent:
                                            first_audio_sent = True
                                            first_audio_ts = now_audio
                                            first_audio_ms = int((now_audio - start_time) * 1000)
                                            audio_event["tts_start_ms"] = first_audio_ms
                                            audio_event["tts_ttfa_ms"] = first_tts_ttfa_ms
                                            if input_timestamp is not None:
                                                try:
                                                    e2e_tts_start_ms = max(0, int((now_audio - float(input_timestamp)) * 1000))
                                                    audio_event["e2e_tts_start_ms"] = e2e_tts_start_ms
                                                except Exception: pass
                                        await self._handle_broadcast_event(job_id, job_type, audio_event)
                            # ── TTS фраза готова ──
                            print_tts_done(latency_ms=int((time.time() - tts_t0) * 1000))
                        except asyncio.CancelledError:
                            tts_queue.task_done()
                            break
                        except Exception as e:
                            logging.error(f"TTS Worker error: {e}", exc_info=True)
                    
                    tts_queue.task_done()
            except asyncio.CancelledError:
                pass
                
        tts_task = asyncio.create_task(tts_worker())
        sentence_buffer = ""
        boundaries = (". ", "? ", "! ", ".\n", "?\n", "!\n")
        
        llm_req_start_ts = time.time()
        queue_overhead_ms = int((llm_req_start_ts - float(stt_finish_ts)) * 1000) if stt_finish_ts else int((llm_req_start_ts - start_time) * 1000)

        pipeline_cancelled = False
        try:
            async for chunk_out in self.op_manager.use_operation(OpRoles.T2T, {"instruction_prompt": instruction_prompt, "messages": history}):
                chunk_content = chunk_out.get('content', '')
                t2t_result += chunk_content
                sentence_buffer += chunk_content
                
                if chunk_content:
                    self._assistant_live_reply += chunk_content
                
                # Ранняя остановка при попытке вывода системных тегов REQUEST:
                if "REQUEST:" in sentence_buffer or "REQUEST:" in chunk_content:
                    logging.info("[LLM Early-Stop] Пресечена генерация тега 'REQUEST:', поток остановлен.")
                    if "REQUEST:" in sentence_buffer:
                        valid_tail = sentence_buffer.split("REQUEST:")[0].strip()
                        sentence_buffer = ""
                        if valid_tail:
                            await tts_queue.put(valid_tail)
                    if "REQUEST:" in t2t_result:
                        t2t_result = t2t_result.split("REQUEST:")[0].strip()
                    if "REQUEST:" in self._assistant_live_reply:
                        self._assistant_live_reply = self._assistant_live_reply.split("REQUEST:")[0].strip()
                    break
                
                token_count += len(chunk_content.split())
                elapsed = time.time() - llm_req_start_ts
                tps = round(token_count / elapsed, 1) if elapsed > 0 else 0

                if chunk_content and not first_token_sent:
                    first_token_sent = True
                    first_token_ts = time.time()
                    llm_ttft_ms = int((first_token_ts - llm_req_start_ts) * 1000)
                    chunk_out["llm_ttft_ms"] = llm_ttft_ms
                    chunk_out["ttft_ms"] = llm_ttft_ms
                    if input_timestamp is not None:
                        try:
                            e2e_ttft_ms = int((first_token_ts - float(input_timestamp)) * 1000)
                            chunk_out["e2e_ttft_ms"] = e2e_ttft_ms
                        except Exception: pass
                
                chunk_out.update({"tps": tps, "latency": llm_ttft_ms if llm_ttft_ms is not None else int(elapsed * 1000)})
                await self._handle_broadcast_event(job_id, job_type, chunk_out)

                while True:
                    best_idx = -1
                    best_bnd = None
                    for bnd in boundaries:
                        idx = sentence_buffer.find(bnd)
                        if idx != -1 and (best_idx == -1 or idx < best_idx):
                            best_idx = idx
                            best_bnd = bnd
                            
                    if best_idx != -1:
                        phrase = sentence_buffer[:best_idx + len(best_bnd)]
                        sentence_buffer = sentence_buffer[best_idx + len(best_bnd):]
                        if phrase.strip():
                            if first_sentence_ready_ts is None:
                                first_sentence_ready_ts = time.time()
                                first_sentence_ms = int((first_sentence_ready_ts - llm_req_start_ts) * 1000)
                            await tts_queue.put(phrase)
                    else:
                        break 
        except asyncio.CancelledError:
            pipeline_cancelled = True
            raise
        finally:
            if pipeline_cancelled:
                # Мгновенная остановка: отменяем TTS воркер и сбрасываем очередь фраз
                tts_task.cancel()
                while not tts_queue.empty():
                    try:
                        tts_queue.get_nowait()
                        tts_queue.task_done()
                    except Exception:
                        break
                try:
                    await tts_task
                except asyncio.CancelledError:
                    pass
            else:
                if sentence_buffer.strip():
                    if first_sentence_ready_ts is None:
                        first_sentence_ready_ts = time.time()
                        first_sentence_ms = int((first_sentence_ready_ts - llm_req_start_ts) * 1000)
                    await tts_queue.put(sentence_buffer.strip())
                
                await tts_queue.put(None)
                await tts_task

        debug_prompt_events = False
        try:
            debug_prompt_events = bool(getattr(Config(), "broadcast_debug_prompt_events", False))
        except Exception:
            pass

        if debug_prompt_events:
            await self._handle_broadcast_event(job_id, job_type, {"instruction_prompt": instruction_prompt})
            await self._handle_broadcast_event(job_id, job_type, {"history": [msg.to_dict() for msg in history]})
            await self._handle_broadcast_event(job_id, job_type, {"raw_content": t2t_result})

        if not t2t_result or not t2t_result.strip():
            await self._handle_broadcast_event(job_id, job_type, {
                "event": "empty_response",
                "reason": "t2t_empty"
            })
            await self._handle_broadcast_success(job_id, job_type)
            logging.warning(f"Response job {job_id} produced empty T2T output.")
            if self._assistant_live_job_id == job_id:
                self._assistant_live_job_id = None
                self._assistant_live_reply = ""
            return

        if full_filtered_text:
            self.prompter.add_chat(self.prompter.character_name, full_filtered_text)
            self._assistant_last_full_reply = full_filtered_text[-2000:]
            self._assistant_last_partial_reply = ""

        if self._assistant_live_job_id == job_id:
            self._assistant_live_job_id = None
            self._assistant_live_reply = ""

        # ── Расчет и отправка сводной телеметрии конвейера ──
        active_provs = self.get_active_providers()
        t2t_info = active_provs.get("t2t", {})
        tts_info = active_provs.get("tts", {})
        stt_info = active_provs.get("stt", {})
        
        speech_start_ts = kwargs.get("speech_start_ts")
        speech_end_ts = kwargs.get("speech_end_ts") or (input_timestamp if input_mode == "voice" else None)
        send_timestamp = kwargs.get("send_timestamp") or (input_timestamp if input_mode != "voice" else None)

        # Момент старта запроса:
        # Для голоса: ПОСЛЕДНИЙ пакет речи пользователя (момент завершения фразы, speech_end_ts)
        # Для текста: момент отправки сообщения в чат (send_timestamp)
        t_start = float((speech_end_ts if input_mode == "voice" else None) or send_timestamp or input_timestamp or start_time)

        # Общая задержка (отправка в чат / последний пакет голоса -> начало воспроизведения звука)
        if first_audio_ts is not None:
            total_latency_ms = max(0, int((first_audio_ts - t_start) * 1000))
        elif first_token_ts is not None:
            total_latency_ms = max(0, int((first_token_ts - t_start) * 1000))
        else:
            total_latency_ms = max(0, int((time.time() - t_start) * 1000))

        user_speech_duration_ms = max(0, int((speech_end_ts - speech_start_ts) * 1000)) if (speech_end_ts and speech_start_ts and speech_end_ts >= speech_start_ts) else None
        turn_taking_latency_ms = max(0, int((first_audio_ts - speech_end_ts) * 1000)) if (first_audio_ts and speech_end_ts) else None

        telemetry = {
            "job_id": job_id,
            "input_mode": input_mode or "text",
            "timestamp": time.time(),
            "total_latency_ms": total_latency_ms,
            "response_latency_ms": total_latency_ms,
            "user_speech_duration_ms": user_speech_duration_ms,
            "turn_taking_latency_ms": turn_taking_latency_ms or total_latency_ms,
            "start_point": "speech_end_packet" if input_mode == "voice" else "chat_message_sent",
            "end_point": "voice_playback_start",
            "stt": {
                "provider": stt_provider or stt_info.get("id"),
                "model": stt_info.get("model"),
                "type": stt_info.get("type", "local"),
                "latency_ms": stt_latency_ms,
                "confidence": stt_confidence,
            },
            "queue_overhead_ms": max(0, queue_overhead_ms) if queue_overhead_ms is not None else 0,
            "llm": {
                "provider": t2t_info.get("id", "unknown"),
                "model": t2t_info.get("model", "default"),
                "type": t2t_info.get("type", "cloud"),
                "ttft_ms": llm_ttft_ms or 0,
                "e2e_ttft_ms": e2e_ttft_ms,
                "duration_ms": int((time.time() - (llm_req_start_ts or start_time)) * 1000),
                "token_count": token_count,
                "char_count": len(full_filtered_text or t2t_result),
                "tps": round(token_count / max(0.001, (time.time() - (llm_req_start_ts or start_time))), 1) if token_count > 0 else 0.0,
                "first_sentence_ms": first_sentence_ms,
            },
            "tts": {
                "provider": tts_info.get("id", "unknown"),
                "model": tts_info.get("model", "default"),
                "type": tts_info.get("type", "cloud"),
                "ttfa_ms": first_tts_ttfa_ms,
                "first_audio_ms": first_audio_ms,
                "e2e_voice_start_ms": e2e_tts_start_ms,
            },
            "total_pipeline_ms": int((time.time() - start_time) * 1000),
        }
        self._last_telemetry = telemetry
        self._telemetry_history.append(telemetry)

        await self._handle_broadcast_event(job_id, job_type, {
            "event": "telemetry",
            "telemetry": telemetry
        })

        # ── Цветная метка: LLM завершён ──
        print_llm_done(
            chars=len(full_filtered_text),
            latency_ms=llm_ttft_ms if llm_ttft_ms is not None else (latency if latency else None),
            tps=tps if 'tps' in dir() else None,
        )

        # Broadcast completion
        await self._handle_broadcast_success(job_id, job_type, {"telemetry": telemetry})
        logging.info(f"Response job {job_id} completed. Content: '{full_filtered_text[:100]}...'") 


    # Context modification
    async def clear_context(
        self,
        job_id: str,
        job_type: JobType
    ):
        await self._handle_broadcast_start(job_id, job_type, {})
        self.prompter.clear_history()
        await self._handle_broadcast_success(job_id, job_type)
        
    async def configure_context(
        self,
        job_id: str,
        job_type: JobType,
        name_translations: Dict[str, str] = None,
        character_name: str = None,
        history_length: int = None,
        instruction_prompt_filename: str = None,
        character_prompt_filename: str = None,
        scene_prompt_filename: str = None
    ):
        await self._handle_broadcast_start(job_id, job_type, {
            "name_translations": name_translations,
            "character_name": character_name,
            "history_length": history_length,
            "instruction_prompt_filename": instruction_prompt_filename,
            "character_prompt_filename": character_prompt_filename,
            "scene_prompt_filename": scene_prompt_filename
        })
        payload = dict()
        if name_translations: payload |= {"name_translations": name_translations}
        if character_name: payload |= {"character_name": character_name}
        if history_length: payload |= {"history_length": history_length}
        if instruction_prompt_filename: payload |= {"instruction_prompt_filename": instruction_prompt_filename}
        if character_prompt_filename: payload |= {"character_prompt_filename": character_prompt_filename}
        if scene_prompt_filename: payload |= {"scene_prompt_filename": scene_prompt_filename}
        
        await self.prompter.configure(payload)
        
        await self._handle_broadcast_success(job_id, job_type)

    async def append_request_context(
        self, 
        job_id: str, 
        job_type: JobType, 
        content: str = None
    ):
        await self._handle_broadcast_start(job_id, job_type, {"content": content})
        self.prompter.add_request(content)
        last_line_o = self.prompter.history[-1]
        await self._handle_broadcast_event(job_id, job_type, {
            "timestamp": last_line_o.time.timestamp(),
            "content": last_line_o.message,
            "line": last_line_o.to_line()
        })
        await self._handle_broadcast_success(job_id, job_type)
        
    def resolve_speaker_name(
        self,
        speaker_id: str | None = None,
        raw_user: str | None = None,
        source_id: str | None = None,
    ) -> str:
        """
        Умное разрешение имени пользователя:
        - Ищет в configs/known_users.json и Config().known_users
        - Поддерживает как простые строки ("id": "Имя"), так и структуры ("id": {"name": "Имя", "discord_nick": "Nick", "aliases": [...]})
        - При появлении нового пользователя в Discord автоматически регистрирует его в known_users.json
        """
        cfg = Config()
        active_scene = getattr(cfg, "active_scene", "local")
        known_users = dict(getattr(cfg, "known_users", {}))

        dynamic_users_path = os.path.join(cfg.CONFIG_DIR, "known_users.json")
        file_users = {}
        if os.path.isfile(dynamic_users_path):
            try:
                with open(dynamic_users_path, "r", encoding="utf-8") as f:
                    file_users = json.load(f)
                    if isinstance(file_users, dict):
                        known_users.update(file_users)
            except Exception as e:
                logging.warning(f"Error reading dynamic known_users.json: {e}")

        def _extract_name(val: Any) -> str | None:
            if isinstance(val, dict):
                return val.get("name") or val.get("discord_nick")
            if isinstance(val, str) and val.strip():
                return val.strip()
            return None

        # 1. Сцена Discord или аудио из Discord
        if active_scene == "discord" or source_id == "discord":
            spk_str = str(speaker_id).strip() if speaker_id else ""
            raw_str = str(raw_user).strip() if raw_user else ""

            # Ищем по Discord snowflake ID
            if spk_str and spk_str in known_users:
                resolved = _extract_name(known_users[spk_str])
                if resolved:
                    return resolved

            # Ищем по никнейму
            if raw_str and raw_str in known_users:
                resolved = _extract_name(known_users[raw_str])
                if resolved:
                    return resolved

            # Новый пользователь Discord — используем его текущий никнейм
            fallback_name = raw_str if raw_str and raw_str.lower() not in ("user", "unknown", "none") else "Незнакомец"
            if spk_str and spk_str not in known_users:
                try:
                    new_entry = {
                        "name": fallback_name,
                        "discord_nick": fallback_name,
                        "aliases": [fallback_name],
                    }
                    known_users[spk_str] = new_entry
                    file_users[spk_str] = new_entry
                    with open(dynamic_users_path, "w", encoding="utf-8") as f:
                        json.dump(file_users, f, ensure_ascii=False, indent=2)
                    logging.info(f"Auto-registered new Discord user in known_users.json: id={spk_str}, nick={fallback_name}")
                except Exception as e:
                    logging.warning(f"Could not auto-register new Discord user: {e}")

            return fallback_name

        # 2. Локальный микрофон / веб-чат
        mic_val = known_users.get("mic")
        resolved = _extract_name(mic_val)
        return resolved or "Вова"

    async def append_conversation_context_text(
        self, 
        job_id: str, 
        job_type: JobType, 
        user: str = None, 
        timestamp: int = None, 
        content: str = None,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        utterance_ids: List[str] = None,
        speaker_id: str = None,
        stt_provider: str = None,
        stt_confidence: float = None,
        stt_latency_ms: int = None,
    ):
        await self._handle_broadcast_start(job_id, job_type, {
            "user": user,
            "timestamp": timestamp,
            "content": content,
            "source_id": source_id,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "utterance_ids": utterance_ids,
            "speaker_id": speaker_id,
            "stt_provider": stt_provider,
            "stt_confidence": stt_confidence,
            "stt_latency_ms": stt_latency_ms,
        })
        
        final_user = self.resolve_speaker_name(speaker_id=speaker_id, raw_user=user, source_id=source_id)

        self.prompter.add_chat(
            final_user,
            content,
            time=(
                datetime.datetime.fromtimestamp(timestamp) \
                if not isinstance(timestamp, datetime.datetime) else timestamp
            )
        )
        last_line_o = self.prompter.history[-1]
        await self._handle_broadcast_event(job_id, job_type, {
            "user": last_line_o.user,
            "timestamp": last_line_o.time.timestamp(),
            "content": last_line_o.message,
            "line": last_line_o.to_line(),
            "source_id": source_id,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "utterance_ids": utterance_ids,
            "speaker_id": speaker_id,
            "stt_provider": stt_provider,
            "stt_confidence": stt_confidence,
            "stt_latency_ms": stt_latency_ms,
        })
        await self._handle_broadcast_success(job_id, job_type)
        
    async def append_conversation_context_audio(
        self,
        job_id: str,
        job_type: JobType,
        user: str = None,
        timestamp: int = None,
        audio_bytes: str = None,
        sr: int = None,
        sw: int = None,
        ch: int = None,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        speaker_id: str = None,
    ):
        # Legacy job path kept for compatibility.
        # Delegate to the main immediate audio pipeline so behavior stays identical
        # between REST `/api/context/conversation/audio` and queued job flow.
        await self._handle_broadcast_start(
            job_id,
            job_type,
            {
                "user": user,
                "timestamp": timestamp,
                "sr": sr,
                "sw": sw,
                "ch": ch,
                "audio_bytes": (audio_bytes is not None),
                "source_id": source_id,
                "turn_id": turn_id,
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
            }
        )
        await self.process_audio_immediate({
            "user": user,
            "timestamp": timestamp,
            "audio_bytes": audio_bytes,
            "sr": sr,
            "sw": sw,
            "ch": ch,
            "source_id": source_id,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
        })
        await self._handle_broadcast_success(job_id, job_type)
            
    def _interrupt_allowed_for_speaker(self, speaker_id: str | None) -> bool:
        policy = str(self._get_microphone_config().get("interrupt_speaker_policy", "any") or "any").strip().lower()
        if policy == "any":
            return True

        if self.job_current is None or self.job_current.done():
            return True

        current_job_type, _ = self.job_map.get(self.job_current_id, (None, None))
        if current_job_type != JobType.RESPONSE:
            return True

        active_speaker = self._response_job_speakers.get(self.job_current_id)
        speaker_norm = str(speaker_id).strip() if speaker_id else ""
        active_norm = str(active_speaker).strip() if active_speaker else ""

        if policy == "same_only":
            return bool(speaker_norm and active_norm and speaker_norm == active_norm)
        if policy == "same_or_unknown":
            if not speaker_norm or not active_norm:
                return True
            return speaker_norm == active_norm
        return True

    async def process_audio_immediate(self, request_data: dict):
        """Немедленная обработка аудио (вне очереди) для мгновенного перебивания."""
        audio_bytes_b64 = request_data.get("audio_bytes")
        if not audio_bytes_b64:
            return

        try:
            audio_bytes = base64.b64decode(audio_bytes_b64)
        except Exception:
            logging.warning("Immediate STT received invalid base64 payload.")
            await self._emit_stt_status("unavailable", reason="invalid_audio_payload")
            return

        try:
            sr = int(request_data.get("sr", 16000))
        except Exception:
            sr = 16000
        try:
            sw = int(request_data.get("sw", 2))
        except Exception:
            sw = 2
        try:
            ch = int(request_data.get("ch", 1))
        except Exception:
            ch = 1

        user = request_data.get("user", "user")
        try:
            timestamp = float(request_data.get("timestamp", time.time()))
        except Exception:
            timestamp = time.time()
        audio_dur = len(audio_bytes) / float(max(1, sr * sw * ch))
        raw_start = request_data.get("speech_start_ts")
        if raw_start is not None:
            try:
                speech_start_ts = float(raw_start)
            except Exception:
                speech_start_ts = max(0.0, timestamp - audio_dur)
        else:
            # Если клиент не передал speech_start_ts, то старт записи был audio_dur секунд назад
            speech_start_ts = max(0.0, timestamp - audio_dur)

        raw_end = request_data.get("speech_end_ts")
        if raw_end is not None:
            try:
                speech_end_ts = float(raw_end)
            except Exception:
                speech_end_ts = speech_start_ts + audio_dur
        else:
            speech_end_ts = speech_start_ts + audio_dur
        source_id = self._safe_source_id(request_data.get("source_id"))
        
        # Фильтрация аудио-потоков по активному режиму (Discord vs Local)
        current_mode = self.get_audio_output_mode()
        if current_mode == "discord" and source_id == "mic":
            logging.debug("Drop mic audio: Discord mode is active")
            return
        if current_mode == "local" and source_id == "discord":
            logging.debug("Drop Discord audio: Local mode is active")
            return

        turn_id = str(request_data.get("turn_id") or uuid.uuid4())
        utterance_id = str(request_data.get("utterance_id") or uuid.uuid4())
        expected_stop_cmd = request_data.get("expected_stop_cmd")

        hook_meta = {}
        try:
            hook_meta = await apply_pre_stt_hooks(request_data, audio_bytes, sr, sw, ch)
        except Exception:
            logging.warning("pre-STT hooks failed", exc_info=True)
        speaker_id = hook_meta.get("speaker_id") or request_data.get("speaker_id")

        # 1. STT inference (only final chunks affect context pipeline).
        prompt = self.prompter.get_history_text() or ""
        content = ""
        stt_provider = "unknown"
        stt_confidence = None
        stt_latency_ms = None
        stt_error = None

        try:
            async for out_d in self.op_manager.use_operation(OpRoles.STT, {
                "prompt": prompt,
                "audio_bytes": audio_bytes,
                "sr": sr,
                "sw": sw,
                "ch": ch,
                "source_id": source_id,
                "turn_id": turn_id,
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "input_timestamp_ms": int(timestamp * 1000),
            }):
                stt_provider = str(out_d.get("provider") or stt_provider)
                if out_d.get("confidence") is not None:
                    stt_confidence = out_d.get("confidence")
                if out_d.get("stt_latency_ms") is not None:
                    stt_latency_ms = out_d.get("stt_latency_ms")
                if out_d.get("speaker_id") and not speaker_id:
                    speaker_id = out_d.get("speaker_id")
                if out_d.get("stt_error"):
                    stt_error = str(out_d.get("stt_error"))

                if not bool(out_d.get("is_final", True)):
                    continue

                chunk_text = str(out_d.get("text") or out_d.get("transcription") or "").strip()
                if chunk_text:
                    if content:
                        content += " "
                    content += chunk_text
        except Exception as e:
            logging.error("Immediate STT failed: %s", e, exc_info=True)
            await self._emit_stt_status(
                "unavailable",
                reason="stt_exception",
                source_id=source_id,
                turn_id=turn_id,
                utterance_id=utterance_id,
            )
            return

        if stt_error in {"timeout", "unavailable", "restarting"}:
            await self._emit_stt_status(
                stt_error,
                reason="stt_provider_signal",
                source_id=source_id,
                turn_id=turn_id,
                utterance_id=utterance_id,
                provider=stt_provider,
            )

        # ── Цветная метка: STT результат ──
        if content and content.strip():
            print_stt_done(content, source=source_id, latency_ms=stt_latency_ms)
        else:
            print_stt("", source=source_id)  # тихо пропускаем пустые

        if not content or len(content.strip()) == 0:
            self._ensure_pending_voice_response_scheduled()
            await self._record_stt_metrics(
                source_id=source_id,
                turn_id=turn_id,
                utterance_id=utterance_id,
                provider=stt_provider,
                latency_ms=stt_latency_ms,
                text="",
                detected_stop_cmd=False,
                expected_stop_cmd=bool(expected_stop_cmd) if isinstance(expected_stop_cmd, bool) else None,
            )
            return

        # 2. Barge-in / intent classification.
        words = re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ-]+", content.lower().strip())
        fillers = {
            "угу", "ага", "понятно", "ясно", "да", "так", "хорошо", "ок", "слышу", "мгм", "ладно",
            "понял", "ого", "ммм", "эмм", "хмм", "интересно", "спасибо", "спасибки", "круто", "кайф",
            "вау", "прикольно", "отлично", "здорово", "агась", "угум", "да-да", "ну да", "ага-ага",
            "ой", "ай", "упс", "ох", "эх", "ух"
        }
        cfg_globals = getattr(Config(), "_config", {})
        
        stop_words = set(cfg_globals.get("stt_stop_words", [
            "стой", "стойте", "постой", "постойте", "остой", "стай", "сто", "ста",
            "стоп", "стопэ", "стопе", "стопчик",
            "хватит", "хорош", "замолчи", "молчи", "помолчи", "молчать",
            "подожди", "погоди", "погодите", "подождите",
            "тихо", "тише", "остановись", "останови", "тормози", "притормози", "выключи", "отключи"
        ]))
        stop_stems = (
            "стоп", "стой", "постой", "остой", "стай", "подож", "погод",
            "хват", "хорош", "замолч", "молч", "тихо", "тише", "останов",
            "тормоз", "выключ", "отключ"
        )
        wake_words = {
            "нира", "нера", "nira",
            # Typical Sherpa misses for "Нира"
            "мира", "миру", "миром", "миро", "ниру", "нире", "нирой", "ниры", "нерра"
        }
        canonical_wake_word = "нира"
        respond_to_short_emotes = True
        short_emote_words = {
            "ха", "хаха", "ха-ха", "ахах", "ахаха", "ахахах",
            "хех", "хе-хе", "гы", "гыы", "гы-гы",
            "лол", "ржу", "ржом", "мда", "гм", "хм",
        }

        mic_cfg = self._get_microphone_config()
        extra_wake_words = mic_cfg.get("wake_words", [])
        if isinstance(extra_wake_words, list):
            for w in extra_wake_words:
                w = str(w).strip().lower()
                if w:
                    wake_words.add(w)
        extra_wake_aliases = mic_cfg.get("wake_word_aliases", [])
        if isinstance(extra_wake_aliases, list):
            for w in extra_wake_aliases:
                w = str(w).strip().lower()
                if w:
                    wake_words.add(w)
        respond_to_short_emotes = bool(mic_cfg.get("respond_to_short_emotes", True))
        extra_short_emotes = mic_cfg.get("short_emote_words", [])
        if isinstance(extra_short_emotes, list):
            for w in extra_short_emotes:
                w = str(w).strip().lower()
                if w:
                    short_emote_words.add(w)
        try:
            prompter_cfg = Config().prompter or {}
            cfg_name = str(prompter_cfg.get("character_name", "")).strip().lower()
            if cfg_name:
                canonical_wake_word = cfg_name
                wake_words.add(cfg_name)
        except Exception:
            pass
        continue_intent = self._is_continue_intent(content)

        # --- Whisper Hallucination Filter ---
        hallucination_markers = [
            "jurisprudence", "kanami", "rison", "nood", "compon", "oard", "irmi", "о чём ты",
            "ᵉ", "ᵒ", "убтрайт", "субтитры", "amara.org", "спасибо за просмотр", "сказать?",
            "продолжение следует", "редактор субтитров", "дима торзок", "диматорзок", "dimatorzok", "торзок",
            "игорь негода", "игорь нигода", "негода", "нигода",
            "до скорой встречи", "до новых встреч", "ставьте лайки", "подписывайтесь",
            "конец связи", "конец фильма", "всем пока", "приятного просмотра",
            "спасибо за внимание", "спасибо за урок", "продолжение в описании",
        ]
        is_hallucination = any(marker in content.lower() for marker in hallucination_markers)
        
        # Точное совпадение типовых галлюцинаций Whisper на вздохах и тишине
        clean_norm_text = re.sub(r"[^\w\s]", "", content.lower()).strip()
        common_hallucinations = {
            "продолжение следует", "субтитры", "субтитры сделал", "субтитры создавал", "субтитры делал",
            "дима торзок", "диматорзок", "торзок", "игорь негода", "игорь нигода", "негода", "нигода",
            "спасибо за просмотр", "до скорой встречи", "до свидания", "всем привет", "пока пока", "конец",
            "спасибо за внимание", "редактор субтитров", "переводчик", "озвучено"
        }
        if clean_norm_text in common_hallucinations or any(clean_norm_text.startswith(h) for h in ("субтитры", "редактор субтитров")):
            is_hallucination = True

        if not is_hallucination and len(words) <= 4:
            # Если в короткой фразе от русской модели больше английских букв, чем русских - это мусор
            en_chars = len(re.findall(r'[a-zA-Z]', content))
            ru_chars = len(re.findall(r'[а-яА-Я]', content))
            if en_chars > ru_chars and en_chars > 0:
                is_hallucination = True

        if is_hallucination:
            logging.info("Whisper hallucination filtered: '%s'", content)
            self._ensure_pending_voice_response_scheduled()
            return
        # ------------------------------------

        if not words:
            self._ensure_pending_voice_response_scheduled()
            await self._record_stt_metrics(
                source_id=source_id,
                turn_id=turn_id,
                utterance_id=utterance_id,
                provider=stt_provider,
                latency_ms=stt_latency_ms,
                text=content,
                detected_stop_cmd=False,
                expected_stop_cmd=bool(expected_stop_cmd) if isinstance(expected_stop_cmd, bool) else None,
            )
            return

        # 1. Проверка на чистое мычание/филлеры/паралингвистику (м-м-м, эээ, хм, ага, угу, ну, да, ой, ай)
        def _check_pure_filler(text_raw: str, word_list: list) -> bool:
            clean_s = re.sub(r"[^\w\s]", " ", text_raw.lower().replace("ё", "е")).strip()
            if not clean_s:
                return True
            parts = clean_s.split()
            all_filler_words = {
                "м", "мм", "ммм", "э", "ээ", "эээ", "хм", "хмм", "гм",
                "ага", "угу", "ну", "да", "мда", "эх", "ох", "ок", "а", "ам", "эм",
                "ой", "ай", "яй", "упс", "ого", "ух", "ах", "ха", "хе", "мгм"
            }
            return all((p in all_filler_words or bool(re.fullmatch(r"^[мmэeаaоo]+$", p))) for p in parts)

        is_pure_filler = _check_pure_filler(content, words)

        # 2. Семантический Stop-детектор (без жесткого словаря: корни + fuzzy Levenshtein)
        def _check_stop_intent(text_raw: str, word_list: list) -> bool:
            # Стоп-команда обычно короткая (1-3 слова)
            if len(word_list) > 3:
                return False
            stop_prefixes = (
                "стой", "стойте", "стоп", "stop", "стай", "остой", "постой", "пастой",
                "останов", "приостанов",
                "хват", "хорош",
                "молч", "тиш", "тихо", "замолк", "помолч", "заткн", "цыц",
                "погод", "обожд", "подожд",
                "баст", "прекрат", "пауз", "отстав"
            )
            stop_targets = ["постой", "хватит", "подожди", "замолчи", "погоди", "остановись", "прекрати", "stop"]
            import difflib
            for w in word_list:
                w_norm = w.lower().replace("ё", "е").strip()
                if not w_norm:
                    continue
                if w_norm in ("сто", "стоп", "стой", "stop") or any(w_norm.startswith(pre) for pre in stop_prefixes):
                    return True
                if len(w_norm) >= 5:
                    for target in stop_targets:
                        if difflib.SequenceMatcher(None, w_norm, target).ratio() >= 0.78:
                            return True
            return False

        is_stop_cmd = _check_stop_intent(content, words)

        # 3. Содержательная (осмысленная) речь vs одиночные звуки/шорохи
        non_filler_words = [w for w in words if w.lower().replace("ё", "е") not in fillers and not re.fullmatch(r"^[мmэe]+$", w.lower())]
        wake_word_hit = any(w in wake_words for w in non_filler_words)
        is_wake_word_only = wake_word_hit and len(non_filler_words) == 1 and not is_stop_cmd

        def _check_substantive_speech(text_raw: str, word_list: list, non_fillers: list, is_wake_hit: bool, is_filler: bool) -> bool:
            if is_filler:
                return False
            # 1. Обращение к Нире по имени
            if is_wake_hit:
                return True
            # 2. Вопрос к Нире
            q_words = ("скажи", "расскажи", "какой", "какая", "почему", "зачем", "где", "когда", "кто", "сколько", "что", "как", "куда", "откуда")
            if "?" in text_raw or any(w.lower().replace("ё", "е") in q_words for w in word_list):
                return True
            # 3. Связная фраза от 3 содержательных слов
            if len(non_fillers) >= 3:
                return True
            return False

        is_substantive = _check_substantive_speech(content, words, non_filler_words, wake_word_hit, is_pure_filler)

        def _is_laughter_like(word: str) -> bool:
            w = word.strip().lower().replace("-", "")
            if len(w) < 2:
                return False
            return bool(re.fullmatch(r"(ха|ах|хе|ех){2,}", w))

        short_emote_hit = respond_to_short_emotes and any(
            (_is_laughter_like(w) or (w in short_emote_words)) for w in words
        )

        # [Barge-in Rework] Мягкие прерывания и Режим слушателя
        now = time.time()
        timeout_s = float(getattr(Config(), "stt_listening_timeout_s", 15) or 15)
        
        # Проверяем авто-сброс режима слушателя по долгому таймауту тишины
        if getattr(self, "is_listening", False):
            last_activity = getattr(self, "_listening_mode_last_activity", 0.0)
            if last_activity and (now - last_activity) > timeout_s:
                self.is_listening = False
                logging.info("Nira exited listening mode due to silence timeout (%.1fs).", timeout_s)

        content_norm = content.lower().replace("ё", "е")
        listen_triggers = set(cfg_globals.get("stt_listen_triggers", ["послушай", "послушай меня", "дай я расскажу", "хочу рассказать", "сейчас расскажу", "слушай"]))
        is_listen_trigger = any(t.lower().replace("ё", "е") in content_norm for t in listen_triggers)
        
        if is_listen_trigger:
            self.is_listening = True
            self._listening_mode_last_activity = now
            logging.info("Nira entered listening mode (via listen trigger).")

        release_triggers = set(cfg_globals.get("stt_release_triggers", ["что думаешь", "как тебе", "что скажешь", "отвечай", "твое мнение", "я все", "я всё", "все", "всё", "конец", "готово"]))
        is_release = any(t.lower().replace("ё", "е") in content_norm for t in release_triggers)
        
        # Прямой вопрос к Нире (например, "Нира, сколько сейчас времени?") также снимает режим слушателя
        question_words = ("скажи", "расскажи", "какой", "какая", "почему", "зачем", "где", "когда", "кто", "сколько", "что", "как")
        is_direct_question = wake_word_hit and ("?" in content or any(w in words for w in question_words))

        if getattr(self, "is_listening", False):
            if is_release:
                self.is_listening = False
                logging.info("Nira exited listening mode (via release trigger).")
                resolved_speaker = self.resolve_speaker_name(speaker_id=speaker_id, raw_user=user, source_id=source_id)
                content += f" [System: {resolved_speaker} закончил(а) рассказ. Дай краткий, естественный комментарий по услышанному, обратившись к {resolved_speaker} по имени.]"
            elif continue_intent:
                self.is_listening = False
                logging.info("Nira exited listening mode (via continue-intent).")
            elif is_direct_question:
                self.is_listening = False
                logging.info("Nira exited listening mode (via direct question to Nira).")
            else:
                # Обновляем таймер активности, чтобы тишина считалась от последней фразы
                self._listening_mode_last_activity = now

        # Решение о генерации ответа:
        # На чистое мычание/филлеры Нира НИКОГДА не генерирует ответ!
        if is_pure_filler:
            should_respond = False
        else:
            should_respond = not getattr(self, "is_listening", False)
        
        # Активна ли Нира в данный момент (синтез TTS или генерация LLM)
        is_assistant_speaking_or_thinking = (
            self.job_current is not None 
            and not self.job_current.done() 
            and self.job_map.get(self.job_current_id, (None, None))[0] == JobType.RESPONSE
        )

        # Прерывание (barge-in):
        # Прерываем речь Ниры ТОЛЬКО если:
        # 1) Пользователь просит остановиться (is_stop_cmd)
        # 2) ИЛИ пользователь говорит что-то осмысленное (is_substantive)
        # Одиночные случайные слова, шумы или шорохи НЕ ПРЕРЫВАЮТ!
        is_significant = is_assistant_speaking_or_thinking and (is_stop_cmd or is_substantive)
        
        # При команде остановки («постой», «бастай», «стой», «стоп», «хватит»):
        # 1) Переводим её в режим слушателя
        # 2) Нира замолкает и не комментирует саму команду остановки
        if is_stop_cmd:
            should_respond = False
            self.is_listening = True
            self._listening_mode_last_activity = now
            logging.info("Nira interrupted by stop intent ('%s') and entered listening mode.", content)

        if is_wake_word_only:
            content = canonical_wake_word

        if is_significant:
            if self._can_interrupt_assistant(speaker_id=speaker_id, reason="stop_intent" if is_stop_cmd else "user_barge_in"):
                reason = "stop_word" if is_stop_cmd else "user_barge_in"
                self._interrupt_jobs(reason=reason)
                asyncio.create_task(self._handle_broadcast_event("GLOBAL_STOP", JobType.RESPONSE, {
                    "event": "stop_audio",
                    "reason": reason,
                    "source_id": source_id,
                    "turn_id": turn_id,
                    "utterance_id": utterance_id,
                    "speaker_id": speaker_id,
                }))
            else:
                logging.info("Interruption ignored by speaker policy or assistant condition.")
                await self._emit_stt_status(
                    "interrupt_ignored",
                    reason="condition_suppressed",
                    source_id=source_id,
                    turn_id=turn_id,
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                )

        await self._record_stt_metrics(
            source_id=source_id,
            turn_id=turn_id,
            utterance_id=utterance_id,
            provider=stt_provider,
            latency_ms=stt_latency_ms,
            text=content,
            detected_stop_cmd=is_stop_cmd,
            expected_stop_cmd=bool(expected_stop_cmd) if isinstance(expected_stop_cmd, bool) else None,
        )

        # 3. Buffer chunks into a single user turn and answer after short quiet window.
        await self._buffer_voice_turn(
            user=user,
            timestamp=timestamp,
            content=content,
            continue_intent=continue_intent,
            should_respond=should_respond,
            source_id=source_id,
            turn_id=turn_id,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            stt_provider=stt_provider,
            stt_confidence=stt_confidence,
            stt_latency_ms=stt_latency_ms,
            speech_start_ts=speech_start_ts,
            speech_end_ts=speech_end_ts,
            is_direct_question=is_direct_question,
        )

    async def on_user_speech_start(self, request_data: Dict[str, Any] | None = None):
        """Early barge-in signal from VAD start (before STT final transcript)."""
        request_data = request_data or {}
        source_id = self._safe_source_id(request_data.get("source_id"))
        current_mode = self.get_audio_output_mode()
        # Если источник не совпадает с текущим активным режимом, полностью игнорируем!
        # Не сбрасываем таймер ответа и не обновляем временную метку тишины!
        if current_mode == "discord" and source_id == "mic":
            return
        if current_mode == "local" and source_id == "discord":
            return

        self._last_speech_start_ts = time.time()
        self._cancel_pending_voice_response()
        turn_id = request_data.get("turn_id")
        speaker_id = request_data.get("speaker_id")

        interrupt_mode = "soft"
        try:
            mic_cfg = Config().microphone or {}
            interrupt_mode = str(mic_cfg.get("speech_start_interrupt_mode", "soft")).strip().lower()
        except Exception:
            pass

        # soft mode: do not cancel active response by speech onset alone.
        # We still cancel deferred auto-response and wait for STT significance check.
        if interrupt_mode != "hard":
            return

        if not self._interrupt_allowed_for_speaker(speaker_id):
            return

        # 1. Если генерация ещё выполняется — прерываем её
        if self.job_current is not None and not self.job_current.done():
            current_job_type, _ = self.job_map.get(self.job_current_id, (None, None))
            if current_job_type == JobType.RESPONSE:
                self._interrupt_jobs(reason="user_voice_start")

        # 2. Мгновенно сбрасываем локальные колонки
        self._stop_local_speakers()

        # 3. Мгновенно глушим воспроизведение аудио в Discord и браузере
        await self._handle_broadcast_event("GLOBAL_STOP", JobType.RESPONSE, {
            "event": "stop_audio",
            "reason": "user_voice_start",
            "source_id": source_id,
            "turn_id": turn_id,
            "speaker_id": speaker_id,
        })

    async def register_custom_context(
        self,
        job_id: str,
        job_type: JobType,
        context_id: str = None,
        context_name: str = None,
        context_description: str = None
    ):
        await self._handle_broadcast_start(job_id, job_type, {"context_id": context_id, "context_name": context_name, "context_description": context_description})
        self.prompter.register_custom_context(context_id, context_name, context_description=context_description)
        await self._handle_broadcast_success(job_id, job_type)
    
    async def remove_custom_context(self,
        job_id: str,
        job_type: JobType,
        context_id: str = None
    ):
        await self._handle_broadcast_start(job_id, job_type, {"context_id": context_id})
        self.prompter.remove_custom_context(context_id)
        await self._handle_broadcast_success(job_id, job_type)
    
    async def add_custom_context(
        self,
        job_id: str,
        job_type: JobType,
        context_id: str = None,
        context_contents: str = None,
        timestamp: int = None
    ):
        await self._handle_broadcast_start(job_id, job_type, {"context_id": context_id, "context_contents": context_contents, "timestamp": timestamp})
        if timestamp is not None: timestamp = datetime.datetime.fromtimestamp(timestamp)
        self.prompter.add_custom_context(context_id, context_contents)
        last_line_o = self.prompter.history[-1]
        await self._handle_broadcast_event(job_id, job_type, {
            "timestamp": last_line_o.time.timestamp(),
            "content": last_line_o.message,
            "line": last_line_o.to_line()
        })
        await self._handle_broadcast_success(job_id, job_type)
            
    # Operation management    
    async def load_operations(
        self,
        job_id: str,
        job_type: JobType,
        ops: List[Dict[str, str]] = []
    ):
        await self._handle_broadcast_start(job_id, job_type, {"ops": ops})
        for op_d in ops:
            await self.op_manager.load_operation(OpRoles(op_d.get('role', None)), op_d.get('id', None), op_d.get('config', dict()))
            await self._handle_broadcast_event(job_id, job_type, {
                "role": op_d.get('role', None), 
                "id": op_d.get('id', None),
                "loose_key": op_d.get("loose_key", None)
            })
        await self._handle_broadcast_success(job_id, job_type)
        
    async def load_operations_from_config(
        self,
        job_id: str,
        job_type: JobType,
    ):
        await self._handle_broadcast_start(job_id, job_type, {})
        await self.op_manager.load_operations_from_config()
        await self._handle_broadcast_success(job_id, job_type)
        
    async def unload_operations(
        self,
        job_id: str,
        job_type: JobType,
        ops: List[Dict[str, str]] = []
    ):
        await self._handle_broadcast_start(job_id, job_type, {"ops": ops})
        for op_d in ops:
            await self.op_manager.close_operation(OpRoles(op_d.get('role', None)), op_d.get('id', None))
            await self._handle_broadcast_event(job_id, job_type, {
                "role": op_d.get('role', None), 
                "id": op_d.get('id', None)
            })
        await self._handle_broadcast_success(job_id, job_type)
        
    async def configure_operations( # TODO document and add endpoint
        self,
        job_id: str,
        job_type: JobType,
        ops: List[Dict[str, str]] = []
    ):
        await self._handle_broadcast_start(job_id, job_type, {"ops": ops})
        for op_d in ops:
            await self.op_manager.configure(OpRoles(op_d.get('role', None)), op_d, op_id=op_d.get('id', None))
            await self._handle_broadcast_event(job_id, job_type, op_d)
        await self._handle_broadcast_success(job_id, job_type)
        
    async def use_operation(
        self,
        job_id: str,
        job_type: JobType,
        role: str = None,
        id: str = None,
        payload: Dict[str, Any] = None
    ):
        await self._handle_broadcast_start(job_id, job_type, {"role": role, "id": id})
        
        if 'audio_bytes' in payload:
            payload['audio_bytes'] = base64.b64decode(payload['audio_bytes'])

        if 'messages' in payload:
            msg_list = list()
            for msg in payload['messages']:
                assert 'type' in msg
                if msg['type'] == "raw":
                    msg_list.append(RawMessage(msg['message']))
                elif msg['type'] == "request":
                    msg_list.append(RequestMessage(msg['message'], datetime.datetime.fromtimestamp(msg['time'])))
                elif msg['type'] == "chat":
                    msg_list.append(ChatMessage(msg['user'], msg['message'], datetime.datetime.fromtimestamp(msg['time'])))
                elif msg['type'] == "tool":
                    msg_list.append(MCPMessage(msg['tool'], msg['message'], datetime.datetime.fromtimestamp(msg['time'])))
                elif msg['type'] == "custom":
                    msg_list.append(CustomMessage(msg['id'], msg['message'], datetime.datetime.fromtimestamp(msg['time'])))
                else:
                    raise Exception("Invalid message type")
            payload['messages'] = msg_list

        try:
            async for chunk_out in self.op_manager.use_operation(OpRoles(role), payload, op_id=id):
                await self._handle_broadcast_event(job_id, job_type, chunk_out)
        except OperationUnloaded:
            op = self.op_manager.loose_load_operation(OpRoles(role), id)
            await op.start()
            async for chunk_out in op(payload):
                if "audio_bytes" in chunk_out: chunk_out["audio_bytes"] = base64.b64encode(chunk_out['audio_bytes']).decode('utf-8')
                await self._handle_broadcast_event(job_id, job_type, chunk_out)
            await op.close()
            
        await self._handle_broadcast_success(job_id, job_type)
    
    # Configuration
    async def load_config(self, job_id: str, job_type: JobType, config_name: str):
        await self._handle_broadcast_start(job_id, job_type, {"config_name": config_name})
        Config().load_from_name(config_name)
        await self._handle_broadcast_success(job_id, job_type)
        
    async def update_config(self, job_id: str, job_type: JobType, config_d: dict = None, **kwargs):
        await self._handle_broadcast_start(job_id, job_type, {"config_d": config_d})
        if isinstance(config_d, dict):
            Config().load_from_dict(**config_d)
        elif kwargs:
            Config().load_from_dict(**kwargs)
        else:
            raise ValueError("update_config requires a config dict payload")
        await self._handle_broadcast_success(job_id, job_type)
    
    async def save_config(self, job_id: str, job_type: JobType, config_name: str):
        await self._handle_broadcast_start(job_id, job_type, {"config_name": config_name})
        Config().save(config_name)
        await self._handle_broadcast_success(job_id, job_type)
    
    ## General helpers ###############################
    async def _handle_broadcast_start(self, job_id: str, job_type: JobType, payload: dict):
        to_broadcast = {
            "job_id": job_id,
            "start": payload
        }
        logging.debug("Broadcasting start ({}) {} {:.500}".format(job_id, job_type.value, str(to_broadcast)))
        await self.event_server.broadcast_event(job_type.value, to_broadcast)
    
    async def _handle_broadcast_event(self, job_id: str, job_type: JobType, payload: dict):
        to_broadcast = {
            "job_id": job_id,
            "finished": False,
            "result": payload
        }
        logging.debug("Broadcasting event ({}) {} {:.500}".format(job_id, job_type.value, str(to_broadcast)))
        await self.event_server.broadcast_event(job_type.value, to_broadcast)
    
    async def _handle_broadcast_success(self, job_id: str, job_type: JobType, result: Dict[str, Any] = None):
        to_broadcast = {
            "job_id": job_id,
            "finished": True,
            "success": True
        }
        if result:
            to_broadcast["result"] = result
        logging.debug("Broadcasting success ({}) {} {}".format(job_id, job_type.value, str(to_broadcast)))
        await self.event_server.broadcast_event(job_type.value, to_broadcast)

    async def _handle_broadcast_cancelled(self, job_id: str, job_type: JobType, reason: str = "cancelled"):
        to_broadcast = {
            "job_id": job_id,
            "finished": True,
            "success": True,
            "result": {
                "event": "cancelled",
                "reason": reason,
                "partial": getattr(self, "_assistant_last_partial_reply", "")
            }
        }
        logging.debug("Broadcasting cancelled ({}) {} {}".format(job_id, job_type.value, str(to_broadcast)))
        await self.event_server.broadcast_event(job_type.value, to_broadcast)
        
    async def _handle_broadcast_error(self, job_id: str, job_type: JobType, err: Exception):
        # TODO: extend with all errors
        error_type = "unknown"
        if isinstance(err, UnknownOpType): error_type = "operation_unknown_type"
        if isinstance(err, UnknownOpRole): error_type = "operation_unknown_role"
        elif isinstance(err, UnknownOpID): error_type = "operation_unknown_id"
        elif isinstance(err, DuplicateFilter): error_type = "operation_duplicate"
        elif isinstance(err, OperationUnloaded): error_type = "operation_unloaded"
        elif isinstance(err, StartActiveError): error_type = "operation_active"
        elif isinstance(err, CloseInactiveError): error_type = "operation_inactive"
        elif isinstance(err, UsedInactiveError): error_type = "operation_inactive"
        elif isinstance(err, UnknownField): error_type = "config_unknown_field"
        elif isinstance(err, UnknownFile): error_type = "config_unknown_file"
        elif isinstance(err, UnknownJobType): error_type = "job_unknown"
        elif isinstance(err, asyncio.CancelledError): error_type = "job_cancelled"
        
        to_broadcast = {
            "job_id": job_id,
            "finished": True,
            "success": False,
            "result": {
                "type": error_type,
                "reason": str(err)
            }
        }
        
        logging.debug("Broadcasting error ({}) {} {}".format(job_id, job_type.value, str(to_broadcast)))
        await self.event_server.broadcast_event(job_type.value, to_broadcast)
