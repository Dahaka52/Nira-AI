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

from utils.helpers.singleton import Singleton
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
    
    async def start(self):
        logging.info("Starting JAIson application layer.")
        self.job_queue = asyncio.Queue()
        self.job_map = dict()
        self.job_skips = dict()
        self.job_loop = asyncio.create_task(self._process_job_loop())
        self.manager_loop = asyncio.create_task(self._process_manager_loop())
        
        self.event_server = ObserverServer()
        
        self.prompter = Prompter()
        await self.prompter.configure(Config().prompter)
        
        self.process_manager = ProcessManager()
        self.op_manager = OperationManager()
        self.mcp_manager = MCPManager()
        await self.mcp_manager.start()
        self.prompter.add_mcp_usage_prompt(self.mcp_manager.get_tooling_prompt(), self.mcp_manager.get_response_prompt())
        await self.op_manager.load_operations_from_config()
        await self.process_manager.reload()
        self._immediate_audio_lock = asyncio.Lock()
        
        # Start microphone if enabled
        from utils.processes.manager import ProcessType
        try:
            await self.process_manager.link("core_hw_mic", ProcessType.HW_MIC)
        except Exception as e:
            logging.error(f"Could not start HW_MIC process: {e}")
        # Start audio output if enabled
        try:
            await self.process_manager.link("core_hw_audio_out", ProcessType.HW_AUDIO_OUT)
        except Exception as e:
            logging.error(f"Could not start HW_AUDIO_OUT process: {e}")

        logging.info("JAIson application layer has started.")
        
    async def stop(self):
        logging.info("Shutting down JAIson application layer")
        for task in list(self._immediate_audio_tasks):
            task.cancel("shutdown")
        self._immediate_audio_tasks.clear()
        self._immediate_audio_pending.clear()
        self._immediate_audio_active = 0
        await self.op_manager.close_operation_all()
        await self.mcp_manager.close()
        from utils.processes.manager import ProcessType
        for link_id, process_type in (
            ("core_hw_mic", ProcessType.HW_MIC),
            ("core_hw_audio_out", ProcessType.HW_AUDIO_OUT),
        ):
            try:
                await self.process_manager.unlink(link_id, process_type)
            except Exception:
                pass
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
        if job_type_enum == JobType.RESPONSE: coro = self.response_pipeline(new_job_id, job_type_enum, **kwargs)
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

    def _interrupt_jobs(self, reason: str = "user_interruption"):
        """Экстренное прерывание: очистка очереди и текущей задачи"""
        logging.info(f"Smart Barge-in: Interrupting and clearing queue due to: {reason}")
        self._cancel_pending_voice_response()
        
        # 1. Очищаем очередь и связанные coroutine в job_map
        while True:
            try:
                queued_job_id = self.job_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            job_type_coro = self.job_map.pop(queued_job_id, None)
            self.job_skips.pop(queued_job_id, None)
            self._response_job_speakers.pop(queued_job_id, None)

            # Close queued coroutine to avoid "coroutine was never awaited" warnings
            if job_type_coro:
                _, coro = job_type_coro
                try:
                    coro.close()
                except Exception:
                    pass

            try:
                self.job_queue.task_done()
            except Exception:
                pass
        
        logging.debug(f"Interrupting current job for Barge-in")
        # 2. Прерываем текущую задачу через стандартный метод
        self._clear_current_job(reason=reason)

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
        should_respond = bool(turn.get("should_respond", True))
        continue_from_text = None
        if continue_intent:
            continue_from_text = (self._assistant_last_partial_reply or self._assistant_last_full_reply or "").strip()
            if continue_from_text:
                logging.info("Continue-intent detected: next response will continue previous thought.")

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

        response_job_id = await self.create_job(
            JobType.RESPONSE,
            input_timestamp=timestamp,
            input_mode="voice",
            continue_from_text=continue_from_text,
            source_id=source_id,
            turn_id=turn_id,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
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
    ):
        merge_window_ms = 2200
        try:
            mic_cfg = Config().microphone or {}
            merge_window_ms = int(mic_cfg.get("voice_merge_window_ms", 2200))
        except Exception:
            pass

        pending = self._pending_voice_turn
        if pending is None:
            self._pending_voice_turn = {
                "user": user,
                "timestamp": timestamp,
                "last_timestamp": timestamp,
                "content": content,
                "source_id": self._safe_source_id(source_id),
                "turn_id": str(turn_id or uuid.uuid4()),
                "utterance_ids": [str(utterance_id)] if utterance_id else [],
                "speaker_id": speaker_id,
                "stt_provider": stt_provider,
                "stt_confidence": stt_confidence,
                "stt_latency_ms": stt_latency_ms,
                "continue_intent": continue_intent,
                "should_respond": bool(should_respond),
            }
        else:
            same_user = pending.get("user") == user
            try:
                gap_ms = max(0.0, (float(timestamp) - float(pending.get("last_timestamp", timestamp))) * 1000.0)
            except Exception:
                gap_ms = merge_window_ms + 1

            if same_user and gap_ms <= merge_window_ms:
                prev = str(pending.get("content", "")).strip()
                cur = str(content).strip()
                pending["content"] = (prev + " " + cur).strip() if prev else cur
                pending["last_timestamp"] = timestamp
                pending["continue_intent"] = bool(pending.get("continue_intent", False) or continue_intent)
                pending["should_respond"] = bool(pending.get("should_respond", False) or should_respond)
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
                    "last_timestamp": timestamp,
                    "content": content,
                    "source_id": self._safe_source_id(source_id),
                    "turn_id": str(turn_id or uuid.uuid4()),
                    "utterance_ids": [str(utterance_id)] if utterance_id else [],
                    "speaker_id": speaker_id,
                    "stt_provider": stt_provider,
                    "stt_confidence": stt_confidence,
                    "stt_latency_ms": stt_latency_ms,
                    "continue_intent": continue_intent,
                    "should_respond": bool(should_respond),
                }

        self._schedule_voice_response()

    def _schedule_voice_response(self):
        self._pending_voice_response_seq += 1
        seq = self._pending_voice_response_seq
        self._cancel_pending_voice_response()

        debounce_ms = 300
        min_quiet_ms_after_speech_start = 350
        try:
            mic_cfg = Config().microphone or {}
            debounce_ms = int(mic_cfg.get("response_debounce_ms", 300))
            min_quiet_ms_after_speech_start = int(mic_cfg.get("response_min_quiet_ms_after_speech_start", 350))
        except Exception:
            pass

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
                                self._assistant_last_partial_reply = partial[-2000:]
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
        await self._handle_broadcast_start(job_id, job_type, {
            "include_audio": include_audio,
            "input_mode": input_mode,
            "source_id": source_id,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
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
        if continue_from_text and isinstance(continue_from_text, str) and continue_from_text.strip():
            tail = continue_from_text.strip()[-1200:]
            instruction_prompt = (
                f"{instruction_prompt}\n\n"
                "### Continuation Guidance ###\n"
                "Пользователь просит продолжить предыдущую мысль. "
                "Начни ответ с естественного продолжения прерванной фразы, затем мягко учти новый пользовательский ввод.\n"
                f"Незавершенный фрагмент: {tail}\n"
            )
        
        # Appy t2t
        t2t_result = ""
        t2t_start_time = 0
        async for chunk_out in self.op_manager.use_operation(OpRoles.T2T, {"instruction_prompt": instruction_prompt, "messages": history}):
            chunk_content = chunk_out.get('content', '')
            t2t_result += chunk_content
            if chunk_content:
                self._assistant_live_reply += chunk_content
            
            # Внедряем метрики для стриминга в чат
            token_count += len(chunk_content.split())
            if t2t_start_time == 0: t2t_start_time = time.time()
            elapsed = time.time() - t2t_start_time
            if latency == 0: latency = int((time.time() - start_time) * 1000)
            tps = round(token_count / elapsed, 1) if elapsed > 0 else 0

            # First-token metrics (for chat responsiveness / TTFT)
            if chunk_content and not first_token_sent:
                first_token_sent = True
                chunk_out["ttft_ms"] = latency
                if input_timestamp is not None:
                    try:
                        chunk_out["e2e_ttft_ms"] = int((time.time() - float(input_timestamp)) * 1000)
                    except Exception:
                        pass
            
            # Транслируем чанк немедленно для стриминга в чате
            chunk_out.update({"tps": tps, "latency": latency})
            await self._handle_broadcast_event(job_id, job_type, chunk_out)

        # Optional heavy debug events; disabled by default to keep WS traffic light.
        debug_prompt_events = False
        try:
            debug_prompt_events = bool(getattr(Config(), "broadcast_debug_prompt_events", False))
        except Exception:
            debug_prompt_events = False

        if debug_prompt_events:
            await self._handle_broadcast_event(job_id, job_type, {"instruction_prompt": instruction_prompt})
            await self._handle_broadcast_event(job_id, job_type, {"history": [msg.to_dict() for msg in history]})
            await self._handle_broadcast_event(job_id, job_type, {"raw_content": t2t_result})

        # Guard: FILTER_TEXT asserts on empty content, so finish gracefully.
        if not t2t_result or not t2t_result.strip():
            await self._handle_broadcast_event(job_id, job_type, {
                "event": "empty_response",
                "reason": "t2t_empty"
            })
            await self._handle_broadcast_success(job_id, job_type)
            logging.warning(f"Response job {job_id} produced empty T2T output. Finished without FILTER_TEXT/TTS.")
            if self._assistant_live_job_id == job_id:
                self._assistant_live_job_id = None
                self._assistant_live_reply = ""
            return

        # Apply text filters and TTS streaming
        full_filtered_text = ""
        # Мы используем генератор фильтров как основной поток для TTS
        async for text_chunk_out in self.op_manager.use_operation(OpRoles.FILTER_TEXT, {"content": t2t_result}):
            chunk_content = text_chunk_out.get('content', '')
            if chunk_content:
                # [FIX] Склеиваем полный текст для истории
                if full_filtered_text and not full_filtered_text.endswith((" ", "\n")):
                    full_filtered_text += " "
                full_filtered_text += chunk_content
                
                # [STREAM] Шлем чанк на TTS немедленно
                if include_audio:
                    async for audio_chunk_out in self.op_manager.use_operation(OpRoles.TTS, text_chunk_out):
                        # Apply tts filters
                        async for final_audio_chunk_out in self.op_manager.use_operation(OpRoles.FILTER_AUDIO, audio_chunk_out):
                            for metric_key in (
                                "tts_provider_latency_ms",
                                "tts_rtf",
                                "tts_audio_s",
                                "tts_total_ms",
                                "tts_first_chunk_ms",
                                "tts_max_gap_ms",
                                "tts_chunks",
                                "tts_text_len",
                            ):
                                if metric_key in audio_chunk_out and metric_key not in final_audio_chunk_out:
                                    final_audio_chunk_out[metric_key] = audio_chunk_out[metric_key]
                            # Broadcast results (single WS event per PCM chunk).
                            audio_event = {
                                "audio_bytes": base64.b64encode(final_audio_chunk_out["audio_bytes"]).decode("utf-8"),
                                "sr": final_audio_chunk_out["sr"],
                                "sw": final_audio_chunk_out["sw"],
                                "ch": final_audio_chunk_out["ch"],
                                "event": "audio_chunk",
                            }
                            for metric_key in (
                                "tts_provider_latency_ms",
                                "tts_rtf",
                                "tts_audio_s",
                                "tts_total_ms",
                                "tts_first_chunk_ms",
                                "tts_max_gap_ms",
                                "tts_chunks",
                                "tts_text_len",
                            ):
                                if metric_key in final_audio_chunk_out:
                                    audio_event[metric_key] = final_audio_chunk_out[metric_key]
                            # First-audio metrics (time to first playable TTS chunk)
                            if not first_audio_sent:
                                first_audio_sent = True
                                tts_start_ms = int((time.time() - start_time) * 1000)
                                audio_event["tts_start_ms"] = tts_start_ms
                                if input_timestamp is not None:
                                    try:
                                        audio_event["e2e_tts_start_ms"] = int((time.time() - float(input_timestamp)) * 1000)
                                    except Exception:
                                        pass
                            await self._handle_broadcast_event(job_id, job_type, audio_event)
        
        if full_filtered_text:
            self.prompter.add_chat(self.prompter.character_name, full_filtered_text)
            self._assistant_last_full_reply = full_filtered_text[-2000:]
            self._assistant_last_partial_reply = ""

        if self._assistant_live_job_id == job_id:
            self._assistant_live_job_id = None
            self._assistant_live_reply = ""
                        
        # Broadcast completion
        await self._handle_broadcast_success(job_id, job_type)
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
        self.prompter.add_chat(
            user,
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
        source_id = self._safe_source_id(request_data.get("source_id"))
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

        if not content or len(content.strip()) == 0:
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
        fillers = {"угу", "ага", "понятно", "ясно", "да", "так", "хорошо", "ок", "слышу", "мгм", "ладно", "понял", "ого", "ммм", "эмм", "хмм", "интересно"}
        stop_words = {"стой", "стоп", "хватит", "замолчи", "подожди", "тихо", "молчи", "выключи"}
        stop_stems = ("стоп", "стой", "подож", "хват", "замолч", "тихо", "молч", "выключ")
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
        greeting_words = {
            "привет", "здравствуй", "здравствуйте", "хай", "хелло", "hello", "hey",
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
        try:
            barge_in_min_non_filler_words = int(mic_cfg.get("barge_in_min_non_filler_words", 2))
        except Exception:
            barge_in_min_non_filler_words = 2
        if barge_in_min_non_filler_words < 1:
            barge_in_min_non_filler_words = 1

        continue_intent = self._is_continue_intent(content)

        if not words:
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

        def _is_stop_like(word: str) -> bool:
            w = word.strip().lower()
            if not w:
                return False
            if w in stop_words:
                return True
            if w == "сто":
                return True
            return any(w.startswith(stem) for stem in stop_stems)

        stop_like_count = sum(1 for w in words if _is_stop_like(w))
        contains_stop_word = stop_like_count > 0
        non_filler_words = [w for w in words if w not in fillers]
        non_stop_words = [w for w in non_filler_words if not _is_stop_like(w)]
        wake_word_hit = any(w in wake_words for w in non_filler_words)
        is_wake_word_only = wake_word_hit and len(non_filler_words) == 1 and not contains_stop_word

        def _is_laughter_like(word: str) -> bool:
            w = word.strip().lower().replace("-", "")
            if len(w) < 2:
                return False
            return bool(re.fullmatch(r"(ха|ах|хе|ех){2,}", w))

        short_emote_hit = respond_to_short_emotes and any(
            (_is_laughter_like(w) or (w in short_emote_words)) for w in words
        )
        greeting_hit = len(non_filler_words) == 1 and any(w in greeting_words for w in non_filler_words)

        is_backchannel = len(words) <= 2 and len(non_filler_words) == 0
        is_stop_command_only = contains_stop_word and (
            len(non_stop_words) == 0 or stop_like_count >= max(1, len(words) - 1)
        )
        is_significant = (
            contains_stop_word
            or continue_intent
            or len(non_filler_words) >= barge_in_min_non_filler_words
        )
        should_respond = (
            continue_intent
            or len(non_filler_words) >= 2
            or is_wake_word_only
            or short_emote_hit
            or greeting_hit
        ) and not is_stop_command_only

        if is_backchannel and not continue_intent and not short_emote_hit:
            logging.info("Backchannel detected (no response): '%s'", content)
            await self._record_stt_metrics(
                source_id=source_id,
                turn_id=turn_id,
                utterance_id=utterance_id,
                provider=stt_provider,
                latency_ms=stt_latency_ms,
                text=content,
                detected_stop_cmd=contains_stop_word,
                expected_stop_cmd=bool(expected_stop_cmd) if isinstance(expected_stop_cmd, bool) else None,
            )
            return

        if is_wake_word_only:
            content = canonical_wake_word

        if is_significant:
            if self._interrupt_allowed_for_speaker(speaker_id):
                self._interrupt_jobs(reason="user_speaking_significant")
                asyncio.create_task(self._handle_broadcast_event("GLOBAL_STOP", JobType.RESPONSE, {
                    "event": "stop_audio",
                    "reason": "user_speaking_significant",
                    "source_id": source_id,
                    "turn_id": turn_id,
                    "utterance_id": utterance_id,
                    "speaker_id": speaker_id,
                }))
            else:
                await self._emit_stt_status(
                    "interrupt_ignored",
                    reason="speaker_policy",
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
            detected_stop_cmd=contains_stop_word,
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
        )

    async def on_user_speech_start(self, request_data: Dict[str, Any] | None = None):
        """Early barge-in signal from VAD start (before STT final transcript)."""
        request_data = request_data or {}
        self._last_speech_start_ts = time.time()
        self._cancel_pending_voice_response()
        source_id = self._safe_source_id(request_data.get("source_id"))
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

        if self.job_current is None or self.job_current.done():
            return

        current_job_type, _ = self.job_map.get(self.job_current_id, (None, None))
        if current_job_type != JobType.RESPONSE:
            return

        if not self._interrupt_allowed_for_speaker(speaker_id):
            return

        self._interrupt_jobs(reason="user_voice_start")
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
    
    async def _handle_broadcast_success(self, job_id: str, job_type: JobType):
        to_broadcast = {
            "job_id": job_id,
            "finished": True,
            "success": True
        }
        logging.debug("Broadcasting success ({}) {} {}".format(job_id, job_type.value, str(to_broadcast)))
        await self.event_server.broadcast_event(job_type.value, to_broadcast)

    async def _handle_broadcast_cancelled(self, job_id: str, job_type: JobType, reason: str = "cancelled"):
        to_broadcast = {
            "job_id": job_id,
            "finished": True,
            "success": True,
            "result": {
                "event": "cancelled",
                "reason": reason
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
