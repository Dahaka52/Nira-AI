import logging
import asyncio
import uuid
import base64
import datetime
import time
import os
import re
from typing import Dict, Coroutine, List, Any, Tuple
from enum import Enum
from utils.args import args

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
        
        # Start microphone if enabled
        from utils.processes.manager import ProcessType
        try:
            await self.process_manager.link("core_hw_mic", ProcessType.HW_MIC)
        except Exception as e:
            logging.error(f"Could not start HW_MIC process: {e}")

        logging.info("JAIson application layer has started.")
        
    async def stop(self):
        logging.info("Shutting down JAIson application layer")
        await self.op_manager.close_operation_all()
        await self.mcp_manager.close()
        await self.process_manager.unload()
        logging.info("JAIson application layer has been shut down")
    
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
        continue_intent = bool(turn.get("continue_intent", False))
        should_respond = bool(turn.get("should_respond", True))
        continue_from_text = None
        if continue_intent:
            continue_from_text = (self._assistant_last_partial_reply or self._assistant_last_full_reply or "").strip()
            if continue_from_text:
                logging.info("Continue-intent detected: next response will continue previous thought.")

        await self.create_job(JobType.CONTEXT_CONVERSATION_ADD_TEXT, user=user, content=content, timestamp=timestamp)
        if not should_respond:
            logging.info("Voice turn committed as context-only (no RESPONSE job).")
            return

        await self.create_job(
            JobType.RESPONSE,
            input_timestamp=timestamp,
            input_mode="voice",
            continue_from_text=continue_from_text
        )

    async def _buffer_voice_turn(self, user: str, timestamp: float, content: str, continue_intent: bool, should_respond: bool):
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
            else:
                await self._commit_pending_voice_turn()
                self._pending_voice_turn = {
                    "user": user,
                    "timestamp": timestamp,
                    "last_timestamp": timestamp,
                    "content": content,
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
        await self._handle_broadcast_start(job_id, job_type, {"include_audio": include_audio})
    
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
                            # Broadcast results (only the audio data for now)
                            for ws_chunk in chunk_buffer(base64.b64encode(final_audio_chunk_out['audio_bytes']).decode('utf-8')):
                                audio_event = {
                                    "audio_bytes": ws_chunk,
                                    "sr": final_audio_chunk_out['sr'],
                                    "sw": final_audio_chunk_out['sw'],
                                    "ch": final_audio_chunk_out['ch'],
                                    "event": "audio_chunk"
                                }
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
        content: str = None
    ):
        await self._handle_broadcast_start(job_id, job_type, {"user": user, "timestamp": timestamp, "content": content})
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
            "line": last_line_o.to_line()
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
        ch: int = None
    ):
        # Legacy job path kept for compatibility.
        # Delegate to the main immediate audio pipeline so behavior stays identical
        # between REST `/api/context/conversation/audio` and queued job flow.
        await self._handle_broadcast_start(
            job_id,
            job_type,
            {"user": user, "timestamp": timestamp, "sr": sr, "sw": sw, "ch": ch, "audio_bytes": (audio_bytes is not None)}
        )
        await self.process_audio_immediate({
            "user": user,
            "timestamp": timestamp,
            "audio_bytes": audio_bytes,
            "sr": sr,
            "sw": sw,
            "ch": ch,
        })
        await self._handle_broadcast_success(job_id, job_type)
            
    async def process_audio_immediate(self, request_data: dict):
        """Немедленная обработка аудио (вне очереди) для мгновенного перебивания"""
        audio_bytes_b64 = request_data.get('audio_bytes')
        if not audio_bytes_b64: return
        
        audio_bytes = base64.b64decode(audio_bytes_b64)
        sr = request_data.get('sr', 16000)
        sw = request_data.get('sw', 2)
        ch = request_data.get('ch', 1)
        user = request_data.get('user', 'user')
        timestamp = request_data.get('timestamp', time.time())

        # 1. Выполняем STT немедленно
        prompt = self.prompter.get_history_text() or ""
        content = ""
        try:
            async for out_d in self.op_manager.use_operation(OpRoles.STT, {"prompt": prompt, "audio_bytes": audio_bytes, "sr": sr, "sw": sw, "ch": ch}):
                content += out_d.get('transcription', '')
        except Exception as e:
            logging.error(f"Immediate STT failed: {e}")
            return

        if not content or len(content.strip()) == 0:
            return

        # 2. Проверяем Barge-in (Перебивание) + классифицируем реплику
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
        try:
            mic_cfg = Config().microphone or {}
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
            prompter_cfg = Config().prompter or {}
            cfg_name = str(prompter_cfg.get("character_name", "")).strip().lower()
            if cfg_name:
                canonical_wake_word = cfg_name
                wake_words.add(cfg_name)
        except Exception:
            pass
        continue_intent = self._is_continue_intent(content)

        if not words:
            return

        def _is_stop_like(word: str) -> bool:
            w = word.strip().lower()
            if not w:
                return False
            if w in stop_words:
                return True
            # tolerate common short/asr-truncated variants, e.g. "сто"
            if w in {"сто", "стоП"}:
                return True
            return any(w.startswith(stem) for stem in stop_stems)

        stop_like_count = sum(1 for w in words if _is_stop_like(w))
        contains_stop_word = stop_like_count > 0
        non_filler_words = [w for w in words if w not in fillers]
        non_stop_words = [w for w in non_filler_words if not _is_stop_like(w)]
        wake_word_hit = any(w in wake_words for w in non_filler_words)
        is_wake_word_only = wake_word_hit and len(non_filler_words) == 1 and not contains_stop_word

        def _is_laughter_like(word: str) -> bool:
            # Accept noisy laugh patterns: ха-ха, хаха, ахах, etc.
            w = word.strip().lower().replace("-", "")
            if len(w) < 2:
                return False
            return bool(re.fullmatch(r"(ха|ах|хе|ех){2,}", w))

        short_emote_hit = respond_to_short_emotes and any(
            (_is_laughter_like(w) or (w in short_emote_words)) for w in words
        )

        is_backchannel = len(words) <= 2 and len(non_filler_words) == 0
        # "stop command" even with 1 garbage STT token (e.g. "стоп стоп стой сто")
        is_stop_command_only = contains_stop_word and (
            len(non_stop_words) == 0 or stop_like_count >= max(1, len(words) - 1)
        )
        # Реагируем на содержательные фразы (>=2 не-филлер токенов) или явный continue-intent.
        is_significant = contains_stop_word or continue_intent or len(non_filler_words) >= 2
        # Single wake-word ("нира") should produce a response instead of context-only.
        should_respond = (
            continue_intent
            or len(non_filler_words) >= 2
            or is_wake_word_only
            or short_emote_hit
        ) and not is_stop_command_only

        if is_backchannel and not continue_intent and not short_emote_hit:
            logging.info(f"Backchannel detected (no response): '{content}'")
            return

        # Normalize single-word wake aliases (e.g., "миром") into canonical name
        # so context is cleaner and wake behavior stays predictable.
        if is_wake_word_only:
            content = canonical_wake_word

        if is_significant:
            self._interrupt_jobs(reason="user_speaking_significant")

            # Шлем сигнал STOP в UI для очистки статуса Thinking
            asyncio.create_task(self._handle_broadcast_event("GLOBAL_STOP", JobType.RESPONSE, {"event": "stop_audio", "reason": "user_speaking_significant"}))

        # 3. Буферизуем голосовые сегменты в один пользовательский ход и отвечаем после quiet-window
        await self._buffer_voice_turn(
            user=user,
            timestamp=timestamp,
            content=content,
            continue_intent=continue_intent,
            should_respond=should_respond
        )

    async def on_user_speech_start(self):
        """Early barge-in signal from VAD start (before STT final transcript)."""
        self._last_speech_start_ts = time.time()
        self._cancel_pending_voice_response()

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

        self._interrupt_jobs(reason="user_voice_start")
        await self._handle_broadcast_event("GLOBAL_STOP", JobType.RESPONSE, {
            "event": "stop_audio",
            "reason": "user_voice_start"
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
