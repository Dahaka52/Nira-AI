from quart import Quart, request, websocket
import asyncio
import json
import base64
import logging
import psutil
import subprocess
import os
import time
from datetime import datetime
from utils.args import args
from utils.helpers.singleton import Singleton
from utils.jaison import JAIson, JobType, NonexistantJobException
from utils.config import Config
from utils.helpers.observer import BaseObserverClient
from .common import create_response, create_preflight

app = Quart(__name__)
cors_header = {'Access-Control-Allow-Origin': '*'}
_gpu_cache = {"ts": 0.0, "gpus": []}


def _read_gpus_sync():
    gpus = []
    res = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        encoding='utf-8'
    )
    for line in res.strip().split('\n'):
        if not line:
            continue
        idx, name, util, mem_used, mem_total, temp = line.split(', ')
        gpus.append({
            "id": int(idx),
            "name": name,
            "load": int(util),
            "mem_used": int(mem_used),
            "mem_total": int(mem_total),
            "temp": int(temp)
        })
    return gpus

## Websocket Event Broadcasting Server ##

class SocketServerObserver(BaseObserverClient, metaclass=Singleton):
    def __init__(self):
        super().__init__(server=JAIson().event_server)
        self.connections = set()
        self.shutdown_signal = asyncio.Future()

    async def handle_event(self, event_id: str, payload) -> None:
        '''Broadcast events from broadcast server'''
        for key in payload:
            if isinstance(payload[key], bytes):
                  payload[key] = base64.b64encode(payload[key]).decode('utf-8')
        message = json.dumps(create_response(200, event_id, payload))
        logging.debug(f"Broadcasting event to {len(self.connections)} clients")
        dead_connections = set()
        for ws in set(self.connections):
            try:
                await ws.send(message)
            except Exception:
                dead_connections.add(ws)
        for ws in dead_connections:
            self.connections.discard(ws)
            
    def shutdown(self, *args): # TODO set for use somewhere
        self.shutdown_signal.set_result(None)
        
@app.websocket("/")
async def ws():
    sso = SocketServerObserver()
    logging.info("Opened new websocket connection")
    ws = websocket._get_current_object()
    await ws.accept()
    sso.connections.add(ws)
    try:
        while not sso.shutdown_signal.done():
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        sso.connections.discard(ws)
        logging.info("Closed websocket connection")
        raise

## Generic endpoints ###################

@app.route('/api/operations', methods=['GET'])
async def get_loaded_operations():
    return create_response(200, f"Loaded operations gotten", JAIson().get_loaded_operations(), cors_header)
  
@app.route('/api/context/history', methods=['GET'])
async def get_history():
    from utils.prompter import Prompter
    history = Prompter().get_history()
    return create_response(200, "History retrieved", [m.to_dict() for m in history], cors_header)

@app.route('/api/context/history', methods=['OPTIONS'])
async def preflight_history():
    return create_preflight('GET')

@app.route('/api/config', methods=['GET'])
async def get_current_config():
    return create_response(200, f"Current config gotten", JAIson().get_current_config(), cors_header)

@app.route('/api/pipeline', methods=['GET'])
async def get_pipeline_stats():
    j = JAIson()
    queue_size = j.job_queue.qsize() if j.job_queue else 0
    
    # System stats
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    
    gpus = _gpu_cache["gpus"]
    now = time.time()
    if now - _gpu_cache["ts"] > 1.5:
        try:
            gpus = await asyncio.to_thread(_read_gpus_sync)
            _gpu_cache["gpus"] = gpus
            _gpu_cache["ts"] = now
        except Exception:
            pass

    telemetry_data = j.get_pipeline_telemetry()

    return create_response(200, "Pipeline stats retrieved", {
        "current_job_id": j.job_current_id,
        "queue_size": queue_size,
        "status": "active" if j.job_current_id else "idle",
        "stt": j.get_stt_runtime_stats(),
        "loaded_operations": j.get_loaded_operations(),
        "active_providers": telemetry_data.get("active_providers"),
        "audio_output_mode": telemetry_data.get("audio_output_mode", "discord"),
        "telemetry": telemetry_data.get("latest"),
        "telemetry_history": telemetry_data.get("history", []),
        "discord_bridge": telemetry_data.get("discord"),
        "system": {
            "cpu": cpu,
            "ram": ram,
            "gpus": gpus
        }
    }, cors_header)

@app.route('/api/pipeline', methods=['OPTIONS'])
async def preflight_pipeline():
    return create_preflight('GET')

@app.route('/api/pipeline/telemetry', methods=['DELETE'])
async def clear_pipeline_telemetry():
    JAIson().clear_telemetry_history()
    return create_response(200, "Telemetry history cleared", {"ok": True}, cors_header)

@app.route('/api/pipeline/telemetry', methods=['OPTIONS'])
async def preflight_clear_pipeline_telemetry():
    return create_preflight('DELETE')

@app.route('/api/output/mode', methods=['GET'])
async def get_output_mode():
    return create_response(200, "Output mode retrieved", {"mode": JAIson().get_audio_output_mode()}, cors_header)

@app.route('/api/output/mode', methods=['POST', 'PUT'])
async def set_output_mode():
    try:
        data = (await request.get_json(silent=True)) or {}
        mode = data.get("mode", "discord")
        res = await JAIson().set_audio_output_mode(mode)
        return create_response(200, "Output mode updated", res, cors_header)
    except Exception as err:
        return create_response(500, str(err), {}, cors_header)

@app.route('/api/output/mode', methods=['OPTIONS'])
async def preflight_output_mode():
    return create_preflight('GET, POST, PUT')

@app.route('/api/bridge/discord/status', methods=['POST'])
async def set_discord_bridge_status():
    try:
        data = (await request.get_json()) or {}
        JAIson().set_discord_bridge_status(data)
        return create_response(200, "Discord bridge status updated", {"ok": True}, cors_header)
    except Exception as err:
        return create_response(500, str(err), {}, cors_header)

@app.route('/api/bridge/discord/status', methods=['GET'])
async def get_discord_bridge_status():
    return create_response(200, "Discord bridge status retrieved", JAIson().get_discord_bridge_status(), cors_header)

@app.route('/api/bridge/discord/status', methods=['OPTIONS'])
async def preflight_discord_bridge_status():
    return create_preflight('POST, GET')

@app.route('/api/health', methods=['GET'])
async def health_check():
    return create_response(200, "ok", {}, cors_header)

@app.route('/api/health', methods=['OPTIONS'])
async def preflight_health():
    return create_preflight('GET')

## Job management endpoints ###########
@app.route('/api/job', methods=['DELETE'])
async def cancel_job():
    try:
        request_data = await request.get_json()
        assert 'job_id' in request_data
        return create_response(200, f"Job flagged for cancellation", await JAIson().cancel_job(request_data['job_id'], request_data.get('reason')), cors_header)
    except NonexistantJobException as err:
        return create_response(400, f"Job ID does not exist or already finished", {}, cors_header)
    except AssertionError as err:
        return create_response(400, f"Request missing job_id", {}, cors_header)
    except Exception as err:
        return create_response(500, str(err), {}, cors_header)

## Specific job creation endpoints ####

async def _request_job(job_type: JobType):
    try:
        request_data = (await request.get_json()) or dict()
        job_id = await JAIson().create_job(job_type, **request_data)
        return create_response(200, f"{job_type} job created", {"job_id": job_id}, cors_header)
    except Exception as err:
        logging.error(f"Error occured for {job_type} API request", stack_info=True, exc_info=True)
        return create_response(500, str(err), {}, cors_header)

# Main response pipeline
@app.route('/api/response', methods=['POST'])
async def response():
    return await _request_job(JobType.RESPONSE)

# Context - General
@app.route('/api/context', methods=['DELETE'])    
async def context_clear():
    return await _request_job(JobType.CONTEXT_CLEAR)

# Context - Configure
@app.route('/api/context/config', methods=['PUT'])    
async def context_configure():
    return await _request_job(JobType.CONTEXT_CONFIGURE)

# Context - Requests
@app.route('/api/context/request', methods=['POST'])    
async def context_request_add():
    return await _request_job(JobType.CONTEXT_REQUEST_ADD)

# Context - Conversation
@app.route('/api/context/conversation/text', methods=['POST'])    
async def context_conversation_add_text():
    return await _request_job(JobType.CONTEXT_CONVERSATION_ADD_TEXT)

@app.route('/api/context/conversation/audio', methods=['POST'])    
async def context_conversation_add_audio():
    try:
        request_data = (await request.get_json(silent=True)) or dict()
        result = await JAIson().submit_audio_immediate(request_data)
        if not result.get("accepted", False):
            return create_response(200, "Audio dropped due to STT backpressure", result, cors_header)
        return create_response(200, "Audio processing started", result, cors_header)
    except Exception as err:
        logging.error(f"Error occured for audio API request", stack_info=True, exc_info=True)
        return create_response(500, str(err), {}, cors_header)

@app.route('/api/context/conversation/speech_start', methods=['POST'])
async def context_conversation_speech_start():
    try:
        request_data = (await request.get_json(silent=True)) or dict()
        # lightweight path: handle immediately to avoid piling background tasks
        await JAIson().on_user_speech_start(request_data)
        return create_response(200, "Speech start signal processed", {}, cors_header)
    except Exception as err:
        logging.error(f"Error occured for speech_start API request", stack_info=True, exc_info=True)
        return create_response(500, str(err), {}, cors_header)

# Context - Custom
@app.route('/api/context/custom', methods=['PUT'])    
async def context_custom_register():
    return await _request_job(JobType.CONTEXT_CUSTOM_REGISTER)

@app.route('/api/context/custom', methods=['DELETE'])    
async def context_custom_remove():
    return await _request_job(JobType.CONTEXT_CUSTOM_REMOVE)

@app.route('/api/context/custom', methods=['POST'])    
async def context_custom_add():
    return await _request_job(JobType.CONTEXT_CUSTOM_ADD)

# Operation management
@app.route('/api/operations/load', methods=['POST'])    
async def operation_start():
    return await _request_job(JobType.OPERATION_LOAD)

@app.route('/api/operations/reload', methods=['POST'])    
async def operation_reload():
    return await _request_job(JobType.OPERATION_CONFIG_RELOAD)

@app.route('/api/operations/unload', methods=['POST'])    
async def operation_unload():
    return await _request_job(JobType.OPERATION_UNLOAD)

@app.route('/api/operations/config', methods=['POST'])    
async def operation_configure():
    return await _request_job(JobType.OPERATION_CONFIGURE)

@app.route('/api/operations/use', methods=['POST'])    
async def operation_use():
    return await _request_job(JobType.OPERATION_USE)

# Configuration
@app.route('/api/config/load', methods=['PUT'])    
async def config_load():
    return await _request_job(JobType.CONFIG_LOAD)

# Configuration
@app.route('/api/config/update', methods=['PUT'])    
async def config_update():
    return await _request_job(JobType.CONFIG_UPDATE)

@app.route('/api/config/save', methods=['POST'])    
async def config_save():
    return await _request_job(JobType.CONFIG_SAVE)

# Allow CORS
@app.route('/api/job', methods=['OPTIONS']) 
async def preflight_job():
    return create_preflight('DELETE')

@app.route('/api/response', methods=['OPTIONS']) 
async def preflight_response():
    return create_preflight('POST')

@app.route('/api/context', methods=['OPTIONS']) 
async def preflight_context_conversation_clear():
    return create_preflight('DELETE')

@app.route('/api/context/config', methods=['OPTIONS'])    
async def preflight_context_configure():
    return create_preflight('PUT')

@app.route('/api/context/request', methods=['OPTIONS']) 
async def preflight_context_request():
    return create_preflight('POST')

@app.route('/api/context/conversation/text', methods=['OPTIONS']) 
async def preflight_context_conversation_text():
    return create_preflight('POST')

@app.route('/api/context/conversation/audio', methods=['OPTIONS']) 
async def preflight_context_conversation_audio():
    return create_preflight('POST')

@app.route('/api/context/conversation/speech_start', methods=['OPTIONS'])
async def preflight_context_conversation_speech_start():
    return create_preflight('POST')

@app.route('/api/context/custom', methods=['OPTIONS']) 
async def preflight_context_custom():
    return create_preflight('POST, PUT, DELETE')

@app.route('/api/operations', methods=['OPTIONS']) 
async def preflight_operations_info():
    return create_preflight('GET')

@app.route('/api/operations/load', methods=['OPTIONS']) 
async def preflight_operation_start():
    return create_preflight('POST')

@app.route('/api/operations/reload', methods=['OPTIONS']) 
async def preflight_operation_reload():
    return create_preflight('POST')

@app.route('/api/operations/unload', methods=['OPTIONS']) 
async def preflight_operation_unload():
    return create_preflight('POST')

@app.route('/api/operations/config', methods=['OPTIONS'])    
async def preflight_operation_configure():
    return create_preflight('POST')

@app.route('/api/operations/use', methods=['OPTIONS'])
async def preflight_operation_use():
    return create_preflight('POST')

@app.route('/api/config', methods=['OPTIONS']) 
async def preflight_config():
    return create_preflight('GET')

@app.route('/api/config/load', methods=['OPTIONS']) 
async def preflight_config_load():
    return create_preflight('PUT')


@app.route('/api/config/save', methods=['OPTIONS']) 
async def preflight_config_save():
    return create_preflight('POST')

## START ###################################
async def start_web_server(): # TODO launch application plugins here as well
    try:
        global app
        await JAIson().start()
        SocketServerObserver()
        from hypercorn.config import Config as HyperConfig
        from hypercorn.asyncio import serve

        hyper_cfg = HyperConfig()
        hyper_cfg.bind = [f"{args.host}:{args.port}"]
        hyper_cfg.accesslog = None  # Скрываем регулярный access-лог (GET /api/pipeline, POST /api/bridge/discord/status)
        hyper_cfg.errorlog = logging.getLogger("hypercorn.error")
        await serve(app, hyper_cfg)
    except Exception as err:
        logging.error("Stopping server due to exception", exc_info=True)
    finally:    
        await JAIson().stop()
