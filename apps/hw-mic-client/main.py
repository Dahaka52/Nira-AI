import os
import time
import json
import base64
import argparse
import uuid
import numpy as np
import onnxruntime
import requests
import sys
import threading
from collections import deque
from typing import Optional
import sounddevice as sd
from scipy import signal  # Для качественного ресемплинга

# ==============================================================
# КОНФИГУРАЦИЯ
# ==============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--device_index", type=int, default=None, help="Audio device index")
parser.add_argument("--device_name", type=str, default=None, help="Input device name substring (preferred over index)")
parser.add_argument("--device_hostapi", type=str, default=None, help="Optional host API filter (e.g. WASAPI)")
parser.add_argument("--list_devices", action="store_true", help="List input devices and exit")
parser.add_argument("--nira_api", "--jaison_api", dest="nira_api", type=str, default="http://localhost:7272/api/context/conversation/audio", help="Nira API URL")
parser.add_argument("--speech_start_api", type=str, default=None, help="Optional early-barge-in URL (default: derived from nira_api)")
parser.add_argument("--speech_start_min_interval_ms", type=int, default=350, help="Minimum interval between speech_start signals")
parser.add_argument("--speech_start_confirm_ms", type=int, default=100, help="Require this much active speech before sending speech_start")
parser.add_argument("--min_speech_ms_interrupt", type=int, default=80, help="Minimum ms for short interrupt commands to still be sent")
parser.add_argument("--source_id", type=str, default="mic", help="Audio source identifier")
parser.add_argument("--user", type=str, default="Вова", help="User name for mic input")
parser.add_argument("--turn_merge_window_ms", type=int, default=2200, help="Reuse previous turn_id if next phrase starts within this window")
parser.add_argument("--resample_mode", type=str, default="polyphase", choices=["polyphase", "decimate"], help="Resampling mode 48k->16k")
parser.add_argument("--mic_gain_db", type=float, default=12.0, help="Fixed software mic gain in dB")
parser.add_argument("--agc_enable", type=int, default=1, help="Enable simple AGC (1/0)")
parser.add_argument("--agc_target_rms", type=float, default=0.05, help="Target RMS for AGC")
parser.add_argument("--agc_max_gain_db", type=float, default=15.0, help="Maximum AGC gain in dB")
parser.add_argument("--soft_limit", type=float, default=0.97, help="Soft limiter bound")
parser.add_argument("--vad_threshold", type=float, default=0.2, help="Probability threshold for VAD")
parser.add_argument("--vad_start_rms", type=float, default=0.012, help="RMS trigger threshold to start phrase")
parser.add_argument("--vad_hold_rms", type=float, default=0.008, help="RMS hold threshold to continue phrase")
parser.add_argument("--vad_floor_rms", type=float, default=0.001, help="Minimum RMS for running VAD model")
parser.add_argument("--rms_bridge_ms", type=int, default=96, help="How long low-energy gap is tolerated before silence handling")
parser.add_argument("--min_silence_ms", type=int, default=500, help="Milliseconds of silence to split phrase")
parser.add_argument("--min_speech_ms", type=int, default=200, help="Minimum ms of speech to send")
parser.add_argument("--pre_roll_ms", type=int, default=300, help="Milliseconds of audio to keep before speech")
parser.add_argument("--energy_threshold", type=float, default=0.01, help="RMS threshold (gate)")
parser.add_argument("--ws_url", type=str, default="ws://localhost:7272/", help="Nira WebSocket URL for dynamic control")
args = parser.parse_args()


def resolve_speech_start_url(audio_url: str, explicit_url: Optional[str]) -> str:
    if explicit_url:
        return explicit_url
    if audio_url.endswith("/audio"):
        return audio_url[:-len("/audio")] + "/speech_start"
    return audio_url.rstrip("/") + "/speech_start"


SPEECH_START_API = resolve_speech_start_url(args.nira_api, args.speech_start_api)
_last_speech_start_ts_ms = 0.0
_last_turn_id: Optional[str] = None
_last_turn_ts: float = 0.0
_turn_lock = threading.Lock()


def _get_hostapi_name(hostapi_index: int) -> str:
    try:
        hostapis = sd.query_hostapis()
        if 0 <= hostapi_index < len(hostapis):
            return str(hostapis[hostapi_index]["name"])
    except Exception:
        pass
    return "unknown"


def get_input_devices() -> list:
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        hostapi_idx = int(dev.get("hostapi", -1))
        devices.append({
            "index": idx,
            "name": str(dev.get("name", "")),
            "hostapi": _get_hostapi_name(hostapi_idx),
            "max_input_channels": int(dev.get("max_input_channels", 0)),
            "default_samplerate": int(dev.get("default_samplerate", 0)),
        })
    return devices


def print_input_devices() -> None:
    input_devices = get_input_devices()
    if not input_devices:
        print("[MIC] Input devices not found.")
        return

    print("[MIC] Available input devices:")
    for dev in input_devices:
        print(
            f"  [{dev['index']}] {dev['name']} | hostapi={dev['hostapi']} | "
            f"channels={dev['max_input_channels']} | default_sr={dev['default_samplerate']}"
        )


def resolve_input_device_index(
    preferred_index: Optional[int],
    preferred_name: Optional[str],
    preferred_hostapi: Optional[str],
) -> int:
    input_devices = get_input_devices()
    if not input_devices:
        raise RuntimeError("No input devices available")

    # 1) Name match (stable across index shuffles)
    if preferred_name:
        needle = preferred_name.strip().lower()
        hostapi_needle = (preferred_hostapi or "").strip().lower()
        matches = [d for d in input_devices if needle in d["name"].lower()]
        if hostapi_needle:
            hostapi_matches = [d for d in matches if hostapi_needle in d["hostapi"].lower()]
            if hostapi_matches:
                matches = hostapi_matches
        if matches:
            exact = [d for d in matches if d["name"].lower() == needle]
            selected = exact[0] if exact else matches[0]
            if len(matches) > 1:
                print(f"[MIC] WARNING: {len(matches)} devices matched '{preferred_name}'. Using index {selected['index']}.")
            print(f"[MIC] Selected by name: [{selected['index']}] {selected['name']} ({selected['hostapi']})")
            return selected["index"]
        print(f"[MIC] WARNING: device_name '{preferred_name}' not found. Falling back to index/default.")

    # 2) Index fallback
    if preferred_index is not None:
        for dev in input_devices:
            if dev["index"] == preferred_index:
                print(f"[MIC] Selected by index: [{dev['index']}] {dev['name']} ({dev['hostapi']})")
                return preferred_index
        print(f"[MIC] WARNING: device_index={preferred_index} is not a valid input device. Falling back to default.")

    # 3) System default input device fallback
    default_idx = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    if isinstance(default_idx, int) and default_idx >= 0:
        for dev in input_devices:
            if dev["index"] == default_idx:
                print(f"[MIC] Selected by system default: [{dev['index']}] {dev['name']} ({dev['hostapi']})")
                return default_idx

    # 4) Last resort: first available input device
    selected = input_devices[0]
    print(f"[MIC] Selected first available input: [{selected['index']}] {selected['name']} ({selected['hostapi']})")
    return selected["index"]

SAMPLE_RATE = 48000  # Родная частота Fifine (подтверждено пользователем)
CHUNK_MS = 32       # 1536 семплов для 48кГц, идеально делится на 3 (512 семплов для 16кГц)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000) 
TARGET_SR = 16000   # Частота для VAD и Sherpa

# ПАРАМЕТРЫ VAD И БУФЕРИЗАЦИИ
# START_RMS и HOLD_RMS - триггеры по уровню звука, если VAD тормозит
START_RMS = float(args.vad_start_rms)
HOLD_RMS = float(args.vad_hold_rms)
VAD_FLOOR_RMS = float(args.vad_floor_rms)
RMS_BRIDGE_MS = int(max(0, args.rms_bridge_ms))
MIN_SILENCE_MS = args.min_silence_ms  # [SYNC] Теперь берется из config.yaml!
MIN_SPEECH_MS = args.min_speech_ms   
PRE_ROLL_MS = args.pre_roll_ms       # [SPEEDUP] Теперь берется из config.yaml!
PRE_ROLL_CHUNKS = int(PRE_ROLL_MS / CHUNK_MS)
MAX_UTTERANCE_MS = 12000 # Максимальная длина фразы
MIC_GAIN = float(10 ** (args.mic_gain_db / 20.0))
AGC_MAX_GAIN = float(10 ** (args.agc_max_gain_db / 20.0))
SOFT_LIMIT = float(max(0.1, min(1.0, args.soft_limit)))

VAD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")

# ==============================================================
# ИНИЦИАЛИЗАЦИЯ VAD
# ==============================================================

if not os.path.exists(VAD_MODEL_PATH):
    print(f"[FATAL] VAD model not found at {VAD_MODEL_PATH}.")
    exit(1)

print("[INFO] Loading Silero VAD ONNX Session (CPU)...")
vad_session = onnxruntime.InferenceSession(VAD_MODEL_PATH, providers=["CPUExecutionProvider"])

# Модель Silero VAD ожидает state (2, 1, 128)
vad_state = np.zeros((2, 1, 128), dtype=np.float32)
last_vad_prob = 0.0
_last_vad_err_time: float = 0.0

def is_speech(audio_float32: np.ndarray, threshold: float = 0.05) -> bool:
    global vad_state, last_vad_prob, _last_vad_err_time
    
    # 1. Removal of DC offset or normalization (already done by / 32768)
    # But Silero VAD prefers centered audio
    audio_norm = audio_float32 - np.mean(audio_float32)
    
    # Debug: stats
    # print(f"DEBUG: VAD chunk: max={np.max(np.abs(audio_norm)):.4f}, mean={np.mean(audio_norm):.4f}")
    
    # 2. Reshape [batch, len]
    input_tensor = np.expand_dims(audio_norm, axis=0).astype(np.float32)
    
    # Модель VAD ожидает sr как массив [1] (int64)
    sr_scalar = np.array([TARGET_SR], dtype=np.int64)
    
    ort_inputs = {
        'input': input_tensor,
        'state': vad_state, 
        'sr': sr_scalar
    }
    
    try:
        out, new_state = vad_session.run(None, ort_inputs)
        last_vad_prob = float(out[0][0])  # type: ignore[index]
        vad_state = new_state
        return last_vad_prob > threshold
    except Exception as e:
        # Сбрасываем стейт при ошибке
        vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        if time.time() - _last_vad_err_time > 10:
            print(f"\r[VAD DEBUG] {e}")
            _last_vad_err_time = time.time()
        return False

# ==============================================================
# ПАЙПЛАЙН
# ==============================================================

# Streaming anti-aliasing filter for 48k -> 16k decimation
_butter_res = signal.butter(4, 1.0 / 3.0)
_b, _a = _butter_res[0], _butter_res[1]  # type: ignore[assignment]
_zi_init = signal.lfilter_zi(_b, _a)

state = {
    "in_speech": False,
    "active_mode": "local",
    "buffer": [],
    "pre_roll": deque(maxlen=PRE_ROLL_CHUNKS),
    "silence_counter_ms": 0,
    "duration_ms": 0,
    "speech_ms": 0,     # [ADD] Считаем именно голос (без пре-ролла и хвоста тишины)
    "speech_start_sent": False,
    "active_turn_id": None,
    "no_vad_counter_ms": 0,
    "max_rms_recent": 0.0,
    "max_prob_recent": 0.0,
    "last_samples": np.zeros(5),
    "last_log_time": time.time(),
    "resample_zi": _zi_init,
    "current_agc_gain": 1.0
}

def _ws_mode_listener():
    """Слушает события изменения режима вывода из Nira и глушит микрофон в режиме Discord."""
    import websockets.sync.client as ws_sync
    import json
    
    # 1. Начальная синхронизация режима через HTTP API
    try:
        base_api = args.nira_api.rsplit("/api", 1)[0]
        telemetry_url = f"{base_api}/api/pipeline/telemetry"
        r = session.get(telemetry_url, timeout=2)
        if r.status_code == 200:
            data = r.json().get("response") or {}
            mode = data.get("audio_output_mode", "local")
            state["active_mode"] = mode
            if mode == "discord":
                print("\n[MIC] Начальный режим: discord — локальный микрофон спит.")
    except Exception:
        pass

    # 2. Постоянное слушание WebSocket событий
    while True:
        try:
            with ws_sync.connect(args.ws_url) as ws:
                print(f"[MIC] WebSocket подключен ({args.ws_url}). Синхронизация режимов активна.")
                for msg in ws:
                    try:
                        event, _ = json.loads(msg)
                    except Exception:
                        continue
                    if event.get("message") == "audio_output_mode":
                        resp = event.get("response") or {}
                        new_mode = resp.get("mode", "local")
                        old_mode = state.get("active_mode", "local")
                        if new_mode != old_mode:
                            state["active_mode"] = new_mode
                            if new_mode == "discord":
                                state["in_speech"] = False
                                state["buffer"].clear()
                                state["pre_roll"].clear()
                                state["speech_start_sent"] = False
                                print("\n[MIC] Переключено в режим Discord — локальный микрофон отключен.")
                            else:
                                print("\n[MIC] Переключено в режим Local — локальный микрофон активирован.")
        except Exception:
            time.sleep(3)

# [OPTIMIZE] Use Session for faster subsequent requests
session = requests.Session()

def maybe_send_speech_start(turn_id: Optional[str]):
    global _last_speech_start_ts_ms
    now_ms = time.time() * 1000.0
    if (now_ms - _last_speech_start_ts_ms) < max(0, args.speech_start_min_interval_ms):
        return
    _last_speech_start_ts_ms = now_ms
    payload_turn_id = str(turn_id) if turn_id else None

    def _send():
        payload = {"timestamp": time.time(), "source_id": args.source_id}
        if payload_turn_id:
            payload["turn_id"] = payload_turn_id
        try:
            session.post(SPEECH_START_API, json=payload, timeout=1)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


def begin_turn_id(now_s: float) -> str:
    global _last_turn_id, _last_turn_ts
    with _turn_lock:
        if _last_turn_id and ((now_s - _last_turn_ts) * 1000.0) <= max(0, args.turn_merge_window_ms):
            return _last_turn_id
        return str(uuid.uuid4())


def complete_turn_id(turn_id: str, now_s: float):
    global _last_turn_id, _last_turn_ts
    with _turn_lock:
        _last_turn_id = str(turn_id)
        _last_turn_ts = float(now_s)


def send_to_nira(audio_buffer: list, turn_id: Optional[str] = None, speech_start_ts: Optional[float] = None, speech_end_ts: Optional[float] = None):
    """Отправка на сервер (асинхронно из потока)"""
    # [OPTIMIZE] Move concatenation to thread to not block audio_callback
    audio_data = np.concatenate(audio_buffer)
    audio_dur = len(audio_data) / TARGET_SR
    now_s = time.time()
    if speech_end_ts is None:
        speech_end_ts = now_s
    if speech_start_ts is None:
        speech_start_ts = max(0.0, speech_end_ts - audio_dur)
    
    # Конвертируем обратно в int16 bytes
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_data * 32767).astype('<i2').tobytes()
    
    base64_audio = base64.b64encode(audio_int16).decode('utf-8')
    utterance_id = str(uuid.uuid4())
    turn_id = str(turn_id) if turn_id else begin_turn_id(now_s)
    payload = {
        "user": args.user, 
        "timestamp": speech_start_ts,
        "speech_start_ts": speech_start_ts,
        "speech_end_ts": speech_end_ts,
        "audio_bytes": base64_audio,
        "sr": TARGET_SR,  # ИСПРАВЛЕНО: Шлем 16000, так как данные ресемплированы
        "sw": 2, 
        "ch": 1,
        "source_id": args.source_id,
        "turn_id": turn_id,
        "utterance_id": utterance_id,
    }
    
    try:
        response = session.post(args.nira_api, json=payload, timeout=5)
        if response.status_code == 200:
             accepted = True
             try:
                 body = response.json()
                 accepted = bool((body or {}).get("response", {}).get("accepted", True))
             except Exception:
                 pass
             if accepted:
                 complete_turn_id(turn_id, now_s)
                 print(f"\n[API] Фраза отправлена ({len(audio_int16)} байт).")
             else:
                 print(f"\n[API] Фраза отброшена backpressure-политикой ({len(audio_int16)} байт).")
        else:
             print(f"\n[API] Ошибка: {response.status_code}")
    except Exception as e:
        print(f"\n[API] Ошибка соединения: {e}")


def apply_fixed_gain(samples: np.ndarray) -> np.ndarray:
    out = samples.astype(np.float32, copy=False) * MIC_GAIN
    return np.clip(out, -SOFT_LIMIT, SOFT_LIMIT)


def apply_gain_and_agc(samples: np.ndarray) -> np.ndarray:
    global state
    out = samples.astype(np.float32, copy=False)
    if args.agc_enable:
        rms = float(np.sqrt(np.mean(out ** 2)))
        target_gain = 1.0
        if rms > max(1e-6, VAD_FLOOR_RMS):
            target = max(1e-6, float(args.agc_target_rms))
            target_gain = min(target / rms, AGC_MAX_GAIN)
            if target_gain < 1.0:
                target_gain = 1.0
                
        # Сглаживание AGC (Attack/Release), чтобы избежать рваных искажений на границах чанков
        current_gain = state["current_agc_gain"]
        new_gain = current_gain * 0.9 + target_gain * 0.1
        state["current_agc_gain"] = new_gain
        
        if new_gain > 1.0:
            out = out * new_gain

    # Hard clip to avoid numeric overflow before int16 conversion.
    out = np.clip(out, -SOFT_LIMIT, SOFT_LIMIT)
    return out

def audio_callback(indata, frames, time_info, status):
    global state
    if status:
        print(f"[SD STATUS] {status}", file=sys.stderr)

    # Если активен режим Discord — локальный микрофон полностью спит
    if state.get("active_mode") == "discord":
        return

    # 1. Берем канал (Fifine в моно отдает один или два канала)
    ch_native = indata[:, 0]
    
    # 2. Ресемплинг 48к -> 16к.
    # Для устранения алиасинга используем lfilter с сохранением стейта, а затем прореживание.
    zi = state["resample_zi"]
    ch_filtered, zi = signal.lfilter(_b, _a, ch_native, zi=zi)
    state["resample_zi"] = zi
    ch16_raw = ch_filtered[::3].astype(np.float32)

    # Use fixed-gain signal for VAD/RMS decisions (more stable on noise),
    # and AGC-enriched signal only for STT payload quality.
    ch16_vad = apply_fixed_gain(ch16_raw)
    ch16 = apply_gain_and_agc(ch16_vad)
    state["last_samples"] = ch16_vad[:5]
    
    # 3. RMS (считаем по 16кГц сигналу)
    rms = np.sqrt(np.mean(ch16_vad**2))
    if rms > state["max_rms_recent"]: state["max_rms_recent"] = rms
    
    # 4. VAD
    vad_prob = 0.0
    if rms > VAD_FLOOR_RMS: 
        is_speech(ch16_vad, args.vad_threshold)
        vad_prob = last_vad_prob
    
    if vad_prob > state["max_prob_recent"]: state["max_prob_recent"] = vad_prob

    # Логика старта/удержания: VAD + RMS
    if not state["in_speech"]:
        state["pre_roll"].append(ch16.copy())
        state["no_vad_counter_ms"] = 0
        is_active_speech = (vad_prob > args.vad_threshold) or (rms > START_RMS)
    else:
        hold_vad_threshold = max(0.02, args.vad_threshold * 0.4)
        vad_active = vad_prob > hold_vad_threshold
        rms_active = rms > HOLD_RMS

        if vad_active or rms_active:
            state["no_vad_counter_ms"] = 0
            is_active_speech = True
        else:
            # Brief gap bridge for unvoiced consonants / micro-pauses.
            state["no_vad_counter_ms"] += CHUNK_MS
            is_active_speech = state["no_vad_counter_ms"] <= RMS_BRIDGE_MS

    if is_active_speech:
        if not state["in_speech"]:
            state["in_speech"] = True
            print(f"\n[VAD] Голос! (Prob: {vad_prob:.3f}, RMS: {rms:.4f})", end="", flush=True)
            # Добавляем пре-ролл
            if "pre_roll" in state and len(state["pre_roll"]) > 0:
                state["buffer"] = list(state["pre_roll"])
                state["pre_roll"].clear()
            else:
                state["buffer"] = []
            state["speech_ms"] = 0
            state["speech_start_sent"] = False
            state["active_turn_id"] = begin_turn_id(time.time())
            state["no_vad_counter_ms"] = 0
            # Считаем длину в мс (каждый чанк = CHUNK_MS)
            state["duration_ms"] = len(state["buffer"]) * CHUNK_MS
            
        state["buffer"].append(ch16.copy())
        state["duration_ms"] += CHUNK_MS
        state["speech_ms"] += CHUNK_MS # Считаем только активную речь
        state["silence_counter_ms"] = 0

        # Отправляем speech_start после подтверждения непрерывной речи.
        if (not state["speech_start_sent"]) and state["speech_ms"] >= max(0, args.speech_start_confirm_ms):
            maybe_send_speech_start(state.get("active_turn_id"))
            state["speech_start_sent"] = True
        
        # [SAFETY] Отработка MAX_UTTERANCE_MS
        if state["duration_ms"] > MAX_UTTERANCE_MS:
            print(f" [LIMIT: {MAX_UTTERANCE_MS}ms].")
            is_active_speech = False # Принудительно завершаем ниже
            
    if not is_active_speech:
        if state["in_speech"]:
            state["buffer"].append(ch16.copy())
            state["duration_ms"] += CHUNK_MS
            state["silence_counter_ms"] += CHUNK_MS
            
            if state["silence_counter_ms"] >= args.min_silence_ms or state["duration_ms"] > MAX_UTTERANCE_MS:
                state["in_speech"] = False
                print(f" Завершена ({int(state['duration_ms'])}ms, speech: {int(state['speech_ms'])}ms).")
                
                # Шлем, если:
                # 1) обычная фраза длиннее стандартного min_speech_ms
                # 2) ИЛИ короткая "командная" фраза (например "стоп"), если уже был подтвержден speech_start
                meets_regular_min = state["speech_ms"] >= args.min_speech_ms
                likely_voice = (
                    state["max_prob_recent"] >= max(0.001, args.vad_threshold * 0.5)
                    or state["max_rms_recent"] >= START_RMS
                )
                # Отдельная дорожка для коротких команд ("стоп", "стой", "подожди"):
                # позволяем отправку даже если speech_start еще не ушел, но только при признаках реального голоса.
                meets_short_interrupt_min = state["speech_ms"] >= args.min_speech_ms_interrupt and likely_voice
                if meets_regular_min or meets_short_interrupt_min:
                    speech_end_time = time.time() - (state["silence_counter_ms"] / 1000.0)
                    speech_start_time = max(0.0, speech_end_time - (state["duration_ms"] / 1000.0))
                    # [OPTIMIZE] Pass buffer to thread, concatenation happens there
                    threading.Thread(
                        target=send_to_nira,
                        args=(list(state["buffer"]), state.get("active_turn_id"), speech_start_time, speech_end_time),
                        daemon=True,
                    ).start()
                else:
                    print(f"[VAD] Отклонено: слишком коротко ({int(state['speech_ms'])}ms)")
                
                state["buffer"] = []
                state["duration_ms"] = 0
                state["silence_counter_ms"] = 0
                state["speech_start_sent"] = False
                state["active_turn_id"] = None
                state["no_vad_counter_ms"] = 0
                # Сброс состояния VAD после фразы
                vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            # Копим пре-ролл
            if "pre_roll" in state:
                state["pre_roll"].append(ch16.copy())

    # Раз в секунду выводим статус
    now = time.time()
    if not state["in_speech"] and now - state["last_log_time"] > 1.0:
        indicator = "🔊" if state["max_rms_recent"] > args.energy_threshold else "🤫"
        # Отладочный вывод: Prob теперь всегда виден
        samples_str = ", ".join([f"{x:.4f}" for x in state["last_samples"]])
    # sys.stdout.write(f"\r[MIC] {indicator} RMS: {state['max_rms_recent']:.4f} | Prob: {state['max_prob_recent']:.3f} | Smp: [{samples_str}]   ")
    # sys.stdout.flush()
        state["max_rms_recent"] = 0.0
        state["max_prob_recent"] = 0.0
        state["last_log_time"] = now

def run_sd():
    dev_idx = resolve_input_device_index(
        preferred_index=args.device_index,
        preferred_name=args.device_name,
        preferred_hostapi=args.device_hostapi,
    )
    try:
        # Пытаемся открыть поток. Sounddevice сам делает ресемплинг если нужно!
        with sd.InputStream(device=dev_idx,
                            channels=None, # Возвращаем авто-выбор (Fifine лучше работает так)
                            samplerate=SAMPLE_RATE,
                            blocksize=CHUNK_SIZE,
                            dtype='float32',
                            callback=audio_callback):
            
            info = sd.query_devices(dev_idx, 'input')
            hostapi_name = _get_hostapi_name(int(info.get("hostapi", -1)))
            print(f"[INFO] SoundDevice Listening on: {info['name']} (ID: {dev_idx}, hostapi: {hostapi_name})")
            print("========================================================\n")
            
            while True:
                sd.sleep(1000)
                
    except Exception as e:
        print(f"[FATAL] Error in sounddevice: {e}")

if __name__ == "__main__":
    try:
        if args.list_devices:
            print_input_devices()
            raise SystemExit(0)
        # Запускаем синхронизацию режима вывода по WebSocket в фоновом потоке
        threading.Thread(target=_ws_mode_listener, daemon=True).start()
        run_sd()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
