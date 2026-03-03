import os
import time
import json
import base64
import argparse
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
parser.add_argument("--jaison_api", type=str, default="http://localhost:7272/api/context/conversation/audio", help="JAIson API URL")
parser.add_argument("--speech_start_api", type=str, default=None, help="Optional early-barge-in URL (default: derived from jaison_api)")
parser.add_argument("--speech_start_min_interval_ms", type=int, default=900, help="Minimum interval between speech_start signals")
parser.add_argument("--speech_start_confirm_ms", type=int, default=350, help="Require this much active speech before sending speech_start")
parser.add_argument("--min_speech_ms_interrupt", type=int, default=120, help="Minimum ms for short interrupt commands to still be sent")
parser.add_argument("--vad_threshold", type=float, default=0.2, help="Probability threshold for VAD")
parser.add_argument("--min_silence_ms", type=int, default=500, help="Milliseconds of silence to split phrase")
parser.add_argument("--min_speech_ms", type=int, default=200, help="Minimum ms of speech to send")
parser.add_argument("--pre_roll_ms", type=int, default=300, help="Milliseconds of audio to keep before speech")
parser.add_argument("--energy_threshold", type=float, default=0.01, help="RMS threshold (gate)")
args = parser.parse_args()


def resolve_speech_start_url(audio_url: str, explicit_url: Optional[str]) -> str:
    if explicit_url:
        return explicit_url
    if audio_url.endswith("/audio"):
        return audio_url[:-len("/audio")] + "/speech_start"
    return audio_url.rstrip("/") + "/speech_start"


SPEECH_START_API = resolve_speech_start_url(args.jaison_api, args.speech_start_api)
_last_speech_start_ts_ms = 0.0


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
START_RMS = 0.012     # [ADJUST] Был 0.005, слишком много шума ловил
HOLD_RMS = 0.008      # [ADJUST] Был 0.003
MIN_SILENCE_MS = args.min_silence_ms  # [SYNC] Теперь берется из config.yaml!
MIN_SPEECH_MS = args.min_speech_ms   
PRE_ROLL_MS = args.pre_roll_ms       # [SPEEDUP] Теперь берется из config.yaml!
PRE_ROLL_CHUNKS = int(PRE_ROLL_MS / CHUNK_MS)
MAX_UTTERANCE_MS = 12000 # Максимальная длина фразы

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

def is_speech(audio_float32: np.ndarray, threshold: float = 0.05) -> bool:
    global vad_state, last_vad_prob
    
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
        last_vad_prob = float(out[0][0])
        vad_state = new_state
        return last_vad_prob > threshold
    except Exception as e:
        # Сбрасываем стейт при ошибке
        vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        if not hasattr(is_speech, 'last_err_time') or time.time() - is_speech.last_err_time > 10:
            print(f"\r[VAD DEBUG] {e}")
            is_speech.last_err_time = time.time()
        return False

# ==============================================================
# ПАЙПЛАЙН
# ==============================================================

state = {
    "in_speech": False,
    "buffer": [],
    "pre_roll": deque(maxlen=PRE_ROLL_CHUNKS),
    "silence_counter_ms": 0,
    "duration_ms": 0,
    "speech_ms": 0,     # [ADD] Считаем именно голос (без пре-ролла и хвоста тишины)
    "speech_start_sent": False,
    "max_rms_recent": 0.0,
    "max_prob_recent": 0.0,
    "last_samples": np.zeros(5),
    "last_log_time": time.time()
}

# [OPTIMIZE] Use Session for faster subsequent requests
session = requests.Session()


def send_speech_start():
    payload = {"timestamp": time.time()}
    try:
        session.post(SPEECH_START_API, json=payload, timeout=1)
    except Exception:
        pass


def maybe_send_speech_start():
    global _last_speech_start_ts_ms
    now_ms = time.time() * 1000.0
    if (now_ms - _last_speech_start_ts_ms) < max(0, args.speech_start_min_interval_ms):
        return
    _last_speech_start_ts_ms = now_ms
    threading.Thread(target=send_speech_start, daemon=True).start()

def send_to_jaison(audio_buffer: list):
    """Отправка на сервер (асинхронно из потока)"""
    # [OPTIMIZE] Move concatenation to thread to not block audio_callback
    audio_data = np.concatenate(audio_buffer)
    
    # Конвертируем обратно в int16 bytes
    audio_int16 = (audio_data * 32767).astype(np.int16).tobytes()
    
    base64_audio = base64.b64encode(audio_int16).decode('utf-8')
    payload = {
        "user": "Creator", 
        "timestamp": time.time(),
        "audio_bytes": base64_audio,
        "sr": TARGET_SR,  # ИСПРАВЛЕНО: Шлем 16000, так как данные ресемплированы
        "sw": 2, 
        "ch": 1
    }
    
    try:
        response = session.post(args.jaison_api, json=payload, timeout=5)
        if response.status_code == 200:
             print(f"\n[API] Фраза отправлена ({len(audio_int16)} байт).")
        else:
             print(f"\n[API] Ошибка: {response.status_code}")
    except Exception as e:
        print(f"\n[API] Ошибка соединения: {e}")

def audio_callback(indata, frames, time_info, status):
    global state
    if status:
        print(f"[SD STATUS] {status}", file=sys.stderr)

    # 1. Берем канал (Fifine в моно отдает один или два канала)
    ch_native = indata[:, 0]
    
    # 2. Быстрый ресемплинг 3:1 (48000 -> 16000)
    # [OPTIMIZE] Децимация (1 из 3) значительно быстрее resample_poly
    ch16 = ch_native[::3].astype(np.float32)

    state["last_samples"] = ch16[:5]
    
    # 3. RMS (считаем по 16кГц сигналу)
    rms = np.sqrt(np.mean(ch16**2))
    if rms > state["max_rms_recent"]: state["max_rms_recent"] = rms
    
    # 4. VAD
    vad_prob = 0.0
    if rms > 0.001: 
        is_speech(ch16, args.vad_threshold)
        vad_prob = last_vad_prob
    
    if vad_prob > state["max_prob_recent"]: state["max_prob_recent"] = vad_prob

    # Логика старта/удержания: VAD + RMS
    if not state["in_speech"]:
        state["pre_roll"].append(ch16.copy())
        is_active_speech = (vad_prob > args.vad_threshold) or (rms > START_RMS)
    else:
        is_active_speech = (vad_prob > (args.vad_threshold * 0.4)) or (rms > HOLD_RMS)

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
            # Считаем длину в мс (каждый чанк = CHUNK_MS)
            state["duration_ms"] = len(state["buffer"]) * CHUNK_MS
            
        state["buffer"].append(ch16.copy())
        state["duration_ms"] += CHUNK_MS
        state["speech_ms"] += CHUNK_MS # Считаем только активную речь
        state["silence_counter_ms"] = 0

        # Отправляем speech_start только после короткого подтверждения непрерывной речи.
        # Это уменьшает ложные прерывания от шумов/коротких "угу".
        if (not state["speech_start_sent"]) and state["speech_ms"] >= max(0, args.speech_start_confirm_ms):
            maybe_send_speech_start()
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
                    # [OPTIMIZE] Pass buffer to thread, concatenation happens there
                    threading.Thread(target=send_to_jaison, args=(list(state["buffer"]),), daemon=True).start()
                else:
                    print(f"[VAD] Отклонено: слишком коротко ({int(state['speech_ms'])}ms)")
                
                state["buffer"] = []
                state["duration_ms"] = 0
                state["silence_counter_ms"] = 0
                state["speech_start_sent"] = False
                # Сброс состояния VAD после фразы
                vad_state[:] = 0.0
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
        run_sd()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
