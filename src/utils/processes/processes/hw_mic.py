import os
import subprocess
from utils.config import Config
from utils.processes.base import BaseProcess

class HwMicProcess(BaseProcess):
    def __init__(self):
        super().__init__("hw_mic")
        
    async def reload(self):
        await self.unload()
        
        config = Config()
        mic_conf = config.microphone or {}
        
        # Запускаем только если в конфиге enabled: true
        if not mic_conf.get("enabled", False):
            return
            
        print("[ProcessManager] Starting Hardware Microphone (hw-mic-client)...")
        
        # Получаем параметры
        dev_idx = mic_conf.get("device_index", None)
        dev_name = mic_conf.get("device_name", None)
        dev_hostapi = mic_conf.get("device_hostapi", None)
        vad_thresh = mic_conf.get("vad_threshold", 0.15)
        min_silence = mic_conf.get("min_silence_ms", 1500)
        min_speech = mic_conf.get("min_speech_ms", 250)
        pre_roll = mic_conf.get("pre_roll_ms", 300)
        speech_start_min_interval_ms = mic_conf.get("speech_start_min_interval_ms", 900)
        speech_start_confirm_ms = mic_conf.get("speech_start_confirm_ms", 350)
        min_speech_ms_interrupt = mic_conf.get("min_speech_ms_interrupt", 120)
        source_id = mic_conf.get("source_id", "mic")
        turn_merge_window_ms = mic_conf.get("turn_merge_window_ms", 2200)
        resample_mode = mic_conf.get("resample_mode", "polyphase")
        mic_gain_db = mic_conf.get("mic_gain_db", 12.0)
        agc_enable = mic_conf.get("agc_enable", 1)
        agc_target_rms = mic_conf.get("agc_target_rms", 0.05)
        agc_max_gain_db = mic_conf.get("agc_max_gain_db", 15.0)
        soft_limit = mic_conf.get("soft_limit", 0.97)
        vad_start_rms = mic_conf.get("vad_start_rms", 0.012)
        vad_hold_rms = mic_conf.get("vad_hold_rms", 0.008)
        vad_floor_rms = mic_conf.get("vad_floor_rms", 0.001)
        rms_bridge_ms = mic_conf.get("rms_bridge_ms", 96)
        
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../apps/hw-mic-client/main.py"))
        
        import sys
        
        cmd = [sys.executable, "-u", script_path]
        
        if dev_idx is not None:
             cmd.extend(["--device_index", str(dev_idx)])
        if dev_name:
             cmd.extend(["--device_name", str(dev_name)])
        if dev_hostapi:
             cmd.extend(["--device_hostapi", str(dev_hostapi)])
             
        cmd.extend([
            "--vad_threshold", str(vad_thresh),
            "--min_silence_ms", str(min_silence),
            "--min_speech_ms", str(min_speech),
            "--pre_roll_ms", str(pre_roll),
            "--speech_start_min_interval_ms", str(speech_start_min_interval_ms),
            "--speech_start_confirm_ms", str(speech_start_confirm_ms),
            "--min_speech_ms_interrupt", str(min_speech_ms_interrupt),
            "--source_id", str(source_id),
            "--turn_merge_window_ms", str(turn_merge_window_ms),
            "--resample_mode", str(resample_mode),
            "--mic_gain_db", str(mic_gain_db),
            "--agc_enable", str(int(bool(agc_enable))),
            "--agc_target_rms", str(agc_target_rms),
            "--agc_max_gain_db", str(agc_max_gain_db),
            "--soft_limit", str(soft_limit),
            "--vad_start_rms", str(vad_start_rms),
            "--vad_hold_rms", str(vad_hold_rms),
            "--vad_floor_rms", str(vad_floor_rms),
            "--rms_bridge_ms", str(rms_bridge_ms),
            "--jaison_api", "http://localhost:7272/api/context/conversation/audio",
            "--speech_start_api", "http://localhost:7272/api/context/conversation/speech_start"
        ])

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # [OPTIMIZE] Line-buffered for real-time console logs
        )
        
        # Чтение вывода в фоне (просто печатаем в консоль JAIson)
        import threading
        def stream_logs(pipe):
            if pipe:
                for line in iter(pipe.readline, ''):
                    print(line, end='', flush=True)
                pipe.close()

        threading.Thread(target=stream_logs, args=(self.process.stdout,), daemon=True).start()
