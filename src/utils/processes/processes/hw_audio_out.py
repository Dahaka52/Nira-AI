import os
import subprocess

from utils.config import Config
from utils.processes.base import BaseProcess


class HwAudioOutProcess(BaseProcess):
    def __init__(self):
        super().__init__("hw_audio_out")

    async def reload(self):
        await self.unload()

        config = Config()
        out_conf = config.audio_output or {}
        if not out_conf.get("enabled", False):
            return

        print("[ProcessManager] Starting Hardware Audio Output (hw-audio-out-client)...")

        dev_idx = out_conf.get("device_index", None)
        dev_name = out_conf.get("device_name", None)
        dev_hostapi = out_conf.get("device_hostapi", None)
        ws_url = out_conf.get("ws_url", "ws://127.0.0.1:7272/")
        sample_rate = out_conf.get("sample_rate", 24000)
        channels = out_conf.get("channels", 1)
        blocksize = out_conf.get("blocksize", 256)
        max_buffer_ms = out_conf.get("max_buffer_ms", 700)
        clear_on_stop = int(bool(out_conf.get("clear_on_stop", True)))
        output_gain_db = out_conf.get("output_gain_db", 0.0)
        reconnect_delay_ms = out_conf.get("reconnect_delay_ms", 1200)

        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../apps/hw-audio-out-client/main.py")
        )
        import sys

        cmd = [sys.executable, "-u", script_path]

        if dev_idx is not None:
            cmd.extend(["--device_index", str(dev_idx)])
        if dev_name:
            cmd.extend(["--device_name", str(dev_name)])
        if dev_hostapi:
            cmd.extend(["--device_hostapi", str(dev_hostapi)])

        cmd.extend(
            [
                "--ws_url",
                str(ws_url),
                "--sample_rate",
                str(sample_rate),
                "--channels",
                str(channels),
                "--blocksize",
                str(blocksize),
                "--max_buffer_ms",
                str(max_buffer_ms),
                "--clear_on_stop",
                str(clear_on_stop),
                "--output_gain_db",
                str(output_gain_db),
                "--reconnect_delay_ms",
                str(reconnect_delay_ms),
            ]
        )

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        import threading

        def stream_logs(pipe):
            if pipe:
                for line in iter(pipe.readline, ""):
                    print(line, end="", flush=True)
                pipe.close()

        threading.Thread(target=stream_logs, args=(self.process.stdout,), daemon=True).start()
