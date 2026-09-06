import logging
import os
import subprocess
import sys
import threading

from utils.config import Config
from utils.processes.base import BaseProcess


class DiscordProcess(BaseProcess):
    """Runs the Discord bridge outside the API event loop."""

    def __init__(self):
        super().__init__("discord")

    async def reload(self):
        await self.unload()

        config = dict(self.runtime_config or Config().discord or {})
        if not config.get("enabled", False):
            logging.info("Discord bridge is disabled in config.")
            return

        if not os.getenv("DISCORD_BOT_TOKEN"):
            logging.error("Discord bridge was not started: DISCORD_BOT_TOKEN is missing.")
            return

        required = ("guild_id", "voice_channel_id")
        missing = [key for key in required if not config.get(key)]
        if missing:
            logging.error("Discord bridge was not started: missing config fields: %s", ", ".join(missing))
            return

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        script_path = os.path.join(root_dir, "apps", "discord-bridge", "main.py")
        cmd = [
            sys.executable,
            "-u",
            script_path,
            "--guild-id", str(config["guild_id"]),
            "--voice-channel-id", str(config["voice_channel_id"]),
            "--api-url", str(config.get("api_url", "http://127.0.0.1:7272/api/context/conversation/audio")),
            "--ws-url", str(config.get("ws_url", "ws://127.0.0.1:7272/")),
            "--source-id", str(config.get("source_id", "discord")),
            "--silence-ms", str(config.get("silence_ms", 1200)),
            "--min-speech-ms", str(config.get("min_speech_ms", 220)),
        ]
        if not config.get("auto_join", True):
            cmd.append("--no-auto-join")

        logging.info("Starting Discord bridge for guild %s, channel %s.", config["guild_id"], config["voice_channel_id"])
        self.process = subprocess.Popen(
            cmd,
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def stream_logs(pipe):
            if pipe:
                for line in iter(pipe.readline, ""):
                    if "/api/bridge/discord/status" in line:
                        continue
                    print(f"[discord] {line}", end="", flush=True)
                pipe.close()

        threading.Thread(target=stream_logs, args=(self.process.stdout,), daemon=True).start()
