import logging
import os
import sys
import copy
from utils.helpers.time import get_current_time
from utils.args import args

START_TIME = get_current_time(include_ms=False,as_str=False)

# Setup formatters and handlers
class CustomFormatter(logging.Formatter):
    reset = "\x1b[0m"
    accent_user = "\x1b[1;96m"      # bright cyan
    accent_nira = "\x1b[1;95m"      # bright magenta/pink
    accent_system = "\x1b[1;93m"    # bright yellow
    accent_critical = "\x1b[1;91m"  # bright red

    user_markers = (
        "[USER_VOICE]",
        "[USER_TEXT]",
        "[USER_TEXT_API]",
    )
    nira_markers = (
        "[NIRA_REPLY]",
    )
    system_markers = (
        "Smart Barge-in",
        "Queued new response job",
        "Voice turn committed as context-only",
        "Qwen3 streaming segment",
        "Qwen3 stream stats",
    )
    critical_markers = (
        "Buffer overflow",
        "failed",
        "Traceback",
        "timeout",
        "cancelled",
        "is being cancelled",
    )

    base_time = "[%(asctime)s]" + reset
    base_level = "[%(levelname)-5.5s]" + reset
    base_func = "[%(filename)s::%(lineno)d %(funcName)s]:" + reset
    base_msg = "%(message)s" + reset

    template_line = "\x1b[1m\x1b[1;34m" + base_time + " {}" + base_level + " \x1b[1m\x1b[1;33m" + base_func + " " + base_msg

    FORMATS = {
        logging.DEBUG: template_line.format("\x1b[1m\x1b[1;30m\x1b[47m"),
        logging.INFO: template_line.format("\x1b[1m\x1b[1;30m\x1b[42m"),
        logging.WARNING: template_line.format("\x1b[1m\x1b[1;30m\x1b[43m"),
        logging.ERROR: template_line.format("\x1b[1m\x1b[1;30m\x1b[41m"),
        logging.CRITICAL: template_line.format("\x1b[1m\x1b[31m\x1b[45m")
    }

    def _style_message(self, message: str) -> str:
        msg = str(message or "")
        low = msg.lower()

        if any(marker in msg for marker in self.user_markers):
            return f"{self.accent_user}{msg}{self.reset}"
        if any(marker in msg for marker in self.nira_markers):
            return f"{self.accent_nira}{msg}{self.reset}"
        if "response job" in low and "completed. content:" in low:
            return f"{self.accent_nira}{msg}{self.reset}"
        if any(marker.lower() in low for marker in self.critical_markers):
            return f"{self.accent_critical}{msg}{self.reset}"
        if any(marker.lower() in low for marker in self.system_markers):
            return f"{self.accent_system}{msg}{self.reset}"
        return msg

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        local_record = copy.copy(record)
        local_record.msg = self._style_message(record.getMessage())
        local_record.args = ()
        return formatter.format(local_record)
    
def setup_logger():
    global START_TIME
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level))

    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)-5.5s] [%(filename)s::%(lineno)d %(funcName)s]: %(message)s")
    file_handler = logging.FileHandler(
        os.path.join(args.log_dir, "{}.log".format(get_current_time(include_ms=False,as_str=False).strftime("%Y-%m-%d"))))
    file_handler.setFormatter(file_formatter)
        
    logger.addHandler(file_handler)

    if not args.silent:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)

    # [OPTIMIZE] Suppress noisy logs
    logging.getLogger("hypercorn.access").setLevel(logging.WARNING)
    logging.getLogger("hypercorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
