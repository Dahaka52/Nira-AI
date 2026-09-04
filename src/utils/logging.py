import logging
import os
import sys
from utils.helpers.time import get_current_time
from utils.args import args

# Включаем VT100 на Windows для ANSI-цветов
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

START_TIME = get_current_time(include_ms=False, as_str=False)


class CustomFormatter(logging.Formatter):
    """
    Читаемый форматтер для консоли Нира.

    Дизайн строки лога:
      12:34:56  [INFO ]  utils.jaison  Сообщение...
       dim        bold     bright         normal

    Уровни — без заливки фона, только жирный цветной [LEVEL].
    Имена логгеров — яркие, хорошо различимые.
    """
    _RST    = "\x1b[0m"
    _DIM    = "\x1b[2m"
    _BOLD   = "\x1b[1m"

    # Цвета уровней (foreground only, без фона)
    _LEVEL_FG = {
        logging.DEBUG:    "\x1b[90m",    # серый
        logging.INFO:     "\x1b[92m",    # ярко-зелёный
        logging.WARNING:  "\x1b[93m",    # ярко-жёлтый
        logging.ERROR:    "\x1b[91m",    # ярко-красный
        logging.CRITICAL: "\x1b[95m",    # ярко-пурпурный
    }

    # Цвета для имён логгеров (по ключевому слову в имени)
    _NAME_COLORS = [
        ("whisper",   "\x1b[96m"),   # циановый
        ("sherpa",    "\x1b[96m"),   # циановый
        ("stt",       "\x1b[96m"),   # циановый
        ("tts",       "\x1b[38;5;205m"),  # розовый
        ("fish",      "\x1b[38;5;205m"),  # розовый
        ("jaison",    "\x1b[95m"),   # пурпурный
        ("t2t",       "\x1b[95m"),   # пурпурный
        ("llm",       "\x1b[95m"),   # пурпурный
        ("mic",       "\x1b[93m"),   # жёлтый
        ("vad",       "\x1b[93m"),   # жёлтый
        ("discord",   "\x1b[94m"),   # синий
        ("server",    "\x1b[94m"),   # синий
        ("config",    "\x1b[97m"),   # белый
        ("root",      "\x1b[97m"),   # белый (root — самый частый)
    ]
    _DEFAULT_NAME_COLOR = "\x1b[97m"  # белый по умолчанию

    def _get_name_color(self, name: str) -> str:
        low = name.lower()
        for keyword, color in self._NAME_COLORS:
            if keyword in low:
                return color
        return self._DEFAULT_NAME_COLOR

    def format(self, record: logging.LogRecord) -> str:
        lvl_fg    = self._LEVEL_FG.get(record.levelno, "\x1b[97m")
        name_fg   = self._get_name_color(record.name)

        # Составные части строки
        time_part  = f"{self._DIM}{self.formatTime(record, '%H:%M:%S')}{self._RST}"
        level_part = f"{self._BOLD}{lvl_fg}[{record.levelname:<5}]{self._RST}"
        name_part  = f"{self._BOLD}{name_fg}{record.name}{self._RST}"
        msg_part   = record.getMessage()

        # Добавляем exc_info если есть
        result = f"{time_part}  {level_part}  {name_part}  {msg_part}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            result += f"\n{self._DIM}{record.exc_text}{self._RST}"
        return result


def setup_logger():
    global START_TIME

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level))

    # Файловый лог — подробный (с filename и lineno)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-5.5s] [%(filename)s::%(lineno)d %(funcName)s]: %(message)s"
    )
    file_handler = logging.FileHandler(
        os.path.join(
            args.log_dir,
            "{}.log".format(get_current_time(include_ms=False, as_str=False).strftime("%Y-%m-%d"))
        )
    )
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