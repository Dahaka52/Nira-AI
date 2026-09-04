"""
console.py — Утилиты цветного вывода в консоль для Nira.

=======================================================
  ЦВЕТОВАЯ СХЕМА
=======================================================
  STT  распознавание        — циановый       [96m
  VAD  голосовая активность — жёлтый         [93m
  LLM  генерация ответа     — пурпурный      [95m
  TTS  синтез речи          — ярко-розовый   [38;5;205m
  BARGE-IN прерывание       — красный        [91m
  Пользователь (CLI)        — зелёный        [92m
  Нира ответ (CLI)          — пурпурный      [95m
  Этап BOOT                 — синий          [94m
  Этап OK                   — зелёный        [92m
  Этап WARN                 — жёлтый         [93m
  Этап ERROR                — красный        [91m
  Разделители               — тёмно-серый    [90m
=======================================================

ДИЗАЙН BADGE:
  Нет заливки фона — только яркая рамка из символов: [ LABEL ]
  Текст badge — всегда жирный + цвет операции.
  Это максимально читаемо на любом терминале.
=======================================================
"""

import sys
import os

# Включаем ANSI-коды на Windows (через VT100 mode)
if sys.platform == "win32":
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


# ── ANSI escape коды ──────────────────────────────────────────────────────────
RESET     = "\x1b[0m"
BOLD      = "\x1b[1m"
DIM       = "\x1b[2m"
ITALIC    = "\x1b[3m"

# Foreground (high-intensity)
C_BLACK   = "\x1b[90m"           # тёмно-серый → разделители
C_RED     = "\x1b[91m"           # ярко-красный → ошибки, barge-in
C_GREEN   = "\x1b[92m"           # ярко-зелёный → пользователь, ok
C_YELLOW  = "\x1b[93m"           # ярко-жёлтый → VAD, предупреждения
C_BLUE    = "\x1b[94m"           # ярко-синий → boot-этапы
C_MAGENTA = "\x1b[95m"           # ярко-пурпурный → LLM / Нира
C_CYAN    = "\x1b[96m"           # ярко-циановый → STT
C_WHITE   = "\x1b[97m"           # белый
C_PINK    = "\x1b[38;5;205m"     # ярко-розовый (hot pink) → TTS
C_LPURPLE = "\x1b[38;5;135m"     # лавандовый → дополнительный акцент

# Background (используются только в header и divider — НЕ в badge)
BG_BLACK   = "\x1b[40m"
BG_RED     = "\x1b[41m"
BG_GREEN   = "\x1b[42m"
BG_YELLOW  = "\x1b[43m"
BG_BLUE    = "\x1b[44m"
BG_MAGENTA = "\x1b[45m"
BG_CYAN    = "\x1b[46m"
BG_WHITE   = "\x1b[47m"
BG_GREY    = "\x1b[100m"


# ── Приватные хелперы ─────────────────────────────────────────────────────────

def _width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def _badge(label: str, color: str) -> str:
    """
    Читаемый badge без заливки фона.
    Формат: BOLD + color + [LABEL] + RESET
    Пример: \x1b[1m\x1b[96m[STT]\x1b[0m
    """
    return f"{BOLD}{color}[{label}]{RESET}"


# ── Публичный API ─────────────────────────────────────────────────────────────

def print_divider(char: str = "─", color: str = C_BLACK) -> None:
    print(f"{color}{char * _width()}{RESET}")


def print_header(title: str) -> None:
    """Большой заголовок при старте."""
    width = _width()
    border = f"{BOLD}{C_MAGENTA}{'=' * width}{RESET}"
    inner = title.center(width - 2)
    middle = f"{BOLD}{C_MAGENTA}|{RESET}{BOLD}{C_WHITE}{inner}{RESET}{BOLD}{C_MAGENTA}|{RESET}"
    print(border)
    print(middle)
    print(border)


def print_stage(stage: str, message: str, status: str = "info") -> None:
    """
    Цветной маркер этапа пайплайна.
    status: 'info' | 'ok' | 'warn' | 'error' | 'boot'

    Дизайн: [LABEL] message  — без заливки фона, яркий текст.
    """
    configs = {
        "info":  (C_BLACK,   "·"),
        "ok":    (C_GREEN,   "✓"),
        "warn":  (C_YELLOW,  "!"),
        "error": (C_RED,     "✗"),
        "boot":  (C_BLUE,    "▶"),
    }
    color, icon = configs.get(status, configs["info"])
    label = f"{stage.upper():<8}"
    badge = f"{BOLD}{color}{icon} {label}{RESET}"
    msg   = f"{color}{message}{RESET}"
    print(f"  {badge}  {msg}")


# ── Операции: STT ─────────────────────────────────────────────────────────────

def print_stt(text: str, source: str = "mic", max_chars: int = 90) -> None:
    """STT: распознанный текст (циановый)."""
    if not text or not text.strip():
        return
    snippet = text.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
    badge  = _badge("STT", C_CYAN)
    src    = f"{C_BLACK}[{source}]{RESET}"
    body   = f"{C_CYAN}{snippet}{RESET}"
    print(f"  {badge} {src} {body}")


def print_stt_done(text: str, source: str = "mic", latency_ms: int = None) -> None:
    """STT DONE: финальный текст + латентность (циановый, жирный)."""
    if not text or not text.strip():
        return
    snippet = text.strip()
    if len(snippet) > 90:
        snippet = snippet[:90].rsplit(" ", 1)[0] + "..."
    lat   = f" {C_CYAN}[{latency_ms}ms]{RESET}" if latency_ms is not None else ""
    badge = _badge("STT ✓", C_CYAN)
    src   = f"{C_BLACK}[{source}]{RESET}"
    body  = f"{BOLD}{C_CYAN}{snippet}{RESET}{lat}"
    print(f"  {badge} {src} {body}")


# ── Операции: VAD ─────────────────────────────────────────────────────────────

def print_vad_event(event: str, source: str = "mic", detail: str = "") -> None:
    """
    VAD: событие голосовой активности (жёлтый).
    event: 'speech_start' | 'speech_end' | 'silence' | 'noise'
    """
    labels = {
        "speech_start": (">> VOICE ON",  C_YELLOW),
        "speech_end":   ("[] VOICE OFF", C_YELLOW),
        "silence":      ("-- SILENCE",   C_BLACK),
        "noise":        ("~~ NOISE",     C_YELLOW),
    }
    label_text, color = labels.get(event, (event.upper(), C_YELLOW))
    badge = _badge("VAD", color)
    src   = f"{C_BLACK}[{source}]{RESET}"
    ev    = f"{BOLD}{color}{label_text}{RESET}"
    det   = f"  {C_BLACK}{detail}{RESET}" if detail else ""
    print(f"  {badge} {src} {ev}{det}")


# ── Операции: LLM (T2T) ───────────────────────────────────────────────────────

def print_llm_start(source_id: str = None, mode: str = "voice") -> None:
    """LLM: начало генерации ответа (пурпурный)."""
    src   = f" {C_BLACK}[{source_id}]{RESET}" if source_id else ""
    badge = _badge("LLM", C_MAGENTA)
    msg   = f"{C_MAGENTA}генерирую ответ...{RESET}"
    print(f"\n  {badge}{src} {msg}")


def print_llm_done(chars: int = 0, latency_ms: int = None, tps: float = None) -> None:
    """LLM DONE: статистика генерации (пурпурный, яркий)."""
    stats = []
    if latency_ms is not None:
        stats.append(f"{BOLD}{C_YELLOW}ttft {latency_ms}ms{RESET}")
    if tps is not None:
        stats.append(f"{BOLD}{C_CYAN}{tps:.1f} tok/s{RESET}")
    if chars:
        stats.append(f"{BOLD}{C_WHITE}{chars} симв.{RESET}")
    stat_str = f"  {'  '.join(stats)}" if stats else ""
    badge = _badge("LLM ✓", C_MAGENTA)
    print(f"  {badge}{stat_str}")


# ── Операции: TTS ─────────────────────────────────────────────────────────────

def print_tts_phrase(phrase: str, max_chars: int = 60) -> None:
    """TTS: фраза передана на синтез (ярко-розовый)."""
    if not phrase or not phrase.strip():
        return
    snippet = phrase.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
    badge = _badge("TTS", C_PINK)
    body  = f"{C_PINK}{snippet}{RESET}"
    print(f"  {badge} {body}")


def print_tts_done(latency_ms: int = None) -> None:
    """TTS DONE: фраза синтезирована (ярко-розовый + латентность)."""
    lat   = f"  {BOLD}{C_PINK}{latency_ms}ms{RESET}" if latency_ms is not None else ""
    badge = _badge("TTS ✓", C_PINK)
    print(f"  {badge}{lat}")


# ── Операции: BARGE-IN / прерывание ──────────────────────────────────────────

def print_interrupt(reason: str = "user_speaking", source_id: str = None) -> None:
    """BARGE-IN: прерывание генерации (красный, жирный)."""
    src   = f" {C_BLACK}[{source_id}]{RESET}" if source_id else ""
    badge = _badge("BARGE-IN", C_RED)
    msg   = f"{BOLD}{C_RED}{reason}{RESET}"
    print(f"\n  {badge}{src} {msg}")


# ── CLI диалог ────────────────────────────────────────────────────────────────

def print_user_text(text: str, name: str = "Вы") -> None:
    """Текстовое сообщение пользователя (CLI режим, зелёный)."""
    badge = _badge(name, C_GREEN)
    print(f"\n  {badge} {BOLD}{C_GREEN}{text}{RESET}")


def print_nira_start(name: str = "Нира") -> None:
    """Начало ответа Нира (пурпурный)."""
    badge = _badge(name, C_MAGENTA)
    print(f"\n  {badge} {C_MAGENTA}", end="", flush=True)


def print_nira_chunk(chunk: str) -> None:
    """Очередной чанк ответа LLM (пурпурный, без переноса строки)."""
    print(f"{C_MAGENTA}{chunk}{RESET}", end="", flush=True)


def print_nira_end() -> None:
    """Завершает строку ответа Нира."""
    print(RESET)


def print_welcome() -> None:
    """Приветственный экран при запуске CLI."""
    print()
    print_header("  ✦  N I R A  —  Console Chat  ✦  ")
    print()
    print_stage("Ввод",  "Напишите сообщение и нажмите Enter", "info")
    print_stage("Выход", "'выход', 'exit' или Ctrl+C",          "info")
    print()
    print_divider("─", C_BLACK)
    print()
