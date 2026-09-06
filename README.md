# Nira (Нира)

Асинхронный модульный бэкенд локального AI vTuber / голосового персонажа для общения в реальном времени через локальный микрофон и голосовые каналы Discord.

---

## Архитектура и стек

- **STT (Speech-to-Text):** Локальный оффлайн-движок `GigaAM-v3 NeMo CTC` (int8 ONNX через `sherpa-onnx`) на CPU (<50 мс задержка). Встроенная программная AGC-нормализация громкости и словарь омофонов `alias_map`. Резерв: облачный `Whisper Large v3 Turbo` через Groq.
- **T2T (LLM):** Локальный инференс через `llama-server` (RTX 5070 12GB, архитектура Blackwell `sm_120`, FlashAttention, контекст 8192–16384 токенов). Поддержка внешних OpenAI-совместимых API (Groq, Google AI Studio).
- **TTS (Text-to-Speech):** Потоковый синтез речи `Fish Audio` (модель S2.1-pro) через WebSocket (`wss://api.fish.audio/v1/tts/live`) с MessagePack и прямым выводом PCM 44.1 kHz.
- **Barge-in и фильтрация речи:** Селективное раннее прерывание по VAD (`POST /speech_start`), фильтрация речевых филлеров («м-м-м», «эээ»), адаптивный дебаунс тишины (750 мс на вопросах, 1400 мс при повествовании), режим активного слушателя.
- **Discord Voice Bridge:** Автономный клиент на базе `py-cord` с поддержкой сквозного шифрования DAVE E2EE и двусторонней потоковой передачей звука.
- **Аппаратный микрофон:** `apps/hw-mic-client` на `sounddevice` (WASAPI Exclusive) с ресемплингом 48 kHz ➔ 16 kHz и встроенным Silero VAD.
- **Web UI:** Дашборд `apps/nira-web` (React + TypeScript + Vite) с телеметрией системы (CPU, RAM, GPU), очередью задач и WebSocket Event Bus (`ws://127.0.0.1:7272/`).

---

## Аппаратная конфигурация

| Компонент | Спецификация | Назначение |
| :--- | :--- | :--- |
| **CPU** | AMD Ryzen 7 5700X3D (8C/16T) | STT GigaAM, Quart/Hypercorn, VAD, очереди и буферы |
| **GPU** | NVIDIA GeForce RTX 5070 12GB (sm_120) | 100% VRAM выделено под локальную 12B LLM |
| **RAM** | 32 GB DDR4-3600 | Кэш контекста, база векторов, аудиобуферы |
| **ОС** | Windows 11 64-bit | PowerShell 7 (pwsh), WASAPI |

---

## Установка и запуск

### 1. Клонирование и настройка окружения
```powershell
git clone https://github.com/Dahaka52/Nira-AI.git
cd Nira-AI

# Установка Conda окружения (Python 3.12 + PyTorch с поддержкой CUDA)
.\setup_env_py312_cu130.ps1
```

### 2. Настройка переменных окружения
Создайте файл `nira.env` на основе шаблона:
```powershell
Copy-Item .env-template nira.env
```

Основные переменные в `nira.env`:
- `FISH_API_KEY` — ключ API Fish Audio для синтеза речи.
- `DISCORD_BOT_TOKEN` — токен бота Discord (при использовании Discord моста).
- `GROQ_API_KEY` — ключ Groq (опционально, для облачного T2T/STT).
- `GEMINI_API_KEY` — ключ Google AI Studio (опционально).

### 3. Запуск
```powershell
# Основной скрипт запуска (Backend + Web Dashboard)
.\start_nira.ps1
```
Или через батник: `start_nira.bat`.

После старта доступны:
- REST API: `http://127.0.0.1:7272`
- WebSocket Event Bus: `ws://127.0.0.1:7272/`
- Web Dashboard: `http://localhost:3000`

---

## Структура репозитория

```
├── apps/
│   ├── discord-bridge/     # Мост для голосовых каналов Discord (DAVE E2EE)
│   ├── hw-mic-client/      # Клиент захвата локального микрофона (WASAPI)
│   └── nira-web/           # Веб-интерфейс дашборда (React, Vite)
├── configs/
│   ├── config.yaml         # Главная конфигурация активных модулей и параметров
│   └── known_users.json    # Профили известных спикеров (Создатель, семья, гости)
├── models/                 # Локальные модели (ONNX веса GigaAM, токены, hotwords)
├── prompts/
│   ├── characters/         # Личность и характер Ниры (nira.md)
│   ├── instructions/       # Системные мета-инструкции и языковые ограничения
│   └── scenes/             # Сценарии окружения (local, discord)
├── src/
│   ├── main.py             # Точка входа бэкенда
│   ├── memory/             # Подсистема памяти (store, retrieval, consolidation, identity)
│   └── utils/
│       ├── nira.py         # Центральный диспетчер ядра (Singleton Nira)
│       ├── prompter/       # Сборка системного контекста и управление историей
│       ├── operations/     # Модули STT, T2T, TTS и аудио-фильтров
│       └── server/         # Quart REST API и WebSocket сервер
├── start_nira.ps1          # Главный PowerShell-скрипт запуска
└── docs/                   # Полные архитектурные спецификации
```

---

## Документация

- [АРХИТЕКТУРА проекта Нира.md](АРХИТЕКТУРА%20проекта%20Нира.md) — исчерпывающее техническое руководство по всей кодовой базе и протоколам.
- [ПЛАН_РАЗВИТИЯ.md](ПЛАН_РАЗВИТИЯ.md) — стратегический роадмап (память, эмоции, осознанность, Live2D/VTuber).
- [Разработка архитектуры эмоционального интеллекта.md](Разработка%20архитектуры%20эмоционального%20интеллекта.md) — проектирование аффективного слоя.
