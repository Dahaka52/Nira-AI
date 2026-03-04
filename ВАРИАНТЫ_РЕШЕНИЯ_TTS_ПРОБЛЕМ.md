# Отчёт по TTS для внешнего аудита

Документ собран из текущего состояния репозитория, логов живых прогонов и анализа форков/источников.

## 1) Полная текущая реализация TTS и подключение к backend

### 1.1 Тракт в проекте (фактический)
- Backend: `JAIson` (`src/utils/jaison.py`) с pipeline `STT -> T2T -> FILTER_TEXT -> TTS -> WS events`.
- Активный TTS в конфиге: `tts_active_id: "qwen3"` в [config.yaml](C:/Nirmita/configs/config.yaml).
- TTS operation: [qwen3.py](C:/Nirmita/src/utils/operations/tts/qwen3.py).
- Sidecar process manager: [qwen3_tts_server.py](C:/Nirmita/src/utils/processes/processes/qwen3_tts_server.py).
- Sidecar server: [start_server.py](C:/Nirmita/apps/tts-qwen3-server/start_server.py).
- Realtime output client: [main.py](C:/Nirmita/apps/hw-audio-out-client/main.py).

### 1.2 Как идёт поток данных
1. T2T генерирует текст (llama.cpp).
2. `filter_clean` + `chunker_sentence` (сейчас `mode: full`) формируют chunk для TTS.
3. `qwen3.py` отправляет HTTP stream POST в sidecar: `http://127.0.0.1:6116/v1/tts/stream`.
4. Sidecar стримит PCM16 bytes.
5. Backend отправляет WS событие `response`/`audio_chunk` (base64).
6. `hw-audio-out-client` получает WS, декодирует PCM, при необходимости ресемплит под устройство вывода (у тебя обычно fallback 24000 -> 48000), пишет в `CABLE Input (VB-Audio Virtual Cable)`.

### 1.3 Текущие параметры (важные)
- Модель: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
- Язык: `russian`, спикер `serena`.
- Профиль стриминга:
  - `emit_every_frames: 8`
  - `decode_window_frames: 80`
  - `first_chunk_emit_every: 5`
  - `first_chunk_decode_window: 48`
  - `first_chunk_frames: 48`
  - `overlap_samples: 0`
- Стабилизация тембра:
  - `do_sample: false`
- Sidecar compile:
  - `use_compile: 1`
  - `compile_mode: reduce-overhead`
  - `compile_use_cuda_graphs: 0`
  - `compile_codebook_predictor: 0`
- Warmup:
  - `preload_on_start: 1`
  - `warmup_on_start: 1`
  - `warmup_text: "Привет."`

## 2) Предпринятые действия и решения (что уже делалось)

### 2.1 По интеграции и инфраструктуре
- Подключен sidecar TTS процесс с автозапуском через ProcessManager.
- Добавлен явный выбор CPU/GPU + `gpu_id` + `device` в конфиге.
- Добавлена проверка/подхват SoX bin directory.
- Добавлен hw audio output клиент для вывода в VB-Cable/Voicemeeter (UI звук не воспроизводит).

### 2.2 По latency/quality
- Переключен `chunker_sentence` в `mode: full`.
- Добавлены preload/warmup sidecar.
- Добавлена диагностика TTS метрик и smoke-инструмент [tts_smoke.py](C:/Nirmita/src/tools/tts_smoke.py).
- Тюнились параметры стриминга (emit/decode/first chunk/overlap).
- `do_sample` переведен в `false` для уменьшения плавающего тона.

### 2.3 По устойчивости/ошибкам
- Исправлена «тихая» деградация worker-а sidecar: теперь пишет traceback.
- Добавлен контроль кейса `HTTP 200, но 0 audio bytes` (считается ошибкой).
- Добавлен controlled retry на стороне `qwen3.py` при фейле до первого чанка.

## 3) Обнаруженные проблемы и предполагаемые причины

### 3.1 Наблюдаемые симптомы (пользовательские)
- Рваная речь: «слово -> пауза -> слово».
- Плавающий тон (то выше, то ниже).
- Редкие зависания/деградация тембра до «скрипа».
- Периодические полные обрывы TTS запроса.

### 3.2 Зафиксированные технические ошибки
- `httpx.ConnectError: All connection attempts failed` в backend, когда sidecar перестаёт отвечать.
- `ConnectionResetError(10054)` в `tts_smoke.py`.
- Критично: в логах sidecar пойман traceback:
  - `AssertionError` внутри `torch._inductor.cudagraph_trees.py` (`torch._C._is_key_in_tls`)
  - стэк из `qwen_tts ... decode_streaming -> torch.compile/inductor`.
- В sidecar была warning про dynamic cudagraph формы:
  - «observed N distinct sizes... consider padding / skip dynamic graphs».

### 3.3 Предполагаемые причины (гипотезы)
1. Нестабильность compile/cudagraph пути на Windows при динамических размерах входа в streaming decode.
2. Дропауты из-за комбинации маленьких chunk + realtime-resample 24k->48k на output-клиенте.
3. Дополнительная «рваность» из-за barge-in/частых прерываний при новом speech_start.
4. Отсутствие flash-attn в текущем runtime ухудшает скорость и может усиливать «ступенчатость» подачи chunk.

## 4) Пути решения из форков/источников, которые ещё не опробованы

### 4.1 Python 3.12 sidecar стек (высокий приоритет)
- В форках для Windows рекомендован `Python 3.12` + `torch cu130` + `flash_attn` wheel + `triton-windows`.
- Что уже начато:
  - создано `apps/tts-qwen3-server/.venv312`
  - в нём уже стоят `torch 2.10.0+cu130`, `torchaudio 2.10.0+cu130`
- Что ещё НЕ сделано:
  - поставить `flash_attn` wheel (cp312, win)
  - поставить `triton-windows<3.7`
  - поставить `qwen-tts` в `.venv312`
  - переключить `process.python_executable` на `.venv312` и прогнать A/B метрики

### 4.2 Идеи из reku форка (не полностью внедрены)
- Доработанный streaming-код:
  - Hann crossfade на стыках
  - расширенный EOS detection (против runaway)
  - streaming repetition penalty
- В текущем коде есть часть параметров, но полный набор патчей форка не перенесён.

### 4.3 Идеи из groxaxo backend (не внедрены в полном объёме)
- Optimized backend паттерны:
  - GPU keepalive
  - более структурированный warmup/compile lifecycle
  - голосовые кэши и более продвинутая backend-оркестрация

### 4.4 Дополнительные инженерные шаги (ещё не тестировались)
- Жёсткая нормализация размеров decode-window входа (padding to fixed shapes), чтобы снизить динамику для cudagraph.
- Временное отключение `torch.compile` полностью в runtime 3.10 и сравнение стабильности vs latency.
- Увеличение output буфера (`audio_output.max_buffer_ms`) для сглаживания выпадений.
- Жёсткая унификация sample rate в тракте, чтобы минимизировать runtime-resample.

### 4.5 План B: альтернативный TTS модуль
- Если Qwen3 после миграции на 3.12 и стабилизации compile всё ещё нестабилен:
  - добавить второй TTS operation (Fish Speech / иной streaming TTS),
  - переключать через `tts_active_id`, без переписывания pipeline.

## 5) Просмотренные источники

### 5.1 Репозитории / документация
- Qwen official: https://github.com/QwenLM/Qwen3-TTS
- dffdeeq fork: https://github.com/dffdeeq/Qwen3-TTS-streaming
- reku fork: https://github.com/rekuenkdr/Qwen3-TTS-streaming
- groxaxo backend: https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi

### 5.2 Локальные артефакты (ключевые для аудита)
- Конфиг: [config.yaml](C:/Nirmita/configs/config.yaml)
- Sidecar код: [start_server.py](C:/Nirmita/apps/tts-qwen3-server/start_server.py)
- TTS operation: [qwen3.py](C:/Nirmita/src/utils/operations/tts/qwen3.py)
- Process wrapper: [qwen3_tts_server.py](C:/Nirmita/src/utils/processes/processes/qwen3_tts_server.py)
- Audio output client: [main.py](C:/Nirmita/apps/hw-audio-out-client/main.py)
- Лог sidecar: `logs/qwen3_tts_server.log` (есть traceback с `AssertionError` в cudagraph path)
- Smoke tool: [tts_smoke.py](C:/Nirmita/src/tools/tts_smoke.py)
