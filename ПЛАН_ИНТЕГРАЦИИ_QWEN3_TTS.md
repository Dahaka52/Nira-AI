# План интеграции Qwen3-TTS Streaming в Nira (как сменный TTS-модуль)

Дата фиксации контекста: **2026-03-05**

## 1) Что уже подтверждено по текущему стеку

- Проект Nira/JAIson использует модульный pipeline `operations` и поддерживает заменяемые роли (`stt`, `t2t`, `tts`, `filter_*`).
- Текущая сборка Nira: в `src/utils/operations/tts` есть только `base.py`; рабочие TTS-реализации отключены/отсутствуют.
- В `OperationManager` для `OpTypes.TTS` сейчас всегда `UnknownOpID` (то есть TTS-провайдеры не подключены кодом).
- В runtime-пайплайне `response_pipeline` TTS вызывается потоково: `filter_text -> tts -> filter_audio -> websocket`.
- Текущая среда Nira (локально в `.conda`):
  - Python `3.10.19`
  - torch `2.10.0+cu128`
  - transformers `4.52.4`
  - `sox` (python package) отсутствует
  - `accelerate` отсутствует
  - `flash_attn` отсутствует
- Форк `apps/Qwen3-TTS-streaming-main`:
  - добавляет streaming API (`stream_generate_pcm`, `stream_generate_voice_clone`);
  - в `pyproject.toml` фиксирует `transformers==4.57.3`, требует `sox`, `onnxruntime`, `torchaudio`, и т.д.;
  - в README рекомендует отдельное окружение Python 3.12 (особенно для flash-attn на Windows).

## 2) Ключевые конфликты между вашим стеком и форком Qwen3-TTS

## Конфликт A: `transformers` версия
- В Nira зафиксировано `transformers==4.52.4`.
- Форк Qwen3-TTS требует `transformers==4.57.3` и использует модули, которых нет в 4.52.4 (например `transformers.masking_utils`).
- Риск: прямой импорт Qwen в текущем env ломается/или требует апгрейд, который может затронуть MeloTTS и другие модули.

## Конфликт B: отсутствие `sox`
- Импорт форка падает на `ModuleNotFoundError: sox`.
- Дополнительно нужен установленный SoX binary в PATH (по README форка).

## Конфликт C: различия runtime-окружения и CUDA-стека
- Форк ориентируется на отдельную сборку (Python 3.12, CUDA 13.0 wheel для flash-attn на Windows).
- У вас рабочий стек на Python 3.10 + cu128 для текущей системы.
- Риск: попытка “впихнуть всё в один env” может destabilize текущий production-стек.

## Конфликт D: архитектурный (sync генерация внутри async core)
- Qwen streaming API синхронный и тяжёлый (GPU loop).
- Если вызвать напрямую в async operation без изоляции, можно блокировать event loop backend.

## Конфликт E: формат аудио
- Qwen отдает float32 waveform chunks.
- Nira TTS pipeline ожидает `audio_bytes` (PCM bytes) + `sr/sw/ch`.
- Нужна конвертация float32 -> PCM16 bytes и строгая нормализация формата.

## Конфликт F: управление ресурсами GPU
- У вас уже активен LLM (`llama-server`) и STT-процессы.
- Qwen 1.7B может конфликтовать по VRAM/latency без жесткого pinning GPU и стратегии warmup.

## 3) Рекомендуемая стратегия

Рекомендуется **изолированный sidecar для Qwen3-TTS** (отдельный conda/env + локальный HTTP/WS сервис), а в Nira добавить легковесный `TTSOperation`-клиент.

Почему это лучший вариант:
- не ломает текущий стабильный env Nira;
- убирает конфликт `transformers` и зависимостей;
- позволяет независимо апгрейдить Qwen-форк;
- проще rollback (отключили один TTS-op в конфиге).

## 4) Пошаговый план реализации

## Этап 0. Подготовка и freeze
1. Зафиксировать текущий рабочий стек Nira:
   - сохранить `pip freeze` текущего `.conda` в `docs/env/nira-freeze-2026-03-05.txt`.
2. Создать отдельное окружение под Qwen sidecar (рекомендуемо `python=3.12`).
3. Проверить GPU-распределение:
   - определить, на каком GPU запускать Qwen sidecar;
   - зарезервировать GPU policy (например LLM=0, Qwen=1 если доступно).

## Этап 1. Поднять Qwen sidecar окружение
1. В отдельном env установить зависимости форка по README.
2. Установить SoX:
   - binary + python package `sox`.
3. Провести smoke-import:
   - `import qwen_tts` должен проходить без ошибок.
4. Скачать/подготовить модель(и) Qwen, кешировать в выделенном каталоге.
5. Провести standalone smoke:
   - `stream_generate_voice_clone` возвращает чанки, измерить first chunk latency.

## Этап 2. Реализовать sidecar-сервис для Qwen
1. В `apps/` добавить новый сервис (например `apps/qwen-tts-sidecar`):
   - endpoint `/health`;
   - endpoint `/tts/stream` (SSE/WS/chunked HTTP);
   - endpoint `/tts/generate` (нестриминговый fallback).
2. В sidecar реализовать:
   - singleton загрузку модели;
   - prewarm/warmup;
   - параметры `emit_every_frames`, `decode_window_frames`;
   - timeout + graceful error schema.
3. Нормализовать формат ответа sidecar:
   - chunk payload: PCM16 bytes (или base64), `sr=24000`, `sw=2`, `ch=1`.

## Этап 3. Интеграция в Nira как сменный TTS-модуль
1. Добавить TTS-провайдер в core:
   - новый класс `src/utils/operations/tts/qwen_sidecar.py` (наследник `TTSOperation`);
   - `configure/start/close/_generate`.
2. Добавить загрузку TTS-op в `src/utils/operations/manager.py`:
   - кейс `OpTypes.TTS` для `id: qwen_sidecar`.
3. (Опционально, лучше) сделать `tts` registry по аналогии с STT (`entrypoint`), чтобы провайдеры TTS подключались без hardcode.
4. Добавить поля конфигурации провайдера в YAML-операцию:
   - `base_url`, `timeout_s`, `voice_mode`, `language`, `stream`, `emit_every_frames`, `decode_window_frames`, `max_text_len`, `retry`.
5. Реализовать конвертацию аудио:
   - если sidecar отдаёт float32: в Nira конвертировать в PCM16 bytes;
   - вернуть обязательные поля `audio_bytes/sr/sw/ch`.

## Этап 4. Процесс-менеджмент (опционально, но рекомендуется)
1. Добавить новый `ProcessType` (например `QWEN_TTS`) в `ProcessManager`.
2. Реализовать `src/utils/processes/processes/qwen_tts_sidecar.py`:
   - запуск/остановка sidecar;
   - логирование в `logs/qwen_tts_sidecar.log`;
   - health-check и restart policy.
3. В `qwen_sidecar` operation использовать link/unlink по аналогии с STT runner.

## Этап 5. Конфигурация как сменного TTS
1. В `configs/config.yaml` добавить TTS блок:
   - `role: tts`
   - `id: qwen_sidecar`
   - параметры sidecar.
2. Сделать второй fallback TTS блок (например `pytts`/`melo`, когда вернете реализацию), чтобы быстро откатываться.
3. Добавить короткую инструкцию switch-профилей:
   - отдельные config файлы (`config_qwen.yaml`, `config_safe.yaml`) или `api/config/load`.

## Этап 6. Тесты и валидация
1. Unit:
   - тест конвертации float32 -> PCM16.
   - тест `_parse_chunk` / валидация пустого текста.
2. Integration:
   - `operation_use` для `role=tts,id=qwen_sidecar`.
   - `response_pipeline` с `include_audio=true`, проверка websocket `audio_chunk`.
3. Load/latency:
   - TTFT (text) и TTS-start до/после интеграции;
   - стабильность при длинных репликах.
4. Failure scenarios:
   - sidecar недоступен;
   - timeout;
   - OOM/restart;
   - graceful деградация (ответ без аудио + системный event).

## Этап 7. Документация и эксплуатация
1. Обновить `DEVELOPER.md` раздел TTS (новый провайдер `qwen_sidecar`).
2. Добавить runbook:
   - как поднять env sidecar;
   - как переключить TTS;
   - как читать логи и выполнять rollback.
3. Добавить чеклист релиза.

## 5) Варианты решения конфликтов

## Вариант 1 (рекомендуемый): Изолированный sidecar
- Плюсы:
  - минимальный риск сломать основной Nira env;
  - независимые версии `transformers/torch/flash-attn`;
  - лучший rollback.
- Минусы:
  - нужно поддерживать второй процесс/окружение.

## Вариант 2: Интеграция Qwen прямо в текущий `.conda`
- Что нужно:
  - поднять `transformers` до 4.57.3;
  - установить `sox`, `accelerate`, возможно flash-attn/triton.
- Риски:
  - поломка совместимости с текущими TTS/ML модулями;
  - сложный rollback.
- Использовать только если принципиально нужен single-process runtime.

## Вариант 3: Гибрид
- Основной путь sidecar, но fallback TTS локально в том же Nira.
- Лучший баланс отказоустойчивости: если sidecar умер, pipeline не падает полностью.

## 6) Конкретные меры по каждому конфликту

- A (`transformers`): изоляция окружений; не трогать pinned версию Nira.
- B (`sox`): установить и python пакет, и бинарник SoX; добавить startup self-check.
- C (Python/CUDA): Qwen env отдельно; не смешивать с рабочим env Nira.
- D (async blocking): вынос тяжёлого inference в sidecar (или `asyncio.to_thread` как временный костыль).
- E (audio format): стандартизировать выход sidecar в PCM16 mono 24k + явные `sr/sw/ch`.
- F (GPU contention): hard pinning GPU + warmup + лимиты очереди запросов.

## 7) Минимальный MVP (самый быстрый путь)

1. Поднять Qwen в отдельном env и сделать простой local HTTP sidecar.
2. Добавить один `qwen_sidecar` TTSOperation-клиент в Nira.
3. Подключить через `operations` в `config.yaml`.
4. Проверить end-to-end: `response` job -> `audio_chunk` в WS.
5. Добавить fallback на случай недоступности sidecar.

## 8) Критерии готовности (Definition of Done)

- TTS `id: qwen_sidecar` грузится через `/api/operations/reload` без ошибок.
- В `response`-джобе приходят потоковые `audio_chunk` события.
- При падении sidecar backend не падает и возвращает контролируемую ошибку/деградацию.
- Есть документированный rollback на fallback TTS.
- Измерены и сохранены метрики `tts_start_ms` и стабильность под реальной нагрузкой.

## 9) Rollback план

1. Переключить config на fallback TTS-провайдер (или отключить TTS временно).
2. Выполнить `/api/operations/reload`.
3. Остановить Qwen sidecar и освободить GPU.
4. Проверить, что `response` снова стабилен (минимум text-only режим).

