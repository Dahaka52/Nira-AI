# Анализ И План Перестройки TTS Под dffdeeq (2026-03-04)

## 1. Что сейчас в тракте TTS (фактическая карта)

Текущий путь генерации в проекте:

1. `src/utils/jaison.py`:
   - берет `t2t_result`, прогоняет через `FILTER_TEXT`,
   - на каждый текстовый chunk вызывает `OpRoles.TTS`,
   - стримит PCM чанки в WS `audio_chunk`.
2. `src/utils/operations/tts/qwen3.py`:
   - формирует payload,
   - общается с sidecar (`/v1/tts/stream`),
   - опционально автозапускает sidecar через ProcessManager.
3. `src/utils/processes/processes/qwen3_tts_server.py`:
   - поднимает `apps/tts-qwen3-server/start_server.py`.
4. `apps/tts-qwen3-server/start_server.py`:
   - грузит `qwen_tts.Qwen3TTSModel`,
   - делает stream (voice_clone/custom_voice),
   - отдает PCM через FastAPI StreamingResponse.

## 2. Что в оригинальном форке dffdeeq

Скачанный форк `apps/dffdeeq_Qwen3-TTS-streaming-main` — это **библиотечный модуль**, не сервер.

Ключевая модель использования:

1. `Qwen3TTSModel.from_pretrained(...)`
2. `create_voice_clone_prompt(...)`
3. `stream_generate_voice_clone(...)`

Важно:

- streaming API в оригинале фокусируется на `voice_clone`;
- sidecar/FastAPI транспорт в сам форк не входит;
- ускорение достигается за счет compile/stream-оптимизаций в самом `qwen_tts` runtime.

## 3. Где были расхождения и ошибки относительно оригинала

1. Источник пакета `qwen-tts` был нефиксирован к dffdeeq-оригиналу:
   - sidecar env указывал на другой форк (`Qwen3-TTS-Openai-Fastapi`).
2. Не было явного режима совместимости runtime:
   - один и тот же код пытался работать одновременно в «расширенном» и «оригинальном» API-профиле.
3. Не было жесткой привязки к локальному checkout форка:
   - sidecar мог импортировать любой `qwen_tts`, установленный в venv.
4. В warmup sidecar использовался неудачный `max_frames` (привязан к warmup token budget, а не к runtime frame budget), что искажало warm-state профиль.
5. Локальная старая копия форка и часть runtime-логики опирались на `stream_generate_custom_voice`, а в свежем оригинальном `dffdeeq` этот метод отсутствует:
   - это ключевая причина «скрытых fallback-веток» и несопоставимого поведения при A/B.

## 4. Что уже отрефакторено в этом проходе

### 4.1 Явный runtime flavor

Добавлен параметр `runtime_flavor` (значения: `auto | dffdeeq | extended`) по всей цепочке:

- `src/utils/operations/tts/qwen3.py`
- `src/utils/processes/processes/qwen3_tts_server.py`
- `apps/tts-qwen3-server/start_server.py`

Эффект:

- можно строго запускать sidecar в режиме оригинального `dffdeeq` (`runtime_flavor=dffdeeq`),
- можно осознанно включать расширенный режим (`extended`) под нестандартные API форки.

### 4.2 Жесткое подключение локального форка

Добавлен `qwen_tts_repo_path` в sidecar runtime/config.

Эффект:

- sidecar может импортировать `qwen_tts` из `C:\Nirmita\apps\dffdeeq_Qwen3-TTS-streaming-main`,
- исключается «случайный» импорт несовместимой версии пакета.

### 4.3 Синхронизация окружения sidecar

`apps/tts-qwen3-server/requirements.sidecar.txt` переключен на локальный editable-install:

- `-e ../dffdeeq_Qwen3-TTS-streaming-main`

### 4.4 Фикс warmup-профиля

В sidecar warmup поправлен `max_frames` (берется из runtime frame budget, а не из warmup token budget).

## 5. Целевая архитектура (рекомендуемая)

### Вариант A (рекомендован): строгая совместимость с dffdeeq

1. `runtime_flavor = dffdeeq`
2. `voice_mode = voice_clone`
3. `qwen_tts_repo_path` указывает на локальный dffdeeq checkout
4. sidecar использует только поток `stream_generate_voice_clone` как primary-path

Плюсы:

- минимальная неопределенность,
- чистое A/B сравнение с оригиналом,
- проще дебажить производительность.

### Вариант B: полное встраивание форка как внутреннего модуля

1. держать исходники dffdeeq в `apps/dffdeeq_Qwen3-TTS-streaming-main` как vendored source,
2. sidecar всегда импортирует именно этот путь,
3. обновления форка делать только через явный pull+diff+retest.

Плюсы:

- детерминированный runtime,
- контроль версий в репозитории.

Минусы:

- более тяжелая поддержка при обновлениях upstream.

## 6. Пошаговый план перестройки до production-state

## Этап 1. Зафиксировать «чистый dffdeeq профиль»

1. В конфиге оставить:
   - `runtime_flavor: dffdeeq`
   - `voice_mode: voice_clone`
   - `process.qwen_tts_repo_path: C:\Nirmita\apps\dffdeeq_Qwen3-TTS-streaming-main`
2. Пересобрать sidecar venv (`install_sidecar.ps1`).
3. Проверить `/health`:
   - что `qwen_tts_module_file` указывает на локальный dffdeeq путь,
   - что capability `stream_generate_voice_clone=true`.

Критерий: runtime реально работает на нужном форке.

## Этап 2. Снять baseline в live-пайплайне

1. Прогон фиксированного набора фраз (короткие/средние/длинные).
2. Зафиксировать:
   - `first_chunk_ms`
   - `max_gap_ms`
   - `rtf`
   - частоту `Rebuffering`

Критерий: есть объективная точка сравнения после рефакторинга.

## Этап 3. Дожать throughput без смешивания API-режимов

1. Тюнинг только параметров dffdeeq-профиля:
   - `emit_every_frames`
   - `decode_window_frames`
   - `max_frames`
   - `use_optimized_decode`
2. Отдельный прогон compile-профиля (без смешения с extended runtime).

Критерий: стабильный warm-run без разрывов и без API fallback-сюрпризов.

## Этап 4. Решение по стратегии долгосрочной поддержки

1. Если A-стратегия стабильна и достаточно быстрая: оставляем strict dffdeeq.
2. Если нужен custom_voice streaming: вводим отдельный профиль `extended` и отдельный SLA/метрики для него.

Критерий: в одном боевом профиле не смешивать разные форки и разные контракты API.

## 7. Что считать успешным результатом

Минимальные KPI для «голос не рвется»:

- `first_chunk_ms <= 900ms` (warm-run, короткие реплики),
- `max_gap_ms <= 250..300ms` (короткие/средние реплики),
- `rtf <= 1.0..1.1` на целевых репликах,
- отсутствие длительных rebuffer-серий в live-диалоге.

## 8. Риски и контроль

1. Риск: sidecar подтянет не тот `qwen_tts`.
   - Контроль: проверка `qwen_tts_module_file` в `/health`.
2. Риск: случайный переход в mixed API-path.
   - Контроль: фиксировать `runtime_flavor=dffdeeq` в боевом профиле.
3. Риск: регресс после обновления форка.
   - Контроль: обновлять форк только через diff + тот же benchmark-набор.
