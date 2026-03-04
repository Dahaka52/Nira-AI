# План внедрения TTS как отдельного подключаемого модуля (v2)

Дата актуализации: 2026-03-04  
Проект: `C:\Nirmita`

## 1. Цель и рамки

Цель этого этапа: внедрить Qwen3-TTS в Nira как полностью модульный TTS-провайдер, который:

1. Подключается/переключается через конфиг без правок `jaison.py`.
2. Даёт реальный streaming аудио чанков в текущий WS-пайплайн.
3. Стабильно работает на Windows 11 и RTX 5060 с VRAM бюджетом 4-6 ГБ.
4. Поддерживает русский язык и эмоциональные инструкции.
5. Имеет производственный контур под Discord/OBS (виртуальный кабель).

В этом документе зафиксированы архитектурное решение, зависимости и детальный пошаговый план.

## 2. Привязка к целям из шапки АРХИТЕКТУРА

1. Устойчивый характер: TTS должен быть консистентным по голосу и темпу, без артефактов между чанками.
2. Осознание себя в моменте: потоковый звук должен синхронно идти с response-событиями и interrupt-политикой.
3. Развитие личности: TTS метрики/логи нужны для self-tuning профилей голоса.
4. Продвинутая эмоциональность: эмоции из `emotion` поля и скобок в тексте должны влиять на просодию.
5. Память: стабильная озвучка длинных ответов без дрейфа голоса.
6. Смекалка/логика: минимальная латентность до первого звука, чтобы диалог ощущался живым.
7. Социальные навыки: естественные короткие реплики и backchannel без долгих пауз.
8. Мульти-люди/стрим: поддержка разных источников вывода (Discord, OBS, виртуальный микрофон) без воспроизведения в Web UI.
9. Дообучение: единый лог TTS-параметров и latency для последующего анализа.
10. Voice ID: архитектура не должна блокировать будущую привязку голоса к speaker_id.
11. Discord/игры: отдельный audio output adapter поверх ядра пайплайна.

## 3. Что уже готово в Nira

1. `response_pipeline()` уже поддерживает `include_audio` и поток `audio_chunk` событий.
2. Метрики `tts_start_ms` и `e2e_tts_start_ms` уже прокинуты в WS/UI.
3. Базовый контракт TTS операции стандартизирован: вход `content`, выход `audio_bytes/sr/sw/ch`.
4. В `OperationManager` роль `tts` существует, но реестр провайдеров пока пустой.
5. Web UI не воспроизводит `audio_chunk` и в целевой архитектуре не должен этого делать; UI только для текста/метрик.

Вывод: ядро готово к подключению TTS, но нужен провайдерный слой, реестр и контур эксплуатации.

## 4. Сводка внешнего исследования (репозитории)

Проверены:

1. `dffdeeq/Qwen3-TTS-streaming`.
2. `rekuenkdr/Qwen3-TTS-streaming`.
3. `groxaxo/Qwen3-TTS-Openai-Fastapi`.
4. Официальный `QwenLM/Qwen3-TTS`.

### 4.1 Ключевые выводы по форкам

1. `rekuenkdr` даёт лучший streaming-контур из рассмотренных:
   1. Two-phase first chunk (`first_chunk_*`) для снижения latency первого звука.
   2. Расширенная EOS-логика и анти-loop `repetition_penalty` для стабильности длинного стрима.
   3. Batch streaming API (не обязателен на старте, но полезен для будущих multi-channel сценариев).
2. `dffdeeq` это базовый фундамент с реальным streaming и ускорениями, плюс явные заметки по Windows-зависимостям.
3. `groxaxo` полезен как production reference:
   1. OpenAI-compatible API/streaming.
   2. Конкурентность через лимит (`TTS_MAX_CONCURRENT`).
   3. Кэширование voice prompts.
   4. Warmup/compile паттерны для снижения p95 latency.
4. Официальный `QwenLM/Qwen3-TTS` подтверждает:
   1. 0.6B/1.7B модели.
   2. Поддержку русского и эмоционального управления через инструкции.
   3. Streaming capability.
   4. vLLM-Omni пока с ограничением online serving (не брать как основной runtime для Nira сейчас).

### 4.2 Принятое решение

Основной движок: `rekuenkdr` (как upstream для streaming-инференса)  
Технические паттерны API/эксплуатации: брать из `groxaxo`  
Fallback: `dffdeeq`/official path (если конкретная оптимизация `rekuenkdr` конфликтует с Win11/CUDA стеком).

## 5. Целевая архитектура TTS для Nira

## 5.1 Принцип модульности

1. `jaison.py` не знает ничего о конкретном TTS-провайдере.
2. Выбор TTS только через конфиг (как для STT), без переписывания пайплайна.
3. Добавление нового TTS = новый модуль + запись в `operations`.

## 5.2 Стандартизованный контракт TTS

Вход:

```json
{
  "content": "Текст для озвучки",
  "emotion": "optional",
  "source_id": "optional",
  "turn_id": "optional",
  "utterance_id": "optional"
}
```

Выход (stream):

```json
{
  "audio_bytes": "bytes pcm_s16le",
  "sr": 24000,
  "sw": 2,
  "ch": 1
}
```

Требования:

1. Чанки должны идти монотонно и без смены формата внутри одного ответа.
2. Генерация не блокирует event loop.
3. Первый аудио чанк должен корректно инициировать `tts_start_ms`.

## 5.3 Рекомендуемая схема интеграции

1. Ввести `tts.registry` по аналогии с `stt.registry`:
   1. built-in mapping.
   2. `entrypoint: "module.path:ClassName"` для динамического подключения.
2. Ввести `tts_active_id` и `tts_strict_active_id` в конфиг.
3. Реализовать `Qwen3TTSOperation` как адаптер с двумя режимами:
   1. `mode: local` (in-process, проще запуск).
   2. `mode: sidecar` (рекомендуется для изоляции зависимостей и стабильности).
4. Для `mode: sidecar` использовать отдельный процесс `qwen3_tts_server` (по паттерну sherpa sidecar).

## 5.4 Эмоции и стиль речи

1. Источник эмоций:
   1. `emotion` поле из `emotion_roberta`.
   2. Маркеры в скобках в тексте (`(радостно)`, `(спокойно)`).
2. Маппинг эмоций:
   1. `joy` -> `instruct="Говори радостно, тепло, с лёгкой улыбкой"`.
   2. `sadness` -> `instruct="Говори мягко, тише, с грустной интонацией"`.
   3. `anger` -> `instruct="Говори напряженно, но без крика"`.
3. При конфликте источников приоритет: явные скобки > `emotion` > neutral.

## 5.5 Вывод в Discord/OBS

1. Не вшивать вывод в `JAIson` ядро.
2. Ввести отдельный output-адаптер `audio_router`:
   1. вход: `audio_chunk` (PCM).
   2. выход: выбранный аудио девайс (VB-Cable/Voicemeeter/обычный выход).
3. Управление маршрутом через конфиг (`audio_output.enabled`, `device_name`, `source_bus`).

## 6. Зависимости и профили под Windows 11 + RTX 5060

## 6.1 Базовый профиль (рекомендуется для старта)

1. Python: оставить текущий backend env (`3.10.x`) для Nira ядра.
2. TTS runtime: отдельный sidecar env (`3.12`) для Qwen3-TTS-streaming.
3. GPU: закрепить TTS за `cuda:1` (RTX 5060), LLM оставить на `cuda:0` (RTX 5070).
4. Модель старта: `Qwen3-TTS-12Hz-0.6B-CustomVoice`.
5. Dtype: `bfloat16` (fallback `float16`).
6. Attention: `flash_attention_2` если wheel доступен; иначе `sdpa`.

## 6.2 Почему sidecar для этого проекта

1. Изоляция тяжелых TTS-зависимостей от backend, где уже большой стек.
2. Возможность жить на Python 3.12 для TTS без миграции всего проекта.
3. Быстрый rollback: выключается `role: tts`, backend продолжает работу.

## 6.3 Пакеты (минимальный target)

Для sidecar env:

1. `torch`, `torchaudio` под CUDA, совместимые с установленным драйвером.
2. `qwen-tts` из выбранного форка (`rekuenkdr` pinned commit/tag).
3. `fastapi`, `uvicorn`, `orjson`, `httpx`.
4. `soundfile`, `librosa`, `einops`, `accelerate`, `transformers`.
5. Опционально: `flash-attn`, `triton-windows` (если сборка/колесо доступны для конкретной связки Python/CUDA/Torch).

## 6.4 Профили качества/скорости

1. `low_latency`:
   1. `first_chunk_emit_every=5`.
   2. `first_chunk_decode_window=48`.
   3. `first_chunk_frames=48`.
   4. `emit_every_frames=10`.
   5. `decode_window_frames=72`.
2. `balanced` (рекомендуемый дефолт):
   1. `first_chunk_emit_every=5`.
   2. `first_chunk_decode_window=48`.
   3. `first_chunk_frames=48`.
   4. `emit_every_frames=12`.
   5. `decode_window_frames=80`.
3. `quality`:
   1. `first_chunk_*` отключить.
   2. `emit_every_frames=14`.
   3. `decode_window_frames=96`.

## 7. Детальный план внедрения

### Шаг 0. Baseline и контрольная точка

1. Зафиксировать текущие метрики без TTS: `ttft_ms`, `e2e_ttft_ms`, стабильность barge-in.
2. Сохранить `configs/config.yaml` в rollback-копию.
3. Проверить, что `include_audio=true` сейчас не ломает response.

Критерий: есть измеримый baseline и быстрый откат.

### Шаг 1. Добавить TTS registry и активный выбор из конфига

Файлы:

1. `src/utils/operations/tts/registry.py` (новый).
2. `src/utils/operations/manager.py`.
3. `src/utils/config.py` (если нужны новые поля).

Сделать:

1. Реализовать `load_tts_operation()` (built-in + `entrypoint`).
2. Включить в `load_op()` ветку `OpTypes.TTS` через registry.
3. Добавить `tts_active_id` и `tts_strict_active_id` по аналогии со STT.

Критерий: backend стартует с выбранным `role: tts` и переключается через reload.

### Шаг 2. Добавить sidecar process type для Qwen3 TTS

Файлы:

1. `src/utils/processes/manager.py`.
2. `src/utils/processes/processes/qwen3_tts_server.py` (новый).
3. `apps/tts-qwen3-server/*` (новый app).

Сделать:

1. Новый `ProcessType.QWEN3_TTS`.
2. Запуск сервера с `host/port/device/model/profile`.
3. Health endpoint и restart semantics.

Критерий: процесс поднимается, переживает reload и корректно закрывается.

### Шаг 3. Реализовать `Qwen3TTSOperation`

Файлы:

1. `src/utils/operations/tts/qwen3.py` (новый).
2. `src/utils/operations/tts/base.py` (минимальные расширения контракта при необходимости).

Сделать:

1. `configure()` с полной валидацией параметров.
2. `start()`:
   1. `mode=sidecar`: link к процессу + readiness check.
   2. `mode=local`: ленивый локальный model load.
3. `_generate(content, emotion, ...)`:
   1. подготовка `instruct` из emotion mapping.
   2. streaming чтение чанков.
   3. нормализация формата в `pcm_s16le`.
4. `close()` с освобождением ресурсов.

Критерий: `use_operation(OpRoles.TTS, ...)` отдаёт валидные чанки без блокировки event loop.

### Шаг 4. Сервисный API sidecar (референс из groxaxo)

Файлы:

1. `apps/tts-qwen3-server/main.py` (новый).
2. `apps/tts-qwen3-server/backend.py` (новый).

Сделать:

1. Endpoint для stream synth (HTTP chunked или WS).
2. Ограничение конкурентности (`max_concurrent=1` по умолчанию для одного GPU контекста).
3. Логи:
   1. TTFB.
   2. total generation time.
   3. RTF.
4. Voice prompt cache (если включён clone режим).

Критерий: при параллельных запросах нет deadlock/краша и предсказуемый tail latency.

### Шаг 5. Конфиг Nira для модульного TTS

Файл: `configs/config.yaml`

Сделать:

1. Добавить:
   1. `tts_active_id`.
   2. `tts_strict_active_id`.
2. Добавить блок `role: tts` с `entrypoint`, `mode`, `process`, `streaming profile`.
3. Подготовить второй пример `role: tts` (например `melo`) для проверки переключаемости.

Критерий: смена `tts_active_id` + reload переключает провайдера без правки кода.

### Шаг 6. Эмоции и текстовые инструкции

Файлы:

1. `src/utils/operations/tts/qwen3.py`.
2. `configs/config.yaml` (таблица маппинга).

Сделать:

1. Нормализатор эмоций из поля и скобок.
2. Шаблон инструкций на русском (без агрессивной стилизации, чтобы не ломать дикцию).
3. Защита от prompt injection в TTS instruct (санитизация).

Критерий: на тест-наборе эмоций слышима смена просодии без роста пустых/битых чанков.

### Шаг 7. UI-гейт (без playback)

Файлы:

1. `apps/nira-web/src/App.tsx`.

Сделать:

1. Зафиксировать контракт: UI не воспроизводит `audio_chunk`.
2. Оставить UI только для текста/метрик TTS.
3. Проверить, что audio-события не вызывают побочных эффектов в интерфейсе.

Критерий: UI стабилен и не содержит встроенного аудио-плеера.

### Шаг 8. Discord/OBS output adapter

Файлы:

1. `src/utils/audio/output_router.py` (новый).
2. Конфиг маршрутизации в `configs/config.yaml`.

Сделать:

1. Вывод PCM в выбранный output device.
2. Поддержка VB-Cable/Voicemeeter.
3. Защита от блокировок (output worker queue + backpressure policy).

Критерий: звук стабильно попадает в Discord/OBS при длительном стриме.

### Шаг 9. Тестовый контур и регрессия TTS

Файлы:

1. `src/tools/tts_smoke.py` (новый).
2. `src/tools/tts_regression.py` (новый).
3. `src/tools/fixtures/tts_regression_manifest.example.json` (новый).

Сделать:

1. Smoke: короткая фраза, длинная фраза, пустой контент, эмоции.
2. Regression:
   1. измерение `tts_start_ms`, `e2e_tts_start_ms`.
   2. количество пустых/битых чанков.
   3. стабильность при interrupt.
3. Автоотчёт в `logs/tts_regression_*.json`.

Критерий: формальная проверка качества перед включением в прод-режим.

### Шаг 10. Нагрузочный прогон

Сделать:

1. 50-100 запросов с `include_audio=true`.
2. Сценарий с частыми барж-инами.
3. Наблюдение VRAM и перезапусков процесса.

Критерий: нет утечек, зависаний, роста p95 сверх лимитов.

### Шаг 11. Документация и эксплуатация

Сделать:

1. Обновить `АРХИТЕКТУРА проекта Нира.md` (блок TTS стандартов и runbook).
2. Обновить `README.md` раздел запуска TTS sidecar.
3. Добавить "быстрый rollback" и "known issues".

Критерий: любой запуск/переключение воспроизводим по документации.

## 8. Целевые метрики качества

Для профиля `balanced` на RTX 5060 (после warmup):

1. `tts_start_ms` p50: до 700 мс.
2. `tts_start_ms` p95: до 1200 мс.
3. VRAM для 0.6B: до 6 ГБ в steady-state.
4. Доля пустых audio chunk sequence: < 1%.
5. Успешность interrupt без подвисания job: > 99% на regression наборе.

## 9. Риски и меры

1. Риск: flash-attn недоступен на конкретной связке Win/Python/CUDA.  
Мера: штатный fallback на `sdpa` + sidecar isolate.

2. Риск: долгий warmup из-за `torch.compile`.  
Мера: предварительный warmup на старте и профиль `reduce-overhead`.

3. Риск: конфликт GPU ресурсов с llama.cpp.  
Мера: закрепить TTS на `cuda:1`, лимит concurrency = 1.

4. Риск: артефакты на стыках chunk.  
Мера: профили `emit_every_frames/decode_window_frames` + optional overlap/crossfade в output adapter.

5. Риск: рост latency при включении `pitch/rvc`.  
Мера: включать по одному фильтру, мерить `tts_start_ms` и RTF после каждого шага.

## 10. Быстрый rollback

1. Удалить/закомментировать активный `role: tts` в `configs/config.yaml`.
2. Очистить `tts_active_id`.
3. Выполнить `POST /api/operations/reload` или перезапуск backend.
4. Остановить `qwen3_tts_server` процесс.

Результат: система возвращается к текстовому режиму без изменения core-кода.

## 11. Статус выполнения (на момент 2026-03-04)

1. Исследование внешних TTS-репозиториев: выполнено.
2. Архитектурное решение и dependency-профили: зафиксированы в этом документе.
3. Реализация модульного Qwen3-TTS в коде Nira: следующий этап (не начат в рамках этого документа).
