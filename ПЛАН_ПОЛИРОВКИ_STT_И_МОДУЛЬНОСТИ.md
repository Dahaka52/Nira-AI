# План полировки STT-тракта и модульной реализации Sherpa ONNX

Дата: 2026-03-04  
Проект: `C:\Nirmita`

## 1. Цель

Довести текущий STT-тракт до production-уровня по стабильности/наблюдаемости и зафиксировать архитектуру, где:

1. Sherpa ONNX работает как полноценный модульный STT-адаптер.
2. Подключение/переключение STT выполняется через стартовый конфиг (`stt_active_id`) без правок пайплайна.
3. Добавление нового STT не требует изменений в `jaison.py` и другом core-коде оркестрации.

## 2. Привязка к целям из шапки АРХИТЕКТУРА

1. Устойчивый характер: STT должен давать стабильный и предсказуемый текст без «шумовых» фраз, иначе ломается persona и continuity.
2. Осознание себя в моменте: нужны `turn_id/utterance_id` и события partial/final, чтобы агент понимал фазу диалога.
3. Развитие личности: качественный лог распознанных реплик и метаданных для последующего самоанализа.
4. Продвинутая эмоциональность: STT должен не терять междометия/эмоциональные маркеры (смех, паузы, короткие реакции).
5. Долгосрочная память: стабильная структура STT-выхода для сохранения в память без мусора.
6. Смекалка/логика: меньше ошибок транскрипции => выше качество reasoning.
7. Социальные навыки: устойчивое распознавание коротких реплик, backchannel, stop/continue.
8. Мульти-люди/стрим: нужна маршрутизация по источнику (`source_id`) и политика прерываний.
9. Дообучение на диалогах: чистый корпус STT + confidence/metadata.
10. Идентификация по голосу: STT-тракт должен иметь расширяемую точку для voice-id/diarization.
11. Discord/игры: единый контракт STT для разных транспортов аудио (mic, discord, stream, game bus).

## 3. Текущий STT-тракт (факты по коду)

1. `hw-mic-client` делает локальный VAD и отправляет:
`/api/context/conversation/speech_start` + `/api/context/conversation/audio`.
2. Backend обрабатывает аудио сразу в `process_audio_immediate()` (вне job queue), вызывает `OpRoles.STT`.
3. Активный STT выбирается через `stt_active_id` в `OperationManager.load_operations_from_config()`.
4. Сейчас для роли STT реализован только `id=sherpa`.
5. Sherpa поднимается как sidecar-процесс (`apps/stt-sherpa-server/start_server.py`) через `ProcessManager`.

## 4. Узкие места текущей реализации

1. Жесткая связка STT-реестра: `load_op()` знает только `sherpa`.
2. Sherpa process config извлекается по условию `id == "sherpa"` внутри process-класса, что мешает унификации.
3. Порт Sherpa захардкожен (`6006`) в process и в дефолтах.
4. Контракт STT пока только final (`transcription`), без формализованных partial/final метаданных.
5. Нет стандартизированного `confidence`, `stt_latency_ms`, `source_id`, `turn_id`, `utterance_id` на выходе STT.
6. `/api/context/conversation/audio` создает background-task без backpressure/лимитов.
7. Ошибки STT в immediate-path логируются, но не имеют системного события деградации для UI/мониторинга.
8. Sherpa и operation частично дублируют ответственность за конфигурацию.

## 5. Целевая архитектура STT V2

## 5.1 Единый контракт STT-адаптера

Вход STT:

```json
{
  "prompt": "optional",
  "audio_bytes": "bytes",
  "sr": 16000,
  "sw": 2,
  "ch": 1,
  "source_id": "mic",
  "turn_id": "optional",
  "utterance_id": "optional"
}
```

Выход STT (event stream):

```json
{
  "text": "...",
  "is_final": true,
  "confidence": 0.0,
  "turn_id": "...",
  "utterance_id": "...",
  "source_id": "mic",
  "provider": "sherpa",
  "stt_latency_ms": 123
}
```

Совместимость с текущим core:

1. Для backward compatibility каждый final event содержит `transcription = text`.
2. `process_audio_immediate()` работает с final событием как раньше.

## 5.2 Декуплинг провайдера и процесса

1. STT-операция отвечает за протокол распознавания и контракт chunk output.
2. ProcessRunner отвечает за запуск/health/restart sidecar-процесса (если провайдер требует внешний процесс).
3. Конфиг process-параметров хранится рядом с STT-операцией, а не ищется по `id == "sherpa"` в глубине process-класса.

## 5.3 Реестр STT без правки пайплайна

Целевой принцип:

1. `jaison.py` не знает конкретные STT.
2. `OperationManager` выбирает STT только по конфигу.
3. Добавление STT-провайдера: новый модуль + запись в конфиг (и, при необходимости, в registry), без изменений в пайплайне ответа/контекста.

## 6. Рекомендуемая схема конфига (целевая)

```yaml
stt_active_id: "sherpa_ru"

operations:
  - role: stt
    id: sherpa_ru
    entrypoint: "utils.operations.stt.sherpa:SherpaSTT"
    ws_url: "ws://127.0.0.1:6106"
    language: "ru"
    process:
      type: "sherpa_onnx_server"
      autostart: true
      port: 6106
      provider: "cpu"
      gpu_id: 1
      model_dir: "C:\\Nirmita\\models\\vosk-model-small-streaming-ru"
      model_variant: "fp32"
      decoding_method: "modified_beam_search"
      num_active_paths: 6
      use_endpoint: 0
      hotwords_file: "C:\\Nirmita\\models\\hotwords.txt"
      hotwords_score: 3.5

  - role: stt
    id: whisper_local
    entrypoint: "utils.operations.stt.whisper_local:WhisperLocalSTT"
    model: "small"
    device: "cuda"
```

## 7. Детальный пошаговый план

### Шаг 0. Baseline и метрики до правок

Сделать:

1. Зафиксировать baseline STT quality и latency на 3 наборах фраз: короткие команды, обычные реплики, шумные условия.
2. Снять baseline метрики: пустые транскрипции, false interrupt, average STT latency.
3. Сохранить рабочий конфиг как rollback.

Критерий:

1. Есть объективная точка сравнения до рефакторинга.

### Шаг 1. Формализовать STT контракт V2

Файлы:

1. `src/utils/operations/stt/base.py`
2. `src/utils/jaison.py`
3. `api.yaml` (описание контрактов)

Сделать:

1. Добавить в базовый STT формат поля `text/is_final/confidence/provider/source_id/turn_id/utterance_id`.
2. Оставить `transcription` как backward-compatible alias для final.
3. В `process_audio_immediate()` обрабатывать только `is_final=true` (если partial появятся).

Критерий:

1. Старый Sherpa продолжает работать, а контракт уже расширен.

### Шаг 2. Разделить operation-config и process-config

Файлы:

1. `src/utils/operations/stt/sherpa.py`
2. `src/utils/processes/processes/sherpa_server.py`

Сделать:

1. Убрать поиск по `id == "sherpa"` внутри process-класса.
2. Передавать нужные параметры process явно из конфигурации активной операции.
3. Убрать жесткую привязку к порту `6006`; порт задается конфигом.

Критерий:

1. Sherpa процесс полностью параметризован через активный STT config.

### Шаг 3. Ввести STT ProcessRunner слой

Файлы:

1. `src/utils/processes/manager.py`
2. `src/utils/processes/base.py`
3. новый модуль: `src/utils/processes/stt_runner.py` (или аналог)

Сделать:

1. Добавить унифицированный интерфейс `ensure_running/health/restart` для STT sidecar.
2. Привязать жизненный цикл runner к `start/close` операции.
3. Добавить retry/backoff при старте STT process.

Критерий:

1. Внешний STT процесс управляется одинаково для любых провайдеров-sidecar.

### Шаг 4. Реестр STT-провайдеров

Файлы:

1. `src/utils/operations/manager.py`
2. новый модуль: `src/utils/operations/stt/registry.py`

Сделать:

1. Вынести STT реестр из `load_op()` в отдельный registry.
2. Добавить поддержку `entrypoint` (динамический импорт) или явного реестра с регистрацией.
3. Обеспечить понятную ошибку, если провайдер не найден.

Критерий:

1. Добавление нового STT не требует правок `jaison.py` и business-pipeline.

### Шаг 5. Доработать stt_active_id и startup поведение

Файлы:

1. `src/utils/operations/manager.py`
2. `configs/config.yaml`

Сделать:

1. Оставить один активный STT по `stt_active_id` как сейчас.
2. Добавить строгую валидацию: если id не найден, явная ошибка старта (опционально через режим strict).
3. Поддержать быстрый runtime-switch через `/api/config/update` + `/api/operations/reload`.

Критерий:

1. Переключение STT воспроизводимо только конфигом.

### Шаг 6. Полировка качества Sherpa ONNX

Файлы:

1. `configs/config.yaml`
2. `src/utils/processes/processes/sherpa_server.py`
3. `apps/hw-mic-client/main.py`

Сделать:

1. Зафиксировать пресеты `low_latency`, `balanced`, `noisy_room`.
2. Доработать hotwords pipeline: контроль дельты качества по наборам фраз.
3. Добавить контролируемую нормализацию/постобработку (без агрессивного искажения текста).
4. Вывести отдельные STT метрики в логи: `stt_latency_ms`, `empty_rate`, `stop_cmd_recall`.

Критерий:

1. Снижение пустых/ошибочных транскрипций в ключевых сценариях.

### Шаг 7. Надежность и backpressure

Файлы:

1. `src/utils/server/app_server.py`
2. `src/utils/jaison.py`

Сделать:

1. Добавить ограничение на количество параллельных `process_audio_immediate` задач.
2. Добавить drop/merge политику для избыточных аудио-сегментов.
3. Добавить системные WS-события деградации STT (timeout/unavailable/restarting).

Критерий:

1. Backend не деградирует при burst-аудио и кратковременных сбоях STT.

### Шаг 8. Сквозные идентификаторы turn/utterance

Файлы:

1. `apps/hw-mic-client/main.py`
2. `src/utils/server/app_server.py`
3. `src/utils/jaison.py`
4. `apps/nira-web/src/App.tsx`

Сделать:

1. Генерировать `utterance_id` на клиенте микрофона.
2. Протаскивать `turn_id/utterance_id/source_id` через STT -> backend -> WS/UI.
3. Логировать и хранить эти идентификаторы в истории/телеметрии.

Критерий:

1. Есть трассируемость каждой реплики от микрофона до ответа.

### Шаг 9. Подготовка к voice-id / multi-speaker

Файлы:

1. новый pre-STT модуль (например `src/utils/operations/filter_audio/speaker_tag.py` или отдельный `role`)
2. `src/utils/jaison.py`

Сделать:

1. Добавить расширяемую точку перед STT для speaker embedding/voice-id.
2. Включить `speaker_id` в STT output contract.
3. Ввести policy для мульти-спикерных interrupt сценариев.

Критерий:

1. Архитектура готова для цели №10 и №8 без переработки core pipeline.

### Шаг 10. ПРОПУСТИТЬ, ПОЗЖЕ ДОБАВЛЮ РЕАЛЬНЫЙ ВТОРОЙ STT. 
Добавить второй STT-провайдер как proof-of-modularity

Сделать:

1. Реализовать второй адаптер (например `whisper_local` или `sensevoice`).
2. Добавить запись в `operations` и переключение `stt_active_id`.
3. Проверить, что старт/перезапуск и response-pipeline идентичны поведению с Sherpa.

Критерий:

1. Новый STT поднимается и работает без изменений `jaison.py`.

### Шаг 11. Тестовый контур и регрессия

Сделать:

1. Smoke тесты: `audio -> stt final`, `speech_start`, stop/continue, wake-word.
2. Regression наборы аудио с ожидаемыми транскрипциями.
3. Нагрузочный тест burst-сегментов и frequent interruptions.

Критерий:

1. Есть автоматизированный минимальный контроль качества STT при каждом изменении.

## 8. Definition of Done

1. Sherpa ONNX работает как модульный STT adapter с явным process lifecycle.
2. Переключение STT выполняется только сменой `stt_active_id` + reload/перезапуск.
3. Добавлен минимум один альтернативный STT-провайдер без правок пайплайна.
4. Есть сквозные `turn_id/utterance_id/source_id` в STT тракте.
5. Есть STT метрики и деградационные события для наблюдаемости.
6. Нет регрессий в interrupt/queue behavior.

## 9. Быстрый runbook переключения STT

1. Добавить новый блок `role: stt` в `operations`.
2. Поставить `stt_active_id` в нужный `id`.
3. Выполнить `POST /api/operations/reload` или перезапуск.
4. Проверить `GET /api/operations` -> поле `stt`.
5. Прогнать smoke: короткая stop-команда, обычная реплика, wake-word.

