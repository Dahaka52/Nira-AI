# План изменений Nira (актуальный)

Дата обновления: 2026-03-03

## 1) Активные задачи (в работе, перед TTS)

### A. Подготовка к интеграции TTS
- [ ] Подключить первый рабочий TTS-модуль в `operations: role=tts` и проверить `response_pipeline(include_audio=true)`.
- [ ] Прогнать smoke-тесты TTS-потока: `text_chunk -> tts -> filter_audio -> ws audio_chunk` без разрывов и зависаний.
- [ ] Добавить базовые параметры TTS в `config.yaml` (модель, устройство, голос, темп) и проверить reload через `/api/operations/reload`.
- [ ] Зафиксировать метрики для TTS-старта в UI (`tts_start_ms`, `e2e_tts_start_ms`) на реальном прогоне.

### B. STT стабильность до/после TTS
- [ ] Проверить расширенный `hotwords.txt` (имена/термины проекта) и донастроить score по живым логам.
- [ ] Добить кейсы "пустой STT на коротких репликах" (смех/междометия) настройками VAD + словарями в конфиге.

### C. Фаза 2 (после TTS)
- [ ] Ввести `turn_id/utterance_id` сквозь mic -> STT -> backend -> UI.
- [ ] Ввести lightweight intent/postprocess (`filler | backchannel | stop | continue | new_topic`) для выбора политики interrupt/queue/respond.
- [ ] Добавить adaptive interrupt policy для мульти-спикер/стрим-сценариев.

---

## 2) Выполнено (сгруппировано по подсистемам)

### A. Очередь задач и отмена
- [✅] Исправлен `cancel_job()` и skip-логика для queued jobs.
- [✅] `_interrupt_jobs()` очищает очередь/карты задач и корректно закрывает queued coroutine.
- [✅] Исправлена обработка `CancelledError` в job loop + корректный `cancelled` event в WS.
- [✅] Исправлен кейс `Job None ... was cancelled`.
- [✅] Добавлен guard для пустого `t2t_result` (нет падения FILTER_TEXT assert).

### B. Прерывания, буферизация и turn-taking (phase 1)
- [✅] Добавлен `/api/context/conversation/speech_start` и вызов из mic-клиента.
- [✅] Реализован режим `speech_start_interrupt_mode: soft|hard`.
- [✅] Реализованы `voice_merge_window_ms`, `response_debounce_ms`, `response_min_quiet_ms_after_speech_start`.
- [✅] Реализован базовый `continue-after-interrupt`.
- [✅] Backchannel не запускает лишний response.
- [✅] Stop-команды (`стоп/стой/подожди/...`) корректно прерывают активный response.
- [✅] Устойчивость stop-команд улучшена для шумного/обрезанного STT.
- [✅] Добавлены отдельные пороги для коротких interrupt-сегментов.

### C. Wake-word и короткие реплики
- [✅] Добавлена нормализация алиасов имени (`мира/миром/... -> нира`) для одиночных обращений.
- [✅] Wake-словарь вынесен в конфиг (`microphone.wake_words`, `wake_word_aliases`).
- [✅] Добавлена мягкая реакция на короткие эмоциональные реплики (смех/хмык) через конфиг:
  `respond_to_short_emotes`, `short_emote_words`.

### D. Sherpa / hotwords
- [✅] Добавлена валидация hotwords по `tokens.txt` перед запуском Sherpa.
- [✅] Добавлена автосанитизация `hotwords` в `logs/sherpa_hotwords.resolved.txt` (невалидные строки отбрасываются).
- [✅] Добавлен автоподхват `bpe_vocab` для `modeling-unit=bpe`, устранён крэш старта с hotwords.
- [✅] Добавлены конфиг-параметры Sherpa: `model_variant`, `decoding_method`, `num_active_paths`, `hotwords_file`, `hotwords_score`, `gpu_id`.

### E. Производительность и устойчивость backend/UI
- [✅] UI отвязан от критического пути response.
- [✅] UI reconnect больше не сбрасывает чат (локальный кэш + merge history).
- [✅] WS broadcast защищён от мёртвых клиентов.
- [✅] Сокращён лишний WS-трафик: debug prompt/history/raw события теперь отключены по умолчанию (`broadcast_debug_prompt_events=false`).
- [✅] Legacy job path `append_conversation_context_audio` приведён к единому поведению через `process_audio_immediate`.

### F. Конфиг и операции
- [✅] Исправлены `Config.save`, `update_config`, `configure_context`.
- [✅] Добавлен `stt_active_id` и выбор активного STT из конфига.
- [✅] Hot-reload промптов усилен (`mtime_ns + size`, fallback, логирование ошибок).

---

## 3) Текущее состояние

- Phase 1 по прерываниям закрыт и стабилен для текущего режима работы.
- Основной следующий этап: интеграция TTS-модуля без деградации latency и без регрессий в прерываниях/очереди.
