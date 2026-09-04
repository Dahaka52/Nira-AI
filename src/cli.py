import asyncio
import logging
import sys
import time
from utils.config import Config
from utils.jaison import JAIson, JobType
from utils.helpers.observer import BaseObserverClient
from utils.console import (
    print_welcome, print_divider, print_stage,
    print_user_text, print_nira_start, print_nira_chunk, print_nira_end,
    C_YELLOW, C_RED, RESET, BOLD,
)

# Отключаем лишние логи сервера для чистоты чата
logging.getLogger().setLevel(logging.WARNING)


class ConsoleChatObserver(BaseObserverClient):
    """Слушатель событий JAIson для цветного вывода ответов Нира."""

    def __init__(self):
        super().__init__(server=JAIson().event_server)
        self.done = asyncio.Event()
        self._nira_started = False

    async def handle_event(self, event_id: str, payload) -> None:
        if event_id == JobType.RESPONSE.value:
            if payload.get("finished"):
                if self._nira_started:
                    print_nira_end()
                    self._nira_started = False
                self.done.set()
            elif "result" in payload and "content" in payload["result"]:
                chunk = payload["result"]["content"]
                if chunk:
                    if not self._nira_started:
                        # Первый чанк — выводим метку «Нира»
                        print_nira_start("Нира")
                        self._nira_started = True
                    print_nira_chunk(chunk)

    def reset(self):
        self.done.clear()
        self._nira_started = False


async def main():
    # ── Инициализация ─────────────────────────────────────────────────────────
    print_stage("INIT", "Загрузка конфига и инициализация Nira…", "boot")
    Config().load_from_name('config')
    j = JAIson()
    await j.start()

    observer = ConsoleChatObserver()

    # ── Приветственный экран ───────────────────────────────────────────────────
    print_welcome()

    try:
        while True:
            # Неблокирующий ввод
            try:
                raw = await asyncio.to_thread(input, "")
            except EOFError:
                break

            text = raw.strip()
            if not text:
                continue
            if text.lower() in ["выход", "exit", "quit"]:
                print_stage("BYE", "Завершение работы Нира…", "warn")
                break

            # Красивый вывод реплики пользователя
            print_user_text(text, name="Вы")

            # 1. Добавляем сообщение пользователя в контекст
            await j.append_conversation_context_text(
                "chat_ctx",
                JobType.CONTEXT_CONVERSATION_ADD_TEXT,
                user="Creator",
                timestamp=int(time.time()),
                content=text,
            )

            observer.reset()

            # 2. Запускаем генерацию ответа
            await j.create_job(JobType.RESPONSE, include_audio=False)

            # Ждём окончания генерации
            await observer.done.wait()
            print_divider()

    except KeyboardInterrupt:
        print(f"\n  {BOLD}{C_YELLOW}[Ctrl+C]{RESET} Завершение…")
    finally:
        await j.stop()


if __name__ == "__main__":
    asyncio.run(main())
