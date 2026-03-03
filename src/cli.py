import asyncio
import logging
import sys
import time
from utils.config import Config
from utils.jaison import JAIson, JobType
from utils.helpers.observer import BaseObserverClient

# Отключаем лишние логи сервера для чистоты чата
logging.getLogger().setLevel(logging.WARNING)

class ConsoleChatObserver(BaseObserverClient):
    def __init__(self):
        super().__init__(server=JAIson().event_server)
        self.done = asyncio.Event()

    async def handle_event(self, event_id: str, payload) -> None:
        if event_id == JobType.RESPONSE.value:
            if payload.get("finished"):
                self.done.set()
            elif "result" in payload and "content" in payload["result"]:
                # Выводим чанки текста (или весь текст целиком, если streaming отключен)
                chunk = payload["result"]["content"]
                print(chunk, end="", flush=True)

async def main():
    print("Инициализация Nira (JAIson)... Загрузка модели может занять 15-20 секунд.")
    Config().load_from_name('config')
    j = JAIson()
    await j.start()
    
    # Подключаем наш консольный слушатель к серверу событий JAIson
    observer = ConsoleChatObserver()
    
    print("\n" + "="*50)
    print(" Nira Console Chat (V1) ")
    print(" Напиши 'выход', 'exit' или нажми Ctrl+C для выхода.")
    print("="*50)

    try:
        while True:
            # Используем asyncio.to_thread для неблокирующего input()
            text = await asyncio.to_thread(input, "\nВы: ")
            text = text.strip()
            
            if not text:
                continue
            if text.lower() in ["выход", "exit", "quit"]:
                break
                
            # 1. Добавляем сообщение пользователя в контекст
            await j.append_conversation_context_text("chat_ctx", JobType.CONTEXT_CONVERSATION_ADD_TEXT, user="Creator", timestamp=int(time.time()), content=text)
            
            print(f"Нира: ", end="", flush=True)
            observer.done.clear()
            
            # 2. Запускаем генерацию ответа
            await j.create_job(JobType.RESPONSE, include_audio=False)
            
            # Ждем окончания генерации ответа
            await observer.done.wait()
            print() # Перенос строки после завершения ответа
            
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    finally:
        await j.stop()

if __name__ == "__main__":
    asyncio.run(main())
