import asyncio
import logging
from utils.operations.t2t.llamacpp import LlamaCppT2T
from utils.config import Config
from utils.prompter.message import ChatMessage
import datetime

logging.basicConfig(level=logging.DEBUG)

async def main():
    Config().load_from_name('config')
    op = LlamaCppT2T()
    await op.start()
    
    # Имитируем запрос
    messages = [ChatMessage("Creator", "Привет!", datetime.datetime.now())]
    instruction_prompt = "You are a helpful assistant."
    
    print("\n--- Отправка запроса ---")
    try:
        async for chunk in op._generate(instruction_prompt=instruction_prompt, messages=messages):
            print(f"CHUNK: {chunk}")
    except Exception as e:
        print(f"ERROR: {e}")
        
    print("--- Завершено ---")
    await op.close()

if __name__ == "__main__":
    asyncio.run(main())
