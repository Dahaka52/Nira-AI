import asyncio
import logging
from config import Config
from jaison import JAIson, JobType

logging.basicConfig(level=logging.DEBUG)

async def main():
    Config().load_from_name('config')
    j = JAIson()
    await j.start()
    
    print("Sending context...")
    await j.append_conversation_context_text("test_job_1", JobType.CONTEXT_CONVERSATION_ADD_TEXT, user="Creator", timestamp=1700000000, content="Привет!")
    
    print("Requesting response...")
    await j.response_pipeline("test_job_2", JobType.RESPONSE, include_audio=False)
    
    print("Done")
    await j.stop()

if __name__ == "__main__":
    asyncio.run(main())
