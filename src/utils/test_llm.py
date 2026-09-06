import asyncio
import logging
from utils.config import Config
from utils.nira import Nira, JobType

logging.basicConfig(level=logging.DEBUG)

async def main():
    Config().load_from_name('config')
    nira = Nira()
    await nira.start()
    
    print("Sending context...")
    await nira.append_conversation_context_text("test_job_1", JobType.CONTEXT_CONVERSATION_ADD_TEXT, user="Creator", timestamp=1700000000, content="Привет!")
    
    print("Requesting response...")
    await nira.response_pipeline("test_job_2", JobType.RESPONSE, include_audio=False)
    
    print("Done")
    await nira.stop()

if __name__ == "__main__":
    asyncio.run(main())
