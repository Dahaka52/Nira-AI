import spacy

from utils.config import Config

from .base import FilterTextOperation

class SentenceChunkerFilter(FilterTextOperation):
    def __init__(self):
        super().__init__("chunker_sentence")
        self.nlp = None
        self.mode = "sentence"  # sentence | full
        
    async def start(self):
        await super().start()
        self.nlp = spacy.load(Config().spacy_model)
        
    async def close(self):
        await super().close()
        self.nlp = None
    
    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        if "mode" in config_d:
            mode = str(config_d["mode"]).strip().lower()
            if mode in ("sentence", "full"):
                self.mode = mode
        return
        
    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {"mode": self.mode}

    async def _generate(self, content: str = None, **kwargs):
        '''Generate a output stream'''
        text = str(content or "").strip()
        if not text:
            return

        # Qwen3 quality is noticeably better when TTS receives full utterance once.
        if self.mode == "full":
            yield {"content": text}
            return

        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
    
        for s in sentences:
            yield {
                "content": s
            }
