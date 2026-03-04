import spacy
import re

from utils.config import Config

from .base import FilterTextOperation

class SentenceChunkerFilter(FilterTextOperation):
    def __init__(self):
        super().__init__("chunker_sentence")
        self.nlp = None
        self.mode = "sentence"  # sentence | full
        self.max_chars = 90
        
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
        if "max_chars" in config_d:
            try:
                self.max_chars = max(20, int(config_d["max_chars"]))
            except Exception:
                pass
        return
        
    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {
            "mode": self.mode,
            "max_chars": int(self.max_chars),
        }

    def _split_long_sentence(self, sentence: str):
        text = str(sentence or "").strip()
        if not text:
            return
        if len(text) <= int(self.max_chars):
            yield text
            return

        # Prefer linguistic pause markers first to keep prosody natural.
        parts = [p.strip() for p in re.split(r"([,;:])", text)]
        merged = []
        buf = ""
        i = 0
        while i < len(parts):
            cur = parts[i]
            if not cur:
                i += 1
                continue
            if cur in (",", ";", ":") and buf:
                candidate = (buf + cur).strip()
                if len(candidate) <= int(self.max_chars):
                    buf = candidate
                    i += 1
                    continue
            candidate = (buf + " " + cur).strip() if buf else cur
            if len(candidate) <= int(self.max_chars):
                buf = candidate
            else:
                if buf:
                    merged.append(buf.strip())
                    buf = cur.strip()
                else:
                    # Hard split for pathological long tokens/no punctuation.
                    chunk = cur
                    while len(chunk) > int(self.max_chars):
                        merged.append(chunk[: int(self.max_chars)].strip())
                        chunk = chunk[int(self.max_chars):].strip()
                    buf = chunk
            i += 1
        if buf:
            merged.append(buf.strip())

        for m in merged:
            if m:
                yield m

    def _pack_pieces(self, pieces):
        """Pack short pieces into chunks up to max_chars to reduce TTS startup overhead."""
        limit = int(self.max_chars)
        buf = ""
        for piece in pieces:
            p = str(piece or "").strip()
            if not p:
                continue
            if not buf:
                buf = p
                continue

            candidate = f"{buf} {p}".strip()
            if len(candidate) <= limit:
                buf = candidate
                continue

            yield buf
            buf = p

        if buf:
            yield buf

    async def _generate(self, content: str = None, **kwargs):
        '''Generate a output stream'''
        text = str(content or "").strip()
        if not text:
            return

        # Qwen3 quality is noticeably better when TTS receives full utterance once.
        if self.mode == "full":
            yield {"content": text}
            return

        # Fast path: short replies should go as a single TTS request.
        if len(text) <= int(self.max_chars):
            yield {"content": text}
            return

        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        pieces = []
        for s in sentences:
            pieces.extend(list(self._split_long_sentence(s)))

        for packed in self._pack_pieces(pieces):
            yield {"content": packed}
