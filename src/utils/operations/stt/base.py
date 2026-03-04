'''
STT Operations (at minimum) require the following fields for input chunks:
- prompt: (str) initial words to help with transcription (Optional)
- audio_bytes: (bytes) pcm audio data
- sr: (int) sample rate
- sw: (int) sample width
- ch: (int) audio channels

Adds to chunk:
- V2 format:
  - text: (str) recognized text
  - is_final: (bool) whether chunk is final transcript
  - confidence: (float|None)
  - provider: (str) stt provider id
  - source_id: (str|None) input stream id (mic/discord/etc)
  - turn_id: (str|None)
  - utterance_id: (str|None)
  - stt_latency_ms: (int) processing latency
- Backward compatibility:
  - transcription: (str) alias of final text for existing pipeline code
'''

import time
from typing import Dict, Any, AsyncGenerator

from ..base import Operation
from ..base.error import UsedInactiveError

class STTOperation(Operation):
    def __init__(self, op_id: str):
        super().__init__("STT", op_id)
        
    ## TO BE OVERRIDEN ####
    async def start(self) -> None:
        '''General setup needed to start generated'''
        await super().start()
    
    async def close(self) -> None:
        '''Clean up resources before unloading'''
        await super().close()
    
    async def _parse_chunk(self, chunk_in: Dict[str, Any]) -> Dict[str, Any]:
        '''Extract information from input for use in _generate'''
        assert "audio_bytes" in chunk_in
        assert isinstance(chunk_in["audio_bytes"], bytes)
        assert len(chunk_in["audio_bytes"]) > 0
        assert "sr" in chunk_in
        assert isinstance(chunk_in["sr"], int)
        assert chunk_in["sr"] > 0
        assert "sw" in chunk_in
        assert isinstance(chunk_in["sw"], int)
        assert chunk_in["sw"] > 0
        assert "ch" in chunk_in
        assert isinstance(chunk_in["ch"], int)
        assert chunk_in["ch"] > 0
        
        return {
            "prompt": chunk_in.get("prompt", ""),
            "audio_bytes": chunk_in["audio_bytes"],
            "sr": chunk_in["sr"],
            "sw": chunk_in["sw"],
            "ch": chunk_in["ch"],
            "source_id": chunk_in.get("source_id"),
            "turn_id": chunk_in.get("turn_id"),
            "utterance_id": chunk_in.get("utterance_id"),
            "speaker_id": chunk_in.get("speaker_id"),
            "input_timestamp_ms": chunk_in.get("input_timestamp_ms"),
        }

    async def __call__(self, chunk_in: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Override base call to normalize STT output contract while staying compatible
        with legacy `transcription` readers.
        """
        if not self.active:
            raise UsedInactiveError(self.op_type, self.op_id)

        started = time.perf_counter()
        kwargs = await self._parse_chunk(chunk_in)

        source_id = kwargs.get("source_id")
        turn_id = kwargs.get("turn_id")
        utterance_id = kwargs.get("utterance_id")
        speaker_id = kwargs.get("speaker_id")

        async for chunk_out in self._generate(**kwargs):
            normalized = dict(chunk_out or {})
            is_final = bool(normalized.get("is_final", True))

            text = normalized.get("text")
            if text is None:
                text = normalized.get("transcription", "")
            text = str(text or "")
            normalized["text"] = text
            normalized["is_final"] = is_final

            if "confidence" not in normalized:
                normalized["confidence"] = None
            if "provider" not in normalized:
                normalized["provider"] = self.op_id
            if "source_id" not in normalized:
                normalized["source_id"] = source_id
            if "turn_id" not in normalized:
                normalized["turn_id"] = turn_id
            if "utterance_id" not in normalized:
                normalized["utterance_id"] = utterance_id
            if "speaker_id" not in normalized:
                normalized["speaker_id"] = speaker_id
            if "stt_latency_ms" not in normalized:
                normalized["stt_latency_ms"] = int((time.perf_counter() - started) * 1000)

            # Legacy alias expected by existing immediate pipeline code.
            if is_final:
                normalized["transcription"] = text

            yield normalized
    
    ## TO BE IMPLEMENTED ####
    async def configure(self, config_d: Dict[str, Any]):
        '''Configure and validate operation-specific configuration'''
        raise NotImplementedError
    
    async def get_configuration(self) -> Dict[str, Any]:
        '''Returns values of configurable fields'''
        raise NotImplementedError
    
    async def _generate(
        self,
        prompt: str = None,
        audio_bytes: bytes = None,
        sr: int = None,
        sw: int = None,
        ch: int = None,
        source_id: str = None,
        turn_id: str = None,
        utterance_id: str = None,
        speaker_id: str = None,
        input_timestamp_ms: int = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        '''Generate a output stream'''
        raise NotImplementedError
    
        yield {
            "transcription": "example transcribed text"
        }
