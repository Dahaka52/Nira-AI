import asyncio
import base64
import json
import logging
import os
import wave
import sys
import websockets

async def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        import yaml
        with open(r"c:\Nirmita\configs\config.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            for op in data.get("operations", []):
                if op.get("id") == "google_stt":
                    api_key = op.get("api_key")
                    break
    
    if not api_key:
        print("API key not found")
        return
        
    model = "gemini-1.5-flash"
    ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
    
    import math
    import struct
    
    # Generate 1 second of 440Hz sine wave
    sine_wave = bytearray()
    for i in range(16000):
        val = int(32767.0 * math.sin(2.0 * math.pi * 440.0 * i / 16000.0))
        sine_wave.extend(struct.pack('<h', val))
        
    with wave.open("test_audio.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(sine_wave)
        
    with open("test_audio.wav", "rb") as f:
        audio_bytes = f.read()
        
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    try:
        async with websockets.connect(ws_url) as ws:
            setup_msg = {
                "setup": {
                    "model": f"models/{model}",
                    "generationConfig": {
                        "responseModalities": ["TEXT"],
                    },
                    "inputAudioTranscription": {
                        "languageCodes": ["ru-RU"]
                    }
                }
            }
            print(f"Sending setup: {setup_msg}")
            await ws.send(json.dumps(setup_msg))
            
            # Read setup complete
            resp = await ws.recv()
            print(f"Received setup response: {resp}")
            
            audio_msg = {
                "clientContent": {
                    "turns": [{
                        "role": "user",
                        "parts": [{
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=16000",
                                "data": audio_base64
                            }
                        }]
                    }],
                    "turnComplete": True
                }
            }
            print("Sending audio...")
            await ws.send(json.dumps(audio_msg))
            
            while True:
                resp = await ws.recv()
                print(f"Received: {resp}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
