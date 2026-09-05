import asyncio
import json
import websockets

async def test():
    key = ''
    ws_url = f'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}'
    try:
        async with websockets.connect(ws_url) as ws:
            setup_msg = {
                'setup': {
                    'model': 'models/gemini-2.0-flash-exp',
                    'generationConfig': {'responseModalities': ['TEXT']},
                    'inputAudioTranscription': {'languageCodes': []}
                }
            }
            # Wait, model 'gemini-3.5-transcribe-live' might not exist. 
            # The documentation from 2026 says `models/gemini-3.5-transcribe-live`. Let's test that!
            setup_msg['setup']['model'] = 'models/gemini-3.5-transcribe-live'
            print('Sending setup...')
            await ws.send(json.dumps(setup_msg))
            print('Setup sent')
            res = await ws.recv()
            print('Response 1:', res)
            
            await ws.send(json.dumps({'realtimeInput': {'audioStreamEnd': True}}))
            print('End sent')
            res = await ws.recv()
            print('Response 2:', res)
    except Exception as e:
        print('Error:', e)

asyncio.run(test())
