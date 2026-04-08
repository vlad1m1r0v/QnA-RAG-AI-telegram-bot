import asyncio
import assemblyai as aai

from src.secrets import secrets

aai.settings.api_key = secrets.assembly_ai_api_key

_transcriber_config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True
)
_transcriber = aai.Transcriber(config=_transcriber_config)


async def transcribe_url(url: str) -> aai.Transcript:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _transcriber.transcribe(url))