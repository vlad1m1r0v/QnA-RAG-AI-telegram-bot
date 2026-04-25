import asyncio
import assemblyai as aai

from src.secrets import secrets

aai.settings.api_key = secrets.assembly_ai_api_key

_WORD_BOOST = [
    "CRM", "CMS", "API", "Backend", "Frontend", "Crypto",
    "Blockchain", "DevOps", "UI", "UX", "SaaS", "MVP",
    "Telegram", "Instagram", "Stripe", "PayPal", "LiqPay",
    "React", "Vue", "Node", "Python", "Docker", "MongoDB",
]

_transcriber_config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    word_boost=_WORD_BOOST,
)
_transcriber = aai.Transcriber(config=_transcriber_config)


async def transcribe_url(url: str) -> aai.Transcript:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _transcriber.transcribe(url))