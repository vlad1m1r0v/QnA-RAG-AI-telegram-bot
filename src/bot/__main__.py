import asyncio

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from src.llm.chains import ask_bot_async
from src.secrets import secrets
from src.utils.logger import get_logger
from src.utils.transcription import transcribe_url

logger = get_logger(__name__)

bot = Bot(
    token=secrets.telegram_bot_key,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2),
)
dp = Dispatcher()


async def _reply_llm(message: Message, question: str) -> None:
    status_msg = await message.answer("Шукаю відповідь\\.\\.\\. ⏳")
    try:
        answer = await ask_bot_async(question)
        await status_msg.edit_text(answer)
    except Exception:
        try:
            await status_msg.edit_text(answer, parse_mode=None)
        except Exception as e:
            logger.error(f"Failed to deliver LLM answer: {e}")
            await status_msg.edit_text("Виникла помилка при отриманні відповіді ❌", parse_mode=None)


@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привіт\\! Я твій AI\\-асистент\\. Надсилай текст або голосове повідомлення 🎙")


@dp.message(F.text)
async def handle_text(message: Message):
    await _reply_llm(message, message.text)


@dp.message(F.voice | F.audio)
async def handle_audio(message: Message):
    status_msg = await message.answer("Обробляю аудіо\\.\\.\\. ⏳")

    try:
        file_id = message.voice.file_id if message.voice else message.audio.file_id
        file = await bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{secrets.telegram_bot_key}/{file.file_path}"

        transcript = await transcribe_url(file_url)

        if transcript.status == "error":
            await status_msg.edit_text(f"Помилка транскрипції: {transcript.error}", parse_mode=None)
            return

        await status_msg.delete()
    except Exception as e:
        logger.error(f"Error in audio transcription: {e}")
        await status_msg.edit_text("Виникла помилка при обробці аудіо ❌", parse_mode=None)
        return

    await _reply_llm(message, transcript.text)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())