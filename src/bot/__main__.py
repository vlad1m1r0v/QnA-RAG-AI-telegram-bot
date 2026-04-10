import asyncio
import re

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver

from src.llm.graph import build_graph, ask_bot_async, reset_state_async
from src.secrets import secrets
from src.utils.logger import get_logger
from src.utils.transcription import transcribe_url

logger = get_logger(__name__)

bot = Bot(
    token=secrets.telegram_bot_key,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

dp = Dispatcher()

_ALLOWED_TAGS = {'b', 'i', 'code', 'pre', 'a', 'blockquote'}


def sanitize_telegram_html(text: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(
        r'<(/?)(\w+)([^>]*)>',
        lambda m: m.group(0) if m.group(2).lower() in _ALLOWED_TAGS else '',
        text,
    )
    return text


async def _reply_llm(message: Message, question: str, graph) -> None:
    status_msg = await message.answer("Шукаю відповідь... ⏳")
    try:
        result = await ask_bot_async(graph, question, str(message.chat.id))
        reply_markup = None
        if result["response_type"] == "brief_ready":
            reply_markup = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="⏱ Естімейт проєкту", callback_data="gen_estimate")
            ]])
        await status_msg.edit_text(sanitize_telegram_html(result["response_text"]), reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"LLM reply failed: {e}")
        await status_msg.edit_text("Виникла помилка при отриманні відповіді ❌", parse_mode=None)


@dp.message(Command("start"))
async def cmd_start(message: Message, graph) -> None:
    await reset_state_async(graph, str(message.chat.id))
    await message.answer(
        "Привіт. Я AI-асистент компанії. Я можу:\n"
        "• Відповідати на питання про компанію\n"
        "• Зібрати бриф вашого проєкту\n\n"
        "Надсилайте текст або голосове повідомлення 🎙",
        parse_mode=None,
    )


@dp.message(Command("reset-memory"))
async def cmd_reset_memory(message: Message, graph) -> None:
    await reset_state_async(graph, str(message.chat.id))
    await message.answer("Чат та бриф очищено. Починаємо з початку 🔄", parse_mode=None)


@dp.callback_query(F.data == "gen_estimate")
async def handle_estimate(callback: CallbackQuery, graph) -> None:
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer()
    await bot.send_chat_action(callback.message.chat.id, "typing")

    thread_id = str(callback.from_user.id)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="__ESTIMATE__")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    await callback.message.answer(
        sanitize_telegram_html(result["messages"][-1].content),
        parse_mode="HTML",
    )


@dp.message(F.text)
async def handle_text(message: Message, graph) -> None:
    await _reply_llm(message, message.text, graph)


@dp.message(F.voice | F.audio)
async def handle_audio(message: Message, graph) -> None:
    status_msg = await message.answer("Обробляю аудіо... ⏳")
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

    await _reply_llm(message, transcript.text, graph)


async def main():
    with MongoDBSaver.from_conn_string(secrets.mongodb_url) as saver:
        graph = build_graph(saver)
        await dp.start_polling(bot, graph=graph)


if __name__ == "__main__":
    asyncio.run(main())