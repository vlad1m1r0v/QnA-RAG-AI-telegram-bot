import re

from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.bot.keyboards import brief_ready_keyboard
from src.bot.states import BriefFSM
from src.llm.graph import ask_bot_async
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_TAGS = {'b', 'i', 'a'}


def sanitize_telegram_html(text: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(
        r'<(/?)(\w+)([^>]*)>',
        lambda m: m.group(0) if m.group(2).lower() in _ALLOWED_TAGS else '',
        text,
    )
    return text


async def reply_llm(message: Message, question: str, graph, state: FSMContext) -> None:
    status_msg = await message.answer("Шукаю відповідь... ⏳")
    try:
        result = await ask_bot_async(graph, question, str(message.chat.id))
        reply_markup = None
        if result["response_type"] == "brief_ready":
            await state.set_state(BriefFSM.brief_ready)
            reply_markup = brief_ready_keyboard()
        await status_msg.edit_text(
            sanitize_telegram_html(result["response_text"]),
            reply_markup=reply_markup,
        )
    except Exception as e:
        logger.error(f"LLM reply failed: {e}")
        await status_msg.edit_text("Виникла помилка при отриманні відповіді ❌", parse_mode=None)