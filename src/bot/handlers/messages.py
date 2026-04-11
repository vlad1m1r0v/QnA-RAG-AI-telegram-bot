from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReplyKeyboardRemove

from src.bot.helpers import reply_llm
from src.bot.instance import bot
from src.bot.states import BriefFSM
from src.secrets import secrets
from src.utils.email import send_brief_email
from src.utils.logger import get_logger
from src.utils.pdf import generate_pdf
from src.utils.transcription import transcribe_url

logger = get_logger(__name__)


async def handle_contact(message: Message, state: FSMContext, graph) -> None:
    await message.answer("Надсилаю бриф менеджеру... ⏳", reply_markup=ReplyKeyboardRemove())

    contact = message.contact
    first = contact.first_name or ""
    last = contact.last_name or ""
    client_name = f"{first} {last}".strip() or "Невідомий"
    client_phone = contact.phone_number

    thread_id = str(message.chat.id)
    graph_state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    agent_state = graph_state.values if graph_state else {}

    try:
        pdf_bytes = generate_pdf(agent_state, client_name, client_phone)
        success = await send_brief_email(client_name, client_phone, pdf_bytes)
    except Exception as e:
        logger.error(f"Brief send failed: {e}")
        success = False

    await state.set_state(BriefFSM.brief_ready)
    if success:
        await message.answer("Бриф успішно надіслано менеджеру! ✅", parse_mode=None)
    else:
        await message.answer("Виникла помилка при відправці брифу ❌", parse_mode=None)


async def handle_text_awaiting_contact(message: Message, state: FSMContext, graph) -> None:
    """User sent text instead of a contact — dismiss keyboard, pass to LLM, reset to idle."""
    await message.answer("Бриф може змінитись.", reply_markup=ReplyKeyboardRemove(), parse_mode=None)
    await state.set_state(BriefFSM.idle)
    await reply_llm(message, message.text, graph, state)


async def handle_text(message: Message, state: FSMContext, graph) -> None:
    await reply_llm(message, message.text, graph, state)


async def handle_audio(message: Message, state: FSMContext, graph) -> None:
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

    await reply_llm(message, transcript.text, graph, state)