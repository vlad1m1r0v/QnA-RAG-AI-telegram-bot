from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReplyKeyboardRemove

from src.bot.helpers import reply_llm
from src.bot.instance import bot
from src.bot.states import BriefFSM
from src.secrets import secrets
from src.utils.email import send_brief_email
from src.utils.logger import get_logger
from src.utils.name import validate_full_name
from src.utils.pdf import generate_pdf
from src.utils.phone import normalize_phone
from src.utils.transcription import transcribe_url

logger = get_logger(__name__)

_PHONE_INVALID_MSG = (
    "Номер телефону не пройшов валідацію.\n"
    "Введіть номер у міжнародному форматі, наприклад: +380991234567"
)
_NAME_INVALID_MSG = (
    "Введіть ім'я та прізвище через пробіл, наприклад: Іван Петренко\n"
    "(мінімум 2 символи кожне)"
)


async def _send_brief(message: Message, state: FSMContext, graph, full_name: str, client_phone: str) -> None:
    await message.answer("Надсилаю бриф менеджеру... ⏳", parse_mode=None)

    thread_id = str(message.chat.id)
    graph_state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    agent_state = graph_state.values if graph_state else {}

    try:
        pdf_bytes = generate_pdf(agent_state, full_name, client_phone)
        success = await send_brief_email(full_name, client_phone, pdf_bytes)
    except Exception as e:
        logger.error(f"Brief send failed: {e}")
        success = False

    await state.set_state(BriefFSM.brief_ready)
    if success:
        await message.answer("Бриф успішно надіслано менеджеру! ✅", parse_mode=None)
    else:
        await message.answer("Виникла помилка при відправці брифу ❌", parse_mode=None)


async def handle_phone_contact(message: Message, state: FSMContext, graph) -> None:
    contact = message.contact

    normalized = normalize_phone(contact.phone_number)
    if normalized is None:
        await message.answer(_PHONE_INVALID_MSG, parse_mode=None)
        return

    first = contact.first_name or ""
    last = contact.last_name or ""
    full_name = f"{first} {last}".strip()

    if len(full_name) < 2:
        await state.update_data(phone=normalized)
        await state.set_state(BriefFSM.awaiting_name)
        await message.answer(
            "Дякую! Тепер введіть ваше ім'я та прізвище через пробіл.",
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=None,
        )
        return

    await message.answer("Дякую!", reply_markup=ReplyKeyboardRemove(), parse_mode=None)
    await _send_brief(message, state, graph, full_name, normalized)


async def handle_phone_text(message: Message, state: FSMContext) -> None:
    normalized = normalize_phone(message.text.strip())
    if normalized is None:
        await message.answer(_PHONE_INVALID_MSG, parse_mode=None)
        return

    await state.update_data(phone=normalized)
    await state.set_state(BriefFSM.awaiting_name)
    await message.answer(
        "Дякую! Тепер введіть ваше ім'я та прізвище через пробіл.",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=None,
    )


async def handle_name_text(message: Message, state: FSMContext, graph) -> None:
    full_name = validate_full_name(message.text)
    if full_name is None:
        await message.answer(_NAME_INVALID_MSG, parse_mode=None)
        return

    data = await state.get_data()
    await _send_brief(message, state, graph, full_name, data["phone"])


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