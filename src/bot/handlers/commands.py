from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.bot.states import BriefFSM
from src.llm.graph import reset_state_async


async def cmd_start(message: Message, state: FSMContext, graph) -> None:
    await reset_state_async(graph, str(message.chat.id))
    await state.set_state(BriefFSM.idle)
    await message.answer(
        "Привіт. Я AI-асистент компанії. Я можу:\n"
        "• Відповідати на питання про компанію\n"
        "• Зібрати бриф вашого проєкту\n\n"
        "Надсилайте текст або голосове повідомлення 🎙",
        parse_mode=None,
    )


async def cmd_reset_memory(message: Message, state: FSMContext, graph) -> None:
    await reset_state_async(graph, str(message.chat.id))
    await state.set_state(BriefFSM.idle)
    await message.answer("Чат та бриф очищено. Починаємо з початку 🔄", parse_mode=None)