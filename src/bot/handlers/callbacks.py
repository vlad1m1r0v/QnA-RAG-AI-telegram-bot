from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery
from langchain_core.messages import HumanMessage

from src.bot.helpers import sanitize_telegram_html
from src.bot.instance import bot
from src.bot.keyboards import contact_keyboard, send_brief_keyboard
from src.bot.states import BriefFSM


async def handle_estimate(callback: CallbackQuery, state: FSMContext, graph) -> None:
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer()
    await bot.send_chat_action(callback.message.chat.id, "typing")

    thread_id = str(callback.from_user.id)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="__ESTIMATE__")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    await state.set_state(BriefFSM.estimation)
    await callback.message.answer(
        sanitize_telegram_html(result["messages"][-1].content),
        parse_mode="HTML",
        reply_markup=send_brief_keyboard(),
    )


async def handle_send_brief(callback: CallbackQuery, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state not in (BriefFSM.brief_ready.state, BriefFSM.estimation.state):
        await callback.answer("Бриф ще не готовий", show_alert=True)
        return
    await callback.answer()
    await state.set_state(BriefFSM.awaiting_contact)
    await callback.message.answer(
        "Будь ласка, поділіться своїм контактом, щоб ми могли надіслати бриф менеджеру.",
        reply_markup=contact_keyboard(),
        parse_mode=None,
    )