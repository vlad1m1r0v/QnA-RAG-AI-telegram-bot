from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)


def brief_ready_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="⏱ Естімейт проєкту", callback_data="gen_estimate"),
        InlineKeyboardButton(text="📩 Відправити менеджеру", callback_data="send_brief"),
    ]])


def send_brief_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📩 Відправити менеджеру", callback_data="send_brief"),
    ]])


def contact_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📱 Поділитись контактом", request_contact=True)]],
        one_time_keyboard=True,
        resize_keyboard=True,
    )