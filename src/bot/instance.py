from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from src.secrets import secrets

bot = Bot(
    token=secrets.telegram_bot_key,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)