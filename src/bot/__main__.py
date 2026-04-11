import asyncio

from aiogram import Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.pymongo import PyMongoStorage
from langgraph.checkpoint.mongodb import MongoDBSaver

from src.bot.handlers.callbacks import handle_estimate, handle_send_brief
from src.bot.handlers.commands import cmd_start, cmd_reset_memory
from src.bot.handlers.messages import (
    handle_audio,
    handle_contact,
    handle_text,
    handle_text_awaiting_contact,
)
from src.bot.instance import bot
from src.bot.states import BriefFSM
from src.llm.graph import build_graph
from src.secrets import secrets


async def main():
    fsm_storage = PyMongoStorage.from_url(secrets.mongodb_url)
    dp = Dispatcher(storage=fsm_storage)

    # State-specific handlers must be registered before generic ones.
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_reset_memory, Command("reset-memory"))
    dp.callback_query.register(handle_estimate, F.data == "gen_estimate")
    dp.callback_query.register(handle_send_brief, F.data == "send_brief")
    dp.message.register(handle_contact, BriefFSM.awaiting_contact, F.contact)
    dp.message.register(handle_text_awaiting_contact, BriefFSM.awaiting_contact, F.text)
    dp.message.register(handle_text, F.text)
    dp.message.register(handle_audio, F.voice | F.audio)

    with MongoDBSaver.from_conn_string(secrets.mongodb_url) as saver:
        graph = build_graph(saver)
        try:
            await dp.start_polling(bot, graph=graph)
        finally:
            await fsm_storage.close()


if __name__ == "__main__":
    asyncio.run(main())