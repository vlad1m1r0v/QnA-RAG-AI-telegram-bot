from aiogram.fsm.state import State, StatesGroup


class BriefFSM(StatesGroup):
    idle = State()
    brief_ready = State()
    estimation = State()
    awaiting_contact = State()