from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import config
from src.secrets import secrets
from src.llm.state import AgentState
from src.llm.retriever import get_retriever
from src.utils.brief import format_brief_state, update_list, is_str_complete, is_list_complete
from src.utils.docs import format_docs
from src.utils.logger import get_logger

logger = get_logger(__name__)

_HTML_FORMAT_RULE = (
    "Використовуй ТІЛЬКИ ці Telegram HTML теги: <b>, <i>.\n"
    "НЕ використовуй <br>, <p>, <div>, <span>, <h1>–<h6> або будь-які інші HTML теги.\n"
)


# ── Structured output schemas ──────────────────────────────────────────────

class RouterOutput(BaseModel):
    has_question: bool = Field(
        description="Message contains a question about company services, technology, portfolio, team, etc."
    )
    has_project_info: bool = Field(
        description="Message contains ANY information useful for a project brief"
    )
    is_nonsense: bool = Field(
        description="Message is completely irrelevant (weather, unrelated prices, etc.). "
                    "Mutually exclusive with has_question and has_project_info."
    )


class BriefUpdateOutput(BaseModel):
    project_type: str | None = Field(default=None)
    project_description: str | None = Field(default=None)
    goals: list[str] | None = Field(
        default=None,
        description="NEW goal items to add. Atomic and specific. null if nothing to add.",
    )
    remove_goals: list[str] | None = Field(
        default=None,
        description="Exact goal items to remove (copy text verbatim from brief state). null if nothing to remove.",
    )
    key_features: list[str] | None = Field(
        default=None,
        description="NEW key feature items to add. null if nothing to add.",
    )
    remove_key_features: list[str] | None = Field(
        default=None,
        description="Exact key feature items to remove (copy text verbatim from brief state). null if nothing to remove.",
    )
    additional_features: list[str] | None = Field(
        default=None,
        description="NEW additional feature items to add. null if nothing to add.",
    )
    remove_additional_features: list[str] | None = Field(
        default=None,
        description="Exact additional feature items to remove (copy text verbatim from brief state). null if nothing to remove.",
    )
    integrations: list[str] | None = Field(
        default=None,
        description="NEW integration items to add. null if nothing to add.",
    )
    remove_integrations: list[str] | None = Field(
        default=None,
        description="Exact integration items to remove (copy text verbatim from brief state). null if nothing to remove.",
    )
    client_materials: list[str] | None = Field(
        default=None,
        description="NEW client material items to add. null if nothing to add.",
    )
    remove_client_materials: list[str] | None = Field(
        default=None,
        description="Exact client material items to remove (copy text verbatim from brief state). null if nothing to remove.",
    )


# ── LLM factory ───────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _get_llm(temperature: float) -> ChatGroq:
    return ChatGroq(
        temperature=temperature,
        groq_api_key=secrets.groq_api_key,
        model_name=config.llm.model,
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_last_human(state: AgentState) -> HumanMessage:
    return next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def _get_last_ai(state: AgentState) -> AIMessage | None:
    return next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)


def _build_history(state: AgentState, n: int = 4) -> tuple[str, list]:
    """Return (summary, last_n_messages_before_current_human)."""
    last_human = _get_last_human(state)
    summary = state.get("summary", "") or ""
    history = [m for m in state["messages"] if m is not last_human][-n:]
    return summary, history


# ── NODE 1: router_node ───────────────────────────────────────────────────

async def router_node(state: AgentState) -> dict:
    last_human = _get_last_human(state)
    llm = _get_llm(0.0).with_structured_output(RouterOutput, method="json_schema")

    system = (
        "Класифікуй намір ОСТАННЬОГО повідомлення користувача. Поверни JSON об'єкт.\n\n"
        "has_question: true якщо повідомлення містить питання про компанію — "
        "її послуги, технології, портфоліо, команду, ціни, контакти або будь-яку іншу тему про компанію.\n"
        "has_project_info: true якщо повідомлення містить БУДЬ-ЯКУ інформацію корисну "
        "для брифу проєкту: тип проєкту, бажаний функціонал, цілі, інтеграції, "
        "цільова аудиторія, бюджет, терміни або відповіді на попередні питання брифу.\n"
        "is_nonsense: true ТІЛЬКИ якщо повідомлення абсолютно нерелевантне — погода, "
        "загальні знання, ціни на непов'язані товари, випадковий чат. "
        "Взаємовиключне з has_question та has_project_info.\n\n"
        "ВАЖЛИВО: has_question та has_project_info можуть бути true одночасно.\n\n"
        "КОНТЕКСТ РОЗМОВИ: нижче наведено останні повідомлення розмови перед поточним. "
        "Переглянь їх, щоб знайти найближче питання асистента, на яке відповідає користувач. "
        "Якщо останнє повідомлення користувача є відповіддю на питання асистента про проєкт — "
        "це has_project_info, навіть якщо відповідь коротка ('так', 'ні', 'перший варіант' тощо). "
        "Якщо асистент не ставив питання (наприклад, надіслав готовий бриф або підтвердження) — "
        "класифікуй повідомлення користувача самостійно за його змістом."
    )

    _, history = _build_history(state)

    result: RouterOutput = await llm.ainvoke([
        SystemMessage(system),
        *history,
        HumanMessage(last_human.content),
    ])
    logger.info(
        f"Router: has_question={result.has_question}, "
        f"has_project_info={result.has_project_info}, "
        f"is_nonsense={result.is_nonsense}"
    )
    return {
        "has_question": result.has_question,
        "has_project_info": result.has_project_info,
        "is_nonsense": result.is_nonsense,
        "qna_response": None,  # Clear stale value from previous turn
    }


# ── NODE 2: qna_node ─────────────────────────────────────────────────────

async def qna_node(state: AgentState) -> dict:
    last_human = _get_last_human(state)
    retriever = get_retriever()
    docs = await retriever.ainvoke(f"query: {last_human.content}")
    context = format_docs(docs)

    summary, history = _build_history(state)
    system = (
            "Ви — AI-асистент компанії. Відповідайте на питання користувача ТІЛЬКИ на основі "
            "наданого контексту про компанію. Відповідь має бути чіткою та корисною.\n"
            "НЕ ставте уточнюючих питань і НЕ згадуйте бриф проєкту.\n\n"
            "ВАЖЛИВІ ОБМЕЖЕННЯ:\n"
            "• Якщо користувач питає про вартість, ціни або бюджет проєкту — не називайте жодних цифр. "
            "Поясніть що точну вартість може визначити лише комерційний відділ після детального "
            "обговорення вимог, і запропонуйте зв'язатись напряму.\n"
            "• Не розкривайте контактну інформацію (телефони, email) у відповіді.\n\n"
            "═══ КОНТЕКСТ ПРО КОМПАНІЮ ═══\n"
            f"{context}\n"
            + (f"\n═══ СУМАРИЗАЦІЯ ПОПЕРЕДНЬОЇ РОЗМОВИ ═══\n{summary}\n" if summary else "")
            + "\nМова відповіді: виключно українська. "
              "Форматування: Telegram HTML (<b>, <i>). Не використовуйте Markdown (**, __ тощо).\n"
            + _HTML_FORMAT_RULE
    )

    response = await _get_llm(0.3).ainvoke([SystemMessage(system), *history, HumanMessage(last_human.content)])
    logger.info("qna_node completed")
    return {"qna_response": response.content}


# ── NODE 3: update_brief_node ─────────────────────────────────────────────

async def update_brief_node(state: AgentState) -> dict:
    last_human = _get_last_human(state)
    brief_state = format_brief_state(state)
    summary, history = _build_history(state)
    llm = _get_llm(0.0).with_structured_output(BriefUpdateOutput, method="json_schema")

    system = (
        "Оновіть бриф проєкту на основі повідомлення користувача.\n\n"
        "Є два типи дій:\n"
        "• ДОДАТИ — якщо користувач надає нову інформацію (відповідає на питання, уточнює деталі)\n"
        "• ВИДАЛИТИ — якщо користувач просить прибрати або змінити вже наявну інформацію\n\n"
        "ПРАВИЛА ДОДАВАННЯ:\n"
        "• Витягуйте ТІЛЬКИ інформацію, яку користувач явно написав.\n"
        "• НЕ робіть висновків, НЕ припускайте, НЕ вигадуйте дані.\n"
        "• Використовуйте історію розмови як контекст для інтерпретації коротких відповідей:\n"
        "  — Якщо бот запитав про ОС і користувач відповів 'яблуко' → iOS/Apple\n"
        "  — Якщо бот запитав про авторизацію і користувач відповів 'номер' → авторизація за номером телефону\n"
        "• Рядкові поля: якщо користувач відповів 'не знаю', 'без різниці', 'не важливо' → 'не визначено'\n"
        "• Поля-списки: якщо користувач відповів 'не потрібно', 'немає' → ['не визначено']\n"
        "• Для полів без нової інформації поверніть null\n"
        "• Деталізуйте атомарно і конкретно (не 'авторизація', а 'Авторизація за номером телефону')\n"
        "• Для полів-списків повертайте ТІЛЬКИ нові пункти, яких ще немає в поточному стані\n\n"
        "ПРАВИЛА ВИДАЛЕННЯ:\n"
        "• Якщо користувач просить прибрати або змінити пункт — скопіюйте його текст ДОСЛІВНО з брифу\n"
        "  в поле remove_*. Якщо змінює (замінює) — одночасно додайте новий варіант і видаліть старий.\n"
        "• Не видаляйте нічого, про що користувач явно не просив.\n\n"
        f"═══ ПОТОЧНИЙ СТАН БРИФУ ═══\n{brief_state}"
        + (f"\n═══ СУМАРИЗАЦІЯ ПОПЕРЕДНЬОЇ РОЗМОВИ ═══\n{summary}\n" if summary else "")
    )

    logger.info(
        f"update_brief_node state received | "
        f"project_type={state.get('project_type')!r} | "
        f"goals={state.get('goals')} | "
        f"key_features={state.get('key_features')} | "
        f"additional_features={state.get('additional_features')}"
    )

    result: BriefUpdateOutput = await llm.ainvoke([
        SystemMessage(system),
        *history,
        HumanMessage(last_human.content),
    ])
    logger.info(f"update_brief_node | add: key_features={result.key_features} | remove: {result.remove_key_features}")

    return {
        "project_type": result.project_type if result.project_type is not None else state.get("project_type"),
        "project_description": result.project_description if result.project_description is not None else state.get("project_description"),
        "goals": update_list(state.get("goals") or [], result.goals, result.remove_goals),
        "key_features": update_list(state.get("key_features") or [], result.key_features, result.remove_key_features),
        "additional_features": update_list(state.get("additional_features") or [], result.additional_features, result.remove_additional_features),
        "integrations": update_list(state.get("integrations") or [], result.integrations, result.remove_integrations),
        "client_materials": update_list(state.get("client_materials") or [], result.client_materials, result.remove_client_materials),
    }


# ── NODE 4: validation_node ───────────────────────────────────────────────

async def validation_node(state: AgentState) -> dict:
    logger.info(
        f"validation INPUT: "
        f"project_type={state.get('project_type')}, "
        f"goals={state.get('goals')}, "
        f"key_features={state.get('key_features')}, "
        f"additional_features={state.get('additional_features')}, "
        f"integrations={state.get('integrations')}, "
        f"client_materials={state.get('client_materials')}"
    )

    empty_fields: list[str] = []

    if not is_str_complete(state.get("project_type")):
        empty_fields.append("Тип проєкту")
    if not is_str_complete(state.get("project_description")):
        empty_fields.append("Опис проєкту")

    list_checks = [
        ("Цілі", "goals", 1),
        ("Ключовий функціонал", "key_features", 2),
        ("Додатковий функціонал", "additional_features", 1),
        ("Інтеграції", "integrations", 1),
        ("Матеріали від клієнта", "client_materials", 1),
    ]
    for name, field, min_items in list_checks:
        if not is_list_complete(state.get(field) or [], min_items):
            empty_fields.append(name)

    brief_status = "complete" if not empty_fields else "in_progress"

    logger.info(f"validation_node: status={brief_status}, empty={empty_fields}")
    return {
        "brief_status": brief_status,
        "empty_fields": empty_fields,
        "weak_fields": [],
    }


# ── NODE 5: clarifying_node ───────────────────────────────────────────────

async def clarifying_node(state: AgentState) -> dict:
    brief_state = format_brief_state(state)
    empty_fields = state.get("empty_fields") or []
    qna_response = state.get("qna_response")
    project_type = state.get("project_type")
    summary, history = _build_history(state)

    # Ask about at most 1 field when project type is unknown, else 2 fields
    fields_to_ask = empty_fields[:1] if not project_type else empty_fields[:2]
    fields_info = f"Відсутні поля: {', '.join(fields_to_ask)}\n" if fields_to_ask else ""

    qna_block = ""
    if qna_response:
        qna_block = (
            "ВАЖЛИВО: У відповіді СПОЧАТКУ включіть наступний текст дослівно, "
            "а потім через порожній рядок задайте уточнюючі питання:\n\n"
            f"{qna_response}\n\n"
        )

    if not project_type:
        task_instruction = (
            "Тип проєкту ще невідомий. Запитайте що хоче створити користувач і "
            "запропонуйте конкретні варіанти:\n"
            "1. Telegram-бот\n"
            "2. Веб-сайт або веб-застосунок\n"
            "3. Мобільний застосунок (iOS, Android)\n"
            "4. CRM або внутрішня система для бізнесу\n"
            "Сформулюйте питання природньо, без нумерації у відповіді — просто перелічте варіанти.\n"
        )
    else:
        task_instruction = (
            f"Тип проєкту відомий: {project_type}. Дійте як досвідчений бізнес-аналітик.\n\n"
            "Для кожного відсутнього поля:\n"
            "1. Запропонуйте 2-3 конкретних типових варіанти саме для цього типу проєкту\n"
            "2. Задайте чітке питання\n"
            "НЕ пояснюйте чому це питання важливе — просто питайте природньо.\n\n"
            "Приклад для авторизації в університетському боті:\n"
            "<i>Як користувачі будуть входити в систему? Для університетських ботів зазвичай:\n"
            "• Номер телефону — ділиться через Telegram, звіряється з базою університету\n"
            "• Студентський ID — вводять вручну\n"
            "• SSO університету — якщо є корпоративна система\n"
            "Який варіант підходить?</i>\n\n"
            "Задавай питання максимум про 1-2 незаповнених поля за раз.\n"
            "Не намагайся заповнити всі поля в одному повідомленні.\n"
            "Обери найважливіші поля і задай питання тільки про них.\n"
            "В наступному повідомленні перейдеш до інших полів.\n"
            "НЕ питайте про поля зі значенням 'не визначено' — вони вже визначені.\n"
        )

    system = (
        "Ви — AI-асистент компанії, що збирає бриф проєкту.\n\n"
        f"{qna_block}"
        f"{task_instruction}\n"
        f"Поля для уточнення:\n{fields_info}\n"
        f"═══ ПОТОЧНИЙ СТАН БРИФУ (тільки для вас) ═══\n{brief_state}\n\n"
        + (f"═══ СУМАРИЗАЦІЯ ПОПЕРЕДНЬОЇ РОЗМОВИ ═══\n{summary}\n\n" if summary else "")
        + "Правила форматування:\n"
        "• Тільки українська мова\n"
        "• Telegram HTML: <b>жирний</b>, <i>курсив</i>\n"
        "• НЕ використовуйте Markdown (**, __ тощо)\n"
        + _HTML_FORMAT_RULE +
        "• НЕ називайте технічних назв полів (project_type, goals тощо)\n"
        "• НЕ показуйте стан брифу або назви полів у відповіді\n"
    )

    response = await _get_llm(0.7).ainvoke([SystemMessage(system), *history])
    logger.info("clarifying_node completed")
    return {
        "messages": [AIMessage(content=response.content)],
        "response_type": "brief_clarifying",
        "qna_response": None,
    }


# ── NODE 6: brief_format_node ─────────────────────────────────────────────

async def brief_format_node(state: AgentState) -> dict:
    brief_state = format_brief_state(state)

    system = (
        "Відформатуйте повний детальний бриф проєкту для відображення користувачу.\n\n"
        "Правила:\n"
        "• Кожна секція — детальний параграф або список, не одне слово\n"
        "• Ключовий функціонал: кожен пункт — повне описове речення\n"
        "• Тільки українська мова\n"
        "• Telegram HTML: <b>жирний</b>, <i>курсив</i>\n"
        "• НЕ використовуйте Markdown (**, __ тощо)\n"
        + _HTML_FORMAT_RULE +
        "\n"
        "Використовуйте цей точний формат:\n\n"
        "<b>Бриф проєкту</b>\n\n"
        "<b>Тип проєкту:</b> [значення]\n\n"
        "<b>Опис проєкту:</b>\n[2-3 речення з контекстом]\n\n"
        "<b>Цілі:</b>\n[яку проблему вирішує, хто отримує користь]\n\n"
        "<b>Ключовий функціонал:</b>\n• [повне описове речення]\n• [ще один пункт]\n\n"
        "<b>Додатковий функціонал:</b>\n• [пункти або 'На старті не передбачено']\n\n"
        "<b>Інтеграції:</b>\n[опис або 'Додаткові інтеграції не плануються']\n\n"
        "<b>Матеріали від клієнта:</b>\n• [конкретні матеріали]\n\n"
        f"═══ ДАНІ БРИФУ ═══\n{brief_state}"
    )

    response = await _get_llm(0.3).ainvoke([SystemMessage(system)])
    logger.info("brief_format_node completed")
    return {
        "messages": [AIMessage(content=response.content)],
        "response_type": "brief_ready",
        "qna_response": None,
    }


# ── NODE 7: nonsense_node ─────────────────────────────────────────────────

async def nonsense_node(state: AgentState) -> dict:
    last_human = _get_last_human(state)
    system = (
        "Користувач надіслав повідомлення, яке не стосується роботи бота. "
        "Ввічливо і коротко поясніть чим ви можете допомогти:\n"
        "• Відповіді на питання про компанію (послуги, технології, портфоліо)\n"
        "• Збір брифу для нового проєкту\n\n"
        "Будьте доброзичливими і лаконічними. "
        "Тільки українська мова. Telegram HTML. Не Markdown.\n"
        + _HTML_FORMAT_RULE
    )
    response = await _get_llm(0.5).ainvoke([
        SystemMessage(system),
        HumanMessage(last_human.content),
    ])
    logger.info("nonsense_node completed")
    return {
        "messages": [AIMessage(content=response.content)],
        "response_type": "brief_clarifying",
        "qna_response": None,
    }


# ── NODE 8: estimation_node ───────────────────────────────────────────────

async def estimation_node(state: AgentState) -> dict:
    brief_state = format_brief_state(state)

    system = (
        "Ти — досвідчений Tech Lead компанії з розробки ПЗ.\n"
        "На основі брифу проєкту надай детальну оцінку часу розробки по стадіях.\n\n"
        "ОБОВ'ЯЗКОВІ стадії (завжди присутні):\n"
        "- Pre-project work (UX) — уточнення вимог, сценарії, планування\n"
        "- Design (UI) — підготовка інтерфейсів, макети\n"
        "- Development — основна розробка\n"
        "- Testing — функціональне та інтеграційне тестування\n\n"
        "ДОДАТКОВІ стадії (додай якщо доречно для цього типу проєкту):\n"
        "- DevOps / Deployment — якщо проєкт потребує складної інфраструктури\n"
        "- ML / AI Integration — якщо є AI компоненти\n"
        "- Інші стадії на твій розсуд як Tech Lead\n\n"
        "Для кожної стадії вкажи:\n"
        "- Назву стадії (технічні абревіатури не перекладати: DevOps, UX, UI, API тощо)\n"
        "- Опис робіт — конкретно що робиться на цій стадії для ЦЬОГО проєкту\n"
        "- Години: Min – Max\n\n"
        "ВАЖЛИВО щодо діапазону годин:\n"
        "- Чим розмитіший і нечіткий бриф — тим більша різниця між Min і Max\n"
        "- Чим конкретніші вимоги — тим менша різниця\n\n"
        "Формат відповіді (Telegram HTML):\n\n"
        "<b>Оцінка часу розробки</b>\n\n"
        "<b>Стадія: Pre-project work (UX)</b>\n"
        "<i>Роботи:</i> [конкретний опис для цього проєкту]\n"
        "<i>Години:</i> X – Y\n\n"
        "... (інші стадії)\n\n"
        "<b>Разом: приблизно X – Y годин</b>\n"
        "<b>Термін: приблизно X – Y місяців</b>\n\n"
        "<i>[Дисклеймер що оцінка орієнтовна, точні терміни і вартість "
        "визначає комерційний відділ після детального обговорення]</i>\n\n"
        + _HTML_FORMAT_RULE +
        f"\n═══ БРИФ ПРОЄКТУ ═══\n{brief_state}"
    )

    response = await _get_llm(0.3).ainvoke([SystemMessage(system)])
    logger.info("estimation_node completed")
    return {
        "messages": [AIMessage(content=response.content)],
        "response_type": "estimation",
        "estimation": response.content,
    }


# ── Summarize node ────────────────────────────────────────────────────────

async def summarize_node(state: AgentState) -> dict:
    messages = state["messages"]
    # Keep the newest keep_messages; compress everything older
    old_messages = messages[:-config.memory.keep_messages]
    existing_summary = state.get("summary", "") or ""

    formatted = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old_messages
    )
    summary_prompt = (
            f"Створіть стислу сумаризацію розмови не більше {config.memory.summary_max_sentences} речень.\n"
            + (f"Існуюча сумаризація: {existing_summary}\n\n" if existing_summary else "")
            + f"Повідомлення для сумаризації:\n{formatted}\n\nСумаризація:"
    )

    response = await _get_llm(config.llm.temperature).ainvoke([HumanMessage(summary_prompt)])
    logger.info(f"Summarized {len(old_messages)} old messages, keeping last {config.memory.keep_messages}")

    return {
        "messages": [RemoveMessage(id=m.id) for m in old_messages],
        "summary": response.content,
    }
