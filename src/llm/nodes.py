from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, AIMessage
from langchain_groq import ChatGroq

from src.config import config
from src.secrets import secrets
from src.llm.schemas import RouterOutput, BriefUpdateOutput
from src.llm.state import AgentState
from src.llm.retriever import get_retriever
from src.utils.brief import (
    format_brief_state, build_history, get_last_human, get_last_ai,
    FIELD_KEY_MAP, HTML_FORMAT_RULE, is_str_complete, is_list_complete,
)
from src.utils.docs import format_docs
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── LLM factory ───────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _get_llm(temperature: float) -> ChatGroq:
    return ChatGroq(
        temperature=temperature,
        groq_api_key=secrets.groq_api_key,
        model_name=config.llm.model,
    )


# ── NODE 1: router_node ───────────────────────────────────────────────────

async def router_node(state: AgentState) -> dict:
    last_human = get_last_human(state)
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

    _, history = build_history(state)

    result: RouterOutput = await llm.ainvoke([
        SystemMessage(system),
        *history,
        HumanMessage(last_human.content),
    ])
    logger.info(
        f"router_node: has_question={result.has_question}, "
        f"has_project_info={result.has_project_info}, "
        f"is_nonsense={result.is_nonsense}"
    )
    return {
        "has_question": result.has_question,
        "has_project_info": result.has_project_info,
        "is_nonsense": result.is_nonsense,
        "qna_response": None,  # clear stale value from previous turn
    }


# ── NODE 2: qna_node ─────────────────────────────────────────────────────

async def qna_node(state: AgentState) -> dict:
    last_human = get_last_human(state)
    retriever = get_retriever()
    summary, history = build_history(state)

    logger.info(f"qna_node entry | summary={'present' if summary else 'none'} | history_len={len(history)}")

    # Rewrite context-dependent questions into self-contained search queries
    if history:
        rewrite_system = (
            "Перепиши запит користувача у самодостатній пошуковий запит без займенників "
            "та посилань на контекст розмови. Поверни ТІЛЬКИ перефразований запит без пояснень. "
            "Якщо запит вже самодостатній — поверни його без змін."
        )
        rewritten = await _get_llm(0.0).ainvoke([
            SystemMessage(rewrite_system),
            *history[-2:],
            HumanMessage(last_human.content),
        ])
        search_query = rewritten.content.strip()
        logger.info(f"qna_node: original={last_human.content!r} | rewritten={search_query!r}")
    else:
        search_query = last_human.content

    docs = await retriever.ainvoke(f"query: {search_query}")
    context = format_docs(docs)

    system = (
            "Ви — AI-асистент компанії. Відповідайте на питання користувача ТІЛЬКИ на основі "
            "наданого контексту про компанію. Відповідь має бути чіткою та корисною.\n"
            "НЕ ставте уточнюючих питань і НЕ згадуйте бриф проєкту.\n\n"
            "ВАЖЛИВІ ОБМЕЖЕННЯ:\n"
            "• Якщо користувач питає про вартість, ціни або бюджет проєкту — не називайте жодних цифр. "
            "Поясніть що точну вартість може визначити лише комерційний відділ після детального "
            "обговорення вимог, і запропонуйте зв'язатись напряму.\n"
            "• Не розкривайте контактну інформацію (телефони, email) у відповіді.\n\n"
            "• Нe повторюй інформацію яка вже була надана в попередніх повідомленнях. "
            "Якщо питання частково вже розкрите — доповни лише новою інформацією.\n\n"
            "═══ КОНТЕКСТ ПРО КОМПАНІЮ ═══\n"
            f"{context}\n"
            + (f"\n═══ СУМАРИЗАЦІЯ ПОПЕРЕДНЬОЇ РОЗМОВИ ═══\n{summary}\n" if summary else "")
            + "\nМова відповіді: виключно українська.\n"
              "СТРУКТУРА ВІДПОВІДІ:\n"
              "• Починай з одного речення що підсумовує відповідь на питання.\n"
              "• Якщо є кілька пунктів — перераховуй маркованим списком після вступного речення.\n"
              "• Якщо відповідь проста і коротка — пиши звичайним текстом без списку.\n"
              "• НЕ починай відповідь одразу зі списку — завжди починай з речення.\n\n"
            + HTML_FORMAT_RULE
    )

    response = await _get_llm(0.3).ainvoke([SystemMessage(system), *history, HumanMessage(last_human.content)])
    logger.info("qna_node completed")
    return {"qna_response": response.content}


# ── NODE 3: update_brief_node ─────────────────────────────────────────────

async def update_brief_node(state: AgentState) -> dict:
    last_human = get_last_human(state)
    last_ai = get_last_ai(state)
    brief = state.get("brief") or {}
    brief_state = format_brief_state(brief)
    summary, history = build_history(state)
    llm = _get_llm(0.0).with_structured_output(BriefUpdateOutput, method="json_schema")

    logger.info(f"update_brief_node entry | brief={brief}")

    last_ai_block = (
        f"═══ ПОПЕРЕДНЄ ПОВІДОМЛЕННЯ АСИСТЕНТА ═══\n{last_ai.content}\n\n"
        if last_ai else ""
    )

    system = (
            "Оновіть бриф проєкту на основі повідомлення користувача.\n\n"
            "ПРАВИЛА:\n"
            "• Витягуйте ТІЛЬКИ інформацію, яку користувач явно написав.\n"
            "• НЕ робіть висновків, НЕ припускайте, НЕ вигадуйте дані.\n"
            "• Використовуйте попереднє повідомлення асистента як контекст для інтерпретації коротких відповідей:\n"
            "  — Якщо бот запитав про ОС і користувач відповів 'яблуко' → iOS/Apple\n"
            "  — Якщо бот запитав про авторизацію і користувач відповів 'номер' → авторизація за номером телефону\n"
            "• Рядкові поля: якщо користувач відповів 'не знаю', 'без різниці', 'не важливо' → 'не визначено'\n"
            "• Поля-списки: НЕ записуйте ['не визначено'] самостійно — це робиться автоматично після двох відмов.\n"
            "  Якщо користувач відхиляє запропоновані варіанти → встановіть rejected_field, але залиште поле null.\n"
            "  Якщо користувач каже що певна категорія йому взагалі не потрібна і це НЕ відповідь на конкретні\n"
            "  запропоновані варіанти (наприклад, 'інтеграцій взагалі не буде') → тоді записуйте ['не визначено'].\n"
            "• Для полів без нової інформації поверніть null\n"
            "• Деталізуйте атомарно і конкретно (не 'авторизація', а 'Авторизація за номером телефону')\n"
            "• Для полів-списків — якщо є зміни, поверніть ПОВНИЙ оновлений список (не лише нові пункти)\n"
            "  Ви бачите поточний стан брифу — враховуйте наявні пункти і зберігайте їх якщо немає вказівок видалити\n"
            "• Якщо користувач змінює щось ('замість А зробимо Б') — поверніть повний список з урахуванням змін\n"
            "• НЕ пропонуйте та НЕ включайте: Яндекс.Метрика, 1С, AmoCRM та інші російські продукти\n\n"
            "ВИЗНАЧЕННЯ ВІДХИЛЕННЯ (rejected_field):\n"
            "• Якщо користувач відхиляє запропоновані варіанти ('жоден не підходить', 'нічого з цього',\n"
            "  'вже казав', 'це не те', 'ні на жоден', 'не потрібно нічого з перерахованого') —\n"
            "  проаналізуй ПОПЕРЕДНЄ ПОВІДОМЛЕННЯ АСИСТЕНТА і визнач яке поле обговорювалось.\n"
            "  Заповни rejected_field відповідною назвою поля (snake_case).\n"
            "• НЕ заповнюй rejected_field якщо користувач просто уточнює або дає нову інформацію.\n\n"
            f"{last_ai_block}"
            f"═══ ПОТОЧНИЙ СТАН БРИФУ ═══\n{brief_state}"
            + (f"\n═══ СУМАРИЗАЦІЯ ПОПЕРЕДНЬОЇ РОЗМОВИ ═══\n{summary}\n" if summary else "")
    )

    result: BriefUpdateOutput = await llm.ainvoke([
        SystemMessage(system),
        *history,
        HumanMessage(last_human.content),
    ])

    # Merge: never overwrite with None; for list fields replace entirely when provided
    new_brief: dict = {
        "project_type": result.project_type if result.project_type is not None else brief.get("project_type"),
        "project_description": result.project_description if result.project_description is not None else brief.get(
            "project_description"),
        "goals": result.goals if result.goals is not None else (brief.get("goals") or []),
        "key_features": result.key_features if result.key_features is not None else (brief.get("key_features") or []),
        "additional_features": result.additional_features if result.additional_features is not None else (
                    brief.get("additional_features") or []),
        "integrations": result.integrations if result.integrations is not None else (brief.get("integrations") or []),
        "client_materials": result.client_materials if result.client_materials is not None else (
                    brief.get("client_materials") or []),
    }

    # Rejection tracking
    rejected_options = dict(state.get("rejected_options") or {})
    valid_fields = {"project_type", "project_description", "goals", "key_features", "additional_features",
                    "integrations", "client_materials"}

    if result.rejected_field and result.rejected_field in valid_fields:
        field = result.rejected_field
        current = rejected_options.get(field, {"options": [], "counts": 0, "last_rejection_message_id": ""})

        if current.get("last_rejection_message_id") == last_human.id:
            logger.info(
                f"update_brief_node | skipping duplicate rejection for '{field}' (already counted for message {last_human.id})")
        else:
            new_counts = current["counts"] + 1

            if new_counts >= 2:
                logger.info(f"update_brief_node | auto-closing field '{field}' as 'не визначено'")
                if field in ("project_type", "project_description"):
                    new_brief[field] = "не визначено"
                else:
                    new_brief[field] = ["не визначено"]

            rejected_options[field] = {
                "options": current["options"],
                "counts": new_counts,
                "last_rejection_message_id": last_human.id,
            }
            logger.info(f"update_brief_node | rejection for '{field}': counts={new_counts}")

    logger.info(f"update_brief_node result | brief={new_brief} | rejected_options={rejected_options}")

    return {
        "brief": new_brief,
        "rejected_options": rejected_options,
    }


# ── NODE 4: validation_node ───────────────────────────────────────────────

async def validation_node(state: AgentState) -> dict:
    brief = state.get("brief") or {}
    logger.info(f"validation_node entry | brief={brief}")

    empty_fields: list[str] = []

    if not is_str_complete(brief.get("project_type")):
        empty_fields.append("Тип проєкту")
    if not is_str_complete(brief.get("project_description")):
        empty_fields.append("Опис проєкту")

    list_checks = [
        ("Цілі", "goals", 1),
        ("Ключовий функціонал", "key_features", 2),
        ("Додатковий функціонал", "additional_features", 1),
        ("Інтеграції", "integrations", 1),
        ("Матеріали від клієнта", "client_materials", 1),
    ]
    for name, field, min_items in list_checks:
        if not is_list_complete(brief.get(field) or [], min_items):
            empty_fields.append(name)

    brief_status = "complete" if not empty_fields else "in_progress"

    logger.info(f"validation_node: status={brief_status}, empty={empty_fields}")
    return {
        "brief_status": brief_status,
        "empty_fields": empty_fields,
    }


# ── NODE 5: clarifying_node ───────────────────────────────────────────────

async def clarifying_node(state: AgentState) -> dict:
    brief = state.get("brief") or {}
    brief_state = format_brief_state(brief)
    empty_fields = state.get("empty_fields") or []
    qna_response = state.get("qna_response")
    project_type = brief.get("project_type")
    summary, history = build_history(state)
    rejected_options = state.get("rejected_options") or {}

    logger.info(f"clarifying_node entry | empty_fields={empty_fields} | rejected_options={rejected_options}")

    # Filter fields auto-closed (counts >= 2)
    active_fields = [
        f for f in empty_fields
        if rejected_options.get(FIELD_KEY_MAP.get(f, f), {}).get("counts", 0) < 2
    ]

    # Max 1 field if project_type unknown, else max 2
    fields_to_ask = active_fields[:1] if not project_type else active_fields[:2]

    # Categorize by attempt count
    first_attempt = [f for f in fields_to_ask if
                     rejected_options.get(FIELD_KEY_MAP.get(f, f), {}).get("counts", 0) == 0]
    second_attempt = [f for f in fields_to_ask if
                      rejected_options.get(FIELD_KEY_MAP.get(f, f), {}).get("counts", 0) == 1]

    qna_block = ""
    if qna_response:
        qna_block = (
            "ВАЖЛИВО: У відповіді СПОЧАТКУ включіть наступний текст дослівно, "
            "а потім через порожній рядок задайте уточнюючі питання:\n\n"
            f"{qna_response}\n\n"
        )

    if not project_type:
        task_instruction = (
            "Тип проєкту ще невідомий. "
            "Запитайте користувача який тип продукту він хоче створити. "
            "Запропонуйте чотири варіанти: Telegram-бот, веб-застосунок, "
            "мобільний застосунок (iOS/Android) або CRM/внутрішня бізнес-система. "
            "Питання оформіть курсивом, варіанти — маркованим списком.\n"
        )
    else:
        first_attempt_instruction = ""
        if first_attempt:
            first_attempt_instruction = (
                f"Тип проєкту відомий: {project_type}. Дійте як досвідчений бізнес-аналітик.\n"
                f"Для кожного поля з переліку {', '.join(first_attempt)}: "
                "самостійно згенеруйте 2-3 типових варіанти саме для цього типу проєкту "
                "і задайте чітке питання. Питайте природньо — не пояснюйте чому питання важливе.\n"
                "Приклад підходу: коротко назвіть доступні варіанти для цього типу проєкту, "
                "потім запитайте який підходить. Варіанти та текст генеруйте самостійно виходячи "
                "з типу проєкту — не переносьте жодного тексту з цієї інструкції у відповідь.\n"
                "Не пропонуйте та не включайте: Яндекс, 1С, AmoCRM та інші російські продукти.\n\n"
            )

        second_attempt_instruction = ""
        if second_attempt:
            prev_opts_block = ""
            for f in second_attempt:
                key = FIELD_KEY_MAP.get(f, f)
                opts = rejected_options.get(key, {}).get("options", [])
                if opts:
                    prev_opts_block += f"Поле «{f}» — раніше пропонувалось:\n" + "\n".join(opts[:2]) + "\n\n"

            second_attempt_instruction = (
                f"Для полів {', '.join(second_attempt)} — користувач вже відхилив запропоновані варіанти.\n"
                "Визнайте що попередні варіанти не підійшли і запропонуйте зовсім інші альтернативи, "
                "або задайте відкрите питання щоб краще зрозуміти потребу.\n"
                "Не повторюйте варіанти які вже пропонувались.\n"
                + (f"Раніше пропоновані варіанти (НЕ повторювати):\n{prev_opts_block}" if prev_opts_block else "")
                + "\n"
            )

        task_instruction = (
            first_attempt_instruction
            + second_attempt_instruction
            + "Задавай питання максимум про 1-2 незаповнених поля за раз.\n"
              "Обери найважливіші поля і задай питання тільки про них.\n"
        )

    fields_info = f"Відсутні поля: {', '.join(fields_to_ask)}\n" if fields_to_ask else ""

    system = (
            "Ви — AI-асистент компанії, що збирає бриф проєкту.\n\n"
            "ВАЖЛИВО: Перед тим як формувати питання, перегляньте ВСЮ ІСТОРІЮ РОЗМОВИ нижче.\n"
            "• НЕ повторюйте варіанти або питання, які вже були задані раніше.\n"
            "• НЕ питайте про теми, які вже обговорювались, навіть якщо поле технічно порожнє.\n"
            "• Враховуйте контекст попередніх відповідей користувача.\n"
            "• НЕ питайте про поля зі значенням 'не визначено' — вони вже закриті.\n\n"
            f"{qna_block}"
            f"{task_instruction}\n"
            f"Поля для уточнення:\n{fields_info}\n"
            f"═══ ПОТОЧНИЙ СТАН БРИФУ (тільки для вас) ═══\n{brief_state}\n\n"
            + (f"═══ СУМАРИЗАЦІЯ ПОПЕРЕДНЬОЇ РОЗМОВИ ═══\n{summary}\n\n" if summary else "")
            + "Тільки українська мова.\n"
            + HTML_FORMAT_RULE
            + "• НЕ називайте технічних назв полів (project_type, goals тощо)\n"
              "• НЕ показуйте стан брифу або назви полів у відповіді\n"
    )

    response = await _get_llm(0.7).ainvoke([SystemMessage(system), *history])

    # Store offered response in rejected_options for first-attempt fields so
    # the next clarifying turn knows what was already suggested
    new_rejected_options = dict(rejected_options)
    for f in first_attempt:
        key = FIELD_KEY_MAP.get(f, f)
        current = new_rejected_options.get(key, {"options": [], "counts": 0, "last_rejection_message_id": ""})
        new_rejected_options[key] = {
            "options": [],
            "counts": current["counts"],
            "last_rejection_message_id": current.get("last_rejection_message_id", ""),
        }

    logger.info(
        f"clarifying_node completed | first_attempt={first_attempt} | "
        f"second_attempt={second_attempt} | active_fields={active_fields}"
    )
    return {
        "messages": [AIMessage(content=response.content)],
        "response_type": "brief_clarifying",
        "qna_response": None,
        "rejected_options": new_rejected_options,
    }


# ── NODE 6: brief_format_node ─────────────────────────────────────────────

async def brief_format_node(state: AgentState) -> dict:
    brief = state.get("brief") or {}
    brief_state = format_brief_state(brief)
    logger.info(f"brief_format_node entry | brief={brief}")

    system = (
            "Відформатуйте повний детальний бриф проєкту для відображення користувачу.\n\n"
            "Правила:\n"
            "• Кожна секція — детальний параграф або список, не одне слово\n"
            "• Ключовий функціонал: кожен пункт — повне описове речення\n"
            "• Тільки українська мова\n"
            + HTML_FORMAT_RULE +
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
    last_human = get_last_human(state)
    logger.info(f"nonsense_node entry | message={last_human.content!r}")
    system = (
            "Користувач надіслав повідомлення, яке не стосується роботи бота. "
            "Ввічливо і коротко поясніть чим ви можете допомогти:\n"
            "• Відповіді на питання про компанію (послуги, технології, портфоліо)\n"
            "• Збір брифу для нового проєкту\n\n"
            "Будьте доброзичливими і лаконічними. "
            "Тільки українська мова.\n"
            + HTML_FORMAT_RULE
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
    brief = state.get("brief") or {}
    brief_state = format_brief_state(brief)
    logger.info(f"estimation_node entry | brief={brief}")

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
            "Роботи: [конкретний опис для цього проєкту]\n"
            "Години: X – Y\n\n"
            "... (інші стадії)\n\n"
            "<b>Разом: приблизно X – Y годин</b>\n"
            "<b>Термін: приблизно X – Y місяців</b>\n\n"
            "[Дисклеймер що оцінка орієнтовна, точні терміни і вартість "
            "визначає комерційний відділ після детального обговорення]\n\n"
            + HTML_FORMAT_RULE +
            f"\n═══ БРИФ ПРОЄКТУ ═══\n{brief_state}"
    )

    response = await _get_llm(0.3).ainvoke([SystemMessage(system)])
    logger.info("estimation_node completed")
    return {
        "messages": [AIMessage(content=response.content)],
        "response_type": "estimation",
        "estimation": response.content,
    }


# ── NODE 9: summarize_node ────────────────────────────────────────────────

async def summarize_node(state: AgentState) -> dict:
    messages = state["messages"]
    # Compress the 5 oldest messages; result has at most 8 messages per turn
    old_messages = messages[:5]
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
    logger.info(
        f"summarize_node: compressed {len(old_messages)} messages, "
        f"{len(messages) - len(old_messages)} remaining"
    )

    return {
        "messages": [RemoveMessage(id=m.id) for m in old_messages],
        "summary": response.content,
    }
