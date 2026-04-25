from langchain_core.messages import HumanMessage

# Maps validation_node's Ukrainian display names → internal snake_case field keys
FIELD_KEY_MAP: dict[str, str] = {
    "Тип проєкту": "project_type",
    "Опис проєкту": "project_description",
    "Цілі": "goals",
    "Ключовий функціонал": "key_features",
    "Додатковий функціонал": "additional_features",
    "Інтеграції": "integrations",
    "Матеріали від клієнта": "client_materials",
}


def format_brief_state(brief: dict) -> str:
    def fmt(val):
        if val is None:
            return "не заповнено"
        if isinstance(val, list):
            return ("• " + "\n• ".join(val)) if val else "не заповнено"
        return str(val) if val else "не заповнено"

    return (
        f"Тип проєкту: {fmt(brief.get('project_type'))}\n"
        f"Опис проєкту: {fmt(brief.get('project_description'))}\n"
        f"Цілі: {fmt(brief.get('goals', []))}\n"
        f"Ключовий функціонал: {fmt(brief.get('key_features', []))}\n"
        f"Додатковий функціонал: {fmt(brief.get('additional_features', []))}\n"
        f"Інтеграції: {fmt(brief.get('integrations', []))}\n"
        f"Матеріали від клієнта: {fmt(brief.get('client_materials', []))}\n"
    )


def build_history(state: dict, n: int = 4):
    """Return (summary, last_n_messages_before_current_human)."""
    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    summary = state.get("summary", "") or ""
    history = [m for m in state["messages"] if m is not last_human][-n:]
    return summary, history


def is_str_complete(val: str | None) -> bool:
    # "не визначено" counts as a determined answer
    return bool(val)


def is_list_complete(val: list[str], min_items: int = 1) -> bool:
    if not val:
        return False
    if "не визначено" in val:
        return True
    return len(val) >= min_items