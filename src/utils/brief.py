from src.llm.state import AgentState


def format_brief_state(state: AgentState) -> str:
    def fmt(val):
        if val is None:
            return "не заповнено"
        if isinstance(val, list):
            return ("• " + "\n• ".join(val)) if val else "не заповнено"
        return str(val) if val else "не заповнено"

    return (
        f"Тип проєкту: {fmt(state.get('project_type'))}\n"
        f"Опис проєкту: {fmt(state.get('project_description'))}\n"
        f"Цілі: {fmt(state.get('goals', []))}\n"
        f"Ключовий функціонал: {fmt(state.get('key_features', []))}\n"
        f"Додатковий функціонал: {fmt(state.get('additional_features', []))}\n"
        f"Інтеграції: {fmt(state.get('integrations', []))}\n"
        f"Матеріали від клієнта: {fmt(state.get('client_materials', []))}\n"
    )


def update_list(existing: list[str], add: list[str] | None, remove: list[str] | None) -> list[str]:
    """Apply additions then removals to a list field. Removal is case-insensitive."""
    result = merge_list(existing, add)
    if remove:
        remove_lower = {r.lower() for r in remove}
        result = [item for item in result if item.lower() not in remove_lower]
    return result


def merge_list(existing: list[str], new_items: list[str] | None) -> list[str]:
    """Merge two lists deduplicating items.

    Rules:
    - Real data replaces a ['не визначено'] placeholder.
    - 'не визначено' is ignored when real data already exists.
    - Otherwise new items are appended without duplicates.
    """
    if not new_items:
        return existing
    if existing == ["не визначено"]:
        return new_items
    if "не визначено" in new_items and existing:
        return existing
    return existing + [i for i in new_items if i not in existing]


def is_str_complete(val: str | None) -> bool:
    return bool(val and val != "не визначено")


def is_list_complete(val: list[str], min_items: int = 1) -> bool:
    if not val:
        return False
    if "не визначено" in val:
        return True
    return len(val) >= min_items