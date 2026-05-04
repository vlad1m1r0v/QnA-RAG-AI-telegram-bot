def validate_full_name(text: str) -> str | None:
    """Return cleaned full name if it has ≥2 parts each ≥2 chars, else None."""
    parts = text.strip().split()
    if len(parts) < 2 or any(len(p) < 2 for p in parts):
        return None
    return " ".join(parts)