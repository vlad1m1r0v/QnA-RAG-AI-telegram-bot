import phonenumbers
from phonenumbers import NumberParseException


def normalize_phone(raw: str) -> str | None:
    """Parse and normalize a phone number to E164 format. Returns None if invalid."""
    try:
        parsed = phonenumbers.parse(raw, None)
        if not phonenumbers.is_valid_number(parsed):
            return None
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except NumberParseException:
        return None