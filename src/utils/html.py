import re

_ALLOWED_TAGS = {'b', 'i', 'a', 'code', 'pre'}


def sanitize_telegram_html(text: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', text)

    # Strip tags not supported by Telegram (preserve their inner content)
    text = re.sub(
        r'<(/?)(\w+)([^>]*)>',
        lambda m: m.group(0) if m.group(2).lower() in _ALLOWED_TAGS else '',
        text,
    )

    # Close any tags the LLM left open, using a stack
    stack: list[str] = []
    for m in re.finditer(r'<(/?)(\w+)[^>]*>', text):
        tag = m.group(2).lower()
        if tag not in _ALLOWED_TAGS:
            continue
        if m.group(1):  # closing tag
            if stack and stack[-1] == tag:
                stack.pop()
        else:  # opening tag
            stack.append(tag)

    for tag in reversed(stack):
        text += f'</{tag}>'

    return text
