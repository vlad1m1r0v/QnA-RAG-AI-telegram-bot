"""PDF generation for project briefs.

Cyrillic support: uses DejaVu Sans TTF (derived from Bitstream Vera, adds full
Cyrillic coverage). Searched from common system font paths — not committed to
the repo. Bitstream Vera fonts bundled with reportlab itself do NOT include
Cyrillic glyphs.
"""

import io
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# DejaVu Sans supports Cyrillic; search common paths across Linux distros.
_FONT_SEARCH = [
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]
_BOLD_FONT_SEARCH = [
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def _find_font(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _register_fonts() -> tuple[str, str]:
    """Register Cyrillic-capable fonts and return (normal, bold) names."""
    regular = _find_font(_FONT_SEARCH)
    bold = _find_font(_BOLD_FONT_SEARCH)

    if regular:
        pdfmetrics.registerFont(TTFont("BriefFont", regular))
        logger.info(f"PDF: registered regular font from {regular}")
    else:
        logger.warning("PDF: DejaVu Sans not found; Cyrillic may not render — using Helvetica fallback")

    if bold:
        pdfmetrics.registerFont(TTFont("BriefFontBold", bold))
        logger.info(f"PDF: registered bold font from {bold}")
    else:
        logger.warning("PDF: DejaVu Sans Bold not found; using Helvetica-Bold fallback")

    return (
        "BriefFont" if regular else "Helvetica",
        "BriefFontBold" if bold else "Helvetica-Bold",
    )


# Field metadata: (state_key, display_label, is_list)
_BRIEF_FIELDS: list[tuple[str, str, bool]] = [
    ("project_type", "Тип проєкту", False),
    ("project_description", "Опис проєкту", False),
    ("goals", "Цілі", True),
    ("key_features", "Ключовий функціонал", True),
    ("additional_features", "Додатковий функціонал", True),
    ("integrations", "Інтеграції", True),
    ("client_materials", "Матеріали від клієнта", True),
]


def generate_pdf(agent_state: dict, client_name: str, client_phone: str) -> bytes:
    """Generate a project brief PDF and return its bytes."""
    font, bold_font = _register_fonts()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    title_style = ParagraphStyle(
        "Title",
        fontName=bold_font,
        fontSize=16,
        leading=20,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        fontName=bold_font,
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName=font,
        fontSize=10,
        leading=14,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName=font,
        fontSize=10,
        leading=14,
    )

    story = []

    # Title
    story.append(Paragraph("Бриф проєкту", title_style))
    story.append(Spacer(1, 0.3 * cm))

    # Client section
    story.append(Paragraph("Клієнт", section_style))
    story.append(Paragraph(f"Ім'я: {client_name}", body_style))
    story.append(Paragraph(f"Телефон: {client_phone}", body_style))
    story.append(Spacer(1, 0.3 * cm))

    # Brief fields
    for key, label, is_list in _BRIEF_FIELDS:
        value = agent_state.get(key)

        if is_list:
            items = value or []
            # Skip empty or placeholder-only lists
            if not items or items == ["не визначено"]:
                continue
            story.append(Paragraph(label, section_style))
            bullet_items = [
                ListItem(Paragraph(item, bullet_style), bulletColor="black")
                for item in items
            ]
            story.append(ListFlowable(bullet_items, bulletType="bullet", leftIndent=12))
        else:
            if not value or value == "не визначено":
                continue
            story.append(Paragraph(label, section_style))
            story.append(Paragraph(value, body_style))

    doc.build(story)
    return buf.getvalue()