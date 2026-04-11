"""Email sending via Mailjet REST API.

mailjet_rest is a synchronous client; calls are wrapped in
asyncio.run_in_executor() to avoid blocking the event loop.
"""

import asyncio
import base64
import functools
from concurrent.futures import ThreadPoolExecutor

from mailjet_rest import Client

from src.secrets import secrets
from src.utils.logger import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mailjet")


def _build_client() -> Client:
    return Client(
        auth=(secrets.mailjet_api_key, secrets.mailjet_secret_key),
        version="v3.1",
    )


def _send_sync(client_name: str, client_phone: str, pdf_bytes: bytes) -> bool:
    """Synchronous Mailjet send — runs in a thread executor."""
    client = _build_client()
    pdf_b64 = base64.b64encode(pdf_bytes).decode()

    data = {
        "Messages": [
            {
                "From": {
                    "Email": secrets.default_email,
                    "Name": "Project Brief Bot",
                },
                "To": [
                    {
                        "Email": secrets.default_email,
                        "Name": "Manager",
                    }
                ],
                "Subject": f"Новий бриф від клієнта: {client_name}",
                "TextPart": (
                    f"Клієнт: {client_name}\n"
                    f"Телефон: {client_phone}\n\n"
                    "Бриф проєкту у вкладенні."
                ),
                "Attachments": [
                    {
                        "ContentType": "application/pdf",
                        "Filename": f"brief_{client_name}.pdf",
                        "Base64Content": pdf_b64,
                    }
                ],
            }
        ]
    }

    response = client.send.create(data=data)
    status = response.status_code

    if status == 200:
        logger.info(f"Brief email sent successfully for client '{client_name}'")
        return True

    logger.error(f"Mailjet returned status {status}: {response.json()}")
    return False


async def send_brief_email(client_name: str, client_phone: str, pdf_bytes: bytes) -> bool:
    """Send brief PDF to the default manager email. Returns True on success."""
    loop = asyncio.get_event_loop()
    fn = functools.partial(_send_sync, client_name, client_phone, pdf_bytes)
    try:
        return await loop.run_in_executor(_executor, fn)
    except Exception as e:
        logger.error(f"send_brief_email failed: {e}")
        return False