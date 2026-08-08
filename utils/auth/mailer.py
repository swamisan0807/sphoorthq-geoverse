"""Password-reset email delivery. Real SMTP if SMTP_HOST/SMTP_USER/
SMTP_PASSWORD are set in the environment (see config/platform.yaml) - same
pattern as utils/qml's real-hardware-vs-simulator fallback: try the real
thing, and when it isn't configured, fall back to something honestly
labeled as local-only rather than silently pretending to have sent mail.

Local fallback: the message is written to datasets/metadata/outbox/ and
logged, and the caller (the forgot-password route) includes the reset link
directly in its API response so the flow is still testable end-to-end
without SMTP credentials.
"""

import logging
import os
import smtplib
import time
from dataclasses import dataclass
from email.message import EmailMessage

from utils.core.paths import METADATA_DIR

logger = logging.getLogger("geoverse.auth.mailer")

OUTBOX_DIR = METADATA_DIR / "outbox"


@dataclass
class SendResult:
    sent: bool
    note: str


def _smtp_config() -> dict | None:
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    if not (host and user and password):
        return None
    return {
        "host": host,
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": user,
        "password": password,
        "from_addr": os.environ.get("SMTP_FROM", user),
        "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower() != "false",
    }


def send_password_reset_email(to_addr: str, reset_link: str) -> SendResult:
    subject = "geoverse - reset your password"
    body = (
        f"A password reset was requested for your geoverse account.\n\n"
        f"Reset your password: {reset_link}\n\n"
        f"This link expires in 15 minutes. If you didn't request this, ignore this email."
    )

    config = _smtp_config()
    if config is None:
        return _write_to_outbox(to_addr, subject, body)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config["from_addr"]
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        with smtplib.SMTP(config["host"], config["port"], timeout=10) as smtp:
            if config["use_tls"]:
                smtp.starttls()
            smtp.login(config["user"], config["password"])
            smtp.send_message(msg)
        logger.info("password reset email sent to %s via %s", to_addr, config["host"])
        return SendResult(sent=True, note=f"emailed via {config['host']}")
    except Exception as e:
        logger.exception("SMTP send failed for %s, falling back to outbox", to_addr)
        fallback = _write_to_outbox(to_addr, subject, body)
        return SendResult(sent=False, note=f"SMTP send failed ({e}); {fallback.note}")


def _write_to_outbox(to_addr: str, subject: str, body: str) -> SendResult:
    """SMTP not configured (SMTP_HOST/SMTP_USER/SMTP_PASSWORD unset) - this
    is the honest local/dev path, not a silent no-op: the would-be email is
    persisted to disk and logged so it's inspectable."""
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    safe_addr = "".join(c if c.isalnum() or c in "@.-_" else "_" for c in to_addr)
    path = OUTBOX_DIR / f"{int(time.time())}_{safe_addr}.txt"
    path.write_text(f"To: {to_addr}\nSubject: {subject}\n\n{body}\n", encoding="utf-8")
    logger.warning(
        "SMTP not configured (set SMTP_HOST/SMTP_USER/SMTP_PASSWORD) - "
        "reset email for %s written to %s instead of actually being sent",
        to_addr,
        path,
    )
    return SendResult(sent=False, note=f"SMTP not configured; message saved to {path}")
