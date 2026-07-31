"""Best-effort SMTP mailer for call summaries / escalation alerts / callbacks.

Synchronous smtplib run in a worker thread so it never blocks the asyncio loop,
and every failure is swallowed + logged — a mail problem must never affect a
live call (spec §9 resilience). Disabled (a no-op) until SMTP is configured.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage

logger = logging.getLogger(__name__)


class Mailer:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        sender: str,
        recipients: list[str],
        use_tls: bool = True,
        verify_cert: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._sender = sender
        self._recipients = [r for r in recipients if r]
        self._use_tls = use_tls
        self._verify_cert = verify_cert
        self._timeout = timeout
        # Enabled once we can connect + send; recipients may be supplied per-send
        # (e.g. department routing), so a global recipient list isn't required.
        self._enabled = bool(host and sender)
        if not self._enabled:
            logger.info("Mailer disabled (SMTP host/sender not set).")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send(
        self, subject: str, body: str, recipients: list[str] | None = None
    ) -> bool:
        """Send an email; never raises. Returns True if it was actually sent.

        `recipients` overrides the default list (used for per-department routing);
        if it resolves to empty, nothing is sent.
        """
        to = [r for r in (recipients if recipients is not None else self._recipients) if r]
        if not (self._enabled and to):
            return False
        try:
            await asyncio.to_thread(self._send_sync, subject, body, to)
            logger.info("Sent email to %s: %s", ", ".join(to), subject)
            return True
        except Exception:  # noqa: BLE001 - mail must never break a call
            logger.exception("Email send failed: %s", subject)
            return False

    def _send_sync(self, subject: str, body: str, recipients: list[str]) -> None:
        msg = EmailMessage()
        msg["From"] = self._sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
            if self._use_tls:
                ctx = ssl.create_default_context()
                if not self._verify_cert:
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                smtp.starttls(context=ctx)
            if self._username:
                smtp.login(self._username, self._password)
            smtp.send_message(msg)
