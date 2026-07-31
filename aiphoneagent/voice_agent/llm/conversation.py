"""Per-call conversation state + persona prompt loading.

The persona/behaviour system prompt is an external per-client file (spec §5.4,
§11) — we load it verbatim and never bake persona text into code. We only warn
if `{PLACEHOLDER}` tokens remain unfilled, since the operator fills them before
deploy.

History is trimmed to an approximate token budget so long calls don't blow the
context window; the system prompt is always kept.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

Message = dict[str, str]

_PLACEHOLDER = re.compile(r"\{[A-Z][A-Z0-9_]*\}")

_FALLBACK_PROMPT = (
    "You are a warm, concise voice support agent answering inbound phone calls. "
    "Reply in plain spoken text only — no markdown, lists, symbols, or URLs. "
    "Keep replies to one to three short sentences. Speak the caller's language. "
    "Only state facts you have been given; if you don't have the information, say "
    "so and offer to connect the caller to a colleague."
)


def load_system_prompt(path: str, replacements: dict[str, str] | None = None) -> str:
    """Read the external persona prompt and fill {PLACEHOLDER}s from config/.env.

    `replacements` maps placeholder names (without braces) to values; only the
    keys provided are substituted. Any remaining {PLACEHOLDER}s are warned about
    so the operator knows what's still unfilled. Falls back to a minimal prompt
    if the file is missing.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(
            "System prompt '%s' not found — using a minimal fallback persona.", path
        )
        return _FALLBACK_PROMPT
    text = p.read_text(encoding="utf-8").strip()
    for key, value in (replacements or {}).items():
        text = text.replace("{" + key + "}", value)
    leftover = sorted(set(_PLACEHOLDER.findall(text)))
    if leftover:
        logger.warning(
            "System prompt still has %d unfilled placeholder(s) (set them in .env): %s",
            len(leftover),
            ", ".join(leftover),
        )
    return text


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_LANG_NAMES = {"el": "Greek", "en": "English", "ar": "Arabic"}

# Slow phone TTS — long replies lag badly, so force brevity.
_BREVITY_DIRECTIVE = (
    "\n\nBE BRIEF: Reply in ONE short sentence whenever possible (two at the very "
    "most). Ask for only one thing at a time. Long replies are slow to speak on the "
    "phone and frustrate the caller — keep every turn short and to the point."
)

# The model has no tools/RAG/hold — stop it promising background actions.
_NO_STALL_DIRECTIVE = (
    "\n\nYOU HAVE NO TOOLS AND CANNOT LOOK ANYTHING UP, put the caller on hold, "
    "or do anything after you reply — there is no system behind you to check. "
    "NEVER say 'please hold', 'let me check', 'one moment', 'I am checking', "
    "'bear with me', or imply you will come back with information. If you don't "
    "already have the answer, say so plainly and either connect the caller to a "
    "colleague or take their name and number for a callback."
)

# The fixed opening greeting is played before the LLM runs, so stop it greeting again.
_NO_REGREET_DIRECTIVE = (
    "\n\nThe opening greeting has ALREADY been spoken to the caller. Do NOT greet "
    "again, do NOT repeat your name or company, and do NOT say 'how can I help' "
    "again. Reply directly to what the caller asks."
)

# Stop the model demanding the same detail over and over (e.g. account number).
_NO_LOOP_DIRECTIVE = (
    "\n\nDON'T GET STUCK: Never ask for the same piece of information more than "
    "twice. If the caller cannot provide something — an account number, an order "
    "number, a detail you asked for — do NOT keep asking. Accept what they CAN give "
    "you (their name, their company, and the problem) and move forward: help if you "
    "can, otherwise take their details for a follow-up or offer to connect them to a "
    "colleague. Knowing the caller's COMPANY is more useful than an account number; "
    "an account or order number is optional, not a requirement to proceed."
)

# Phone numbers are routinely misheard on a phone line — force a read-back.
_CONTACT_DIRECTIVE = (
    "\n\nPHONE NUMBERS: When you take a phone number, ALWAYS read it back digit by "
    "digit and ask the caller to confirm it is correct before moving on. Numbers are "
    "often misheard — if it has the wrong number of digits, contains anything that is "
    "not a digit, or you are unsure, tell the caller it doesn't seem right and ask "
    "them to say it again slowly, then read it back again. Do NOT accept or act on a "
    "phone number until the caller has explicitly confirmed it. A Greek mobile number "
    "has 10 digits."
)

# Always-present instruction letting the model end the call cleanly.
_CLOSING_DIRECTIVE = (
    "\n\nENDING THE CALL: End ONLY when the CALLER signals they are finished — they "
    "say goodbye, or that they need nothing else. NEVER end on your own initiative, "
    "right after collecting details, or in the middle of helping. Before ending, "
    "give a warm, complete farewell (thank them and confirm the next step), and only "
    "then append the token [[HANGUP]] to that same reply — the caller never hears it. "
    "If you are not certain the caller is done, do NOT use the token; instead ask "
    "'Is there anything else I can help you with?'. "
    "ONE exception to ending on your own initiative: a sales / advertising / "
    "solicitation call you are declining — there, decline and append [[HANGUP]] in "
    "that same reply even if the caller keeps talking or asks for a person."
)


class Conversation:
    """Holds the system prompt and rolling user/assistant turns for one call."""

    def __init__(self, system_prompt: str, history_budget_tokens: int = 2000) -> None:
        self._base_system = system_prompt
        self._lang_directive = ""
        self._avail_directive = ""
        self._hours_directive = ""
        self._turns: list[Message] = []
        self._budget = history_budget_tokens

    def set_base_system(self, system_prompt: str) -> None:
        """Swap the rendered persona (e.g. to the variant naming the agent in the
        caller's language). Leaves conversation history untouched."""
        self._base_system = system_prompt

    def set_language(self, lang: str) -> None:
        """Pin the reply language for the whole call (small models drift otherwise)."""
        name = _LANG_NAMES.get(lang.lower(), lang)
        self._lang_directive = (
            f"\n\nIMPORTANT — LANGUAGE: Respond ONLY in {name} for this entire call. "
            f"Every word of every reply must be in {name}. Never switch to another language."
        )

    def set_availability(
        self,
        human_available: bool,
        departments: list[str] | None = None,
        caller_phone: str = "",
    ) -> None:
        """Tell the model whether it may transfer, and to which departments."""
        if human_available:
            depts = [d for d in (departments or []) if d]
            if depts:
                options = ", ".join(depts)
                token_help = (
                    f"end your reply with the token [[TRANSFER:<department>]], choosing the "
                    f"best department for the caller's need from: {options}. If unsure, use "
                    f"[[TRANSFER:{depts[0]}]]."
                )
            else:
                token_help = "end your reply with the token [[TRANSFER]]."
            self._avail_directive = (
                "\n\nESCALATION: A colleague IS available right now. Triage by urgency. If "
                "the caller's matter is SERIOUS, URGENT, or important — an outage, a server "
                "or system down, something broken or not working, a security issue, anything "
                "time-sensitive — or a genuine purchase/sales enquiry from a potential "
                "customer, PROACTIVELY offer to connect them to a person now (e.g. 'This "
                "sounds urgent — would you like me to connect you to our team right away?'). "
                "Only take a message for a follow-up (name, phone number, and reason) when "
                "the matter is minor and not time-sensitive, or when you offered to connect "
                "them and they declined. To transfer: ask ONCE whether they'd like to be "
                "connected; as SOON as the caller agrees — 'yes', 'connect me', or anything "
                f"clearly affirmative — say you're connecting them now and {token_help} Do "
                "NOT ask again after a yes and do NOT keep re-confirming. The caller never "
                "hears the token. NEVER transfer a sales, advertising, or solicitation "
                "caller (someone selling TO us) — decline and end the call instead (see "
                "screening)."
            )
        else:
            if caller_phone:
                phone_part = (
                    f"You already have the caller's phone number ({caller_phone}) from "
                    "caller ID — use it and just confirm it's the right number to call back. "
                    "Do NOT ask them to dictate their number."
                )
            else:
                phone_part = (
                    "Get a COMPLETE phone number — callers say numbers in pieces with pauses, "
                    "so keep gathering digits until you have the whole number (a Greek mobile "
                    "has 10 digits; ask for any missing digits) and read it back to confirm."
                )
            self._avail_directive = (
                "\n\nESCALATION: No colleague is available right now (outside opening hours). "
                "Do NOT transfer — take a message for a callback. Collect the caller's NAME, a "
                f"phone number, and the REASON. {phone_part} An email address is OPTIONAL — you "
                "may offer to take one but do not require it. Do NOT end the call until you "
                "have the name, a valid phone number, and the reason, and the caller has "
                "confirmed; then say the team will get back to them during opening hours and "
                "ask if there is anything else."
            )

    def set_hours(self, hours_text: str, is_open: bool) -> None:
        """Give the model the opening hours and current open/closed status."""
        if hours_text:
            status = "OPEN right now" if is_open else "CLOSED right now"
            self._hours_directive = (
                f"\n\nOPENING HOURS: The business is open {hours_text}, and is {status}. "
                "If the caller asks about opening hours or whether you're open, tell them this."
            )
        else:
            self._hours_directive = ""

    def _system_msg(self) -> Message:
        content = (
            self._base_system
            + self._lang_directive
            + self._hours_directive
            + self._avail_directive
            + _BREVITY_DIRECTIVE
            + _NO_STALL_DIRECTIVE
            + _NO_REGREET_DIRECTIVE
            + _NO_LOOP_DIRECTIVE
            + _CONTACT_DIRECTIVE
            + _CLOSING_DIRECTIVE
        )
        return {"role": "system", "content": content}

    def transcript(self) -> str:
        """Plain-text transcript of the call (for email summaries)."""
        label = {"user": "Caller", "assistant": "Agent"}
        return "\n".join(f"{label.get(t['role'], t['role'])}: {t['content']}" for t in self._turns)

    def add_user(self, text: str) -> None:
        self._turns.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        if text and text.strip():
            self._turns.append({"role": "assistant", "content": text})

    def messages(self) -> list[Message]:
        """Full message list for a request (system + trimmed history)."""
        self._trim()
        return [self._system_msg(), *self._turns]

    def with_user(self, text: str) -> list[Message]:
        """Message list with a one-off user turn appended but NOT stored.

        Used to seed the opening greeting without polluting history.
        """
        self._trim()
        return [self._system_msg(), *self._turns, {"role": "user", "content": text}]

    def messages_with_context(self, context: str) -> list[Message]:
        """messages() with RAG context attached to the latest user turn.

        The context augments only this request — history stays clean (the stored
        user turn is unchanged). No-op if there's no context or no user turn yet.
        """
        msgs = self.messages()
        if context and msgs and msgs[-1]["role"] == "user":
            augmented = (
                f"{msgs[-1]['content']}\n\n"
                "[KNOWLEDGE BASE — answer factual questions using ONLY the entries "
                "below; if they don't contain the answer, say you don't have it and "
                "offer a colleague. Do not read the bracketed labels aloud.]\n"
                f"{context}"
            )
            msgs[-1] = {"role": "user", "content": augmented}
        return msgs

    def _trim(self) -> None:
        def total() -> int:
            return sum(_approx_tokens(t["content"]) for t in self._turns)

        # Drop oldest turns until under budget (keep at least the latest exchange).
        while len(self._turns) > 2 and total() > self._budget:
            del self._turns[0]
        # History after the system prompt must start with a user turn — drop any
        # leading assistant turns left dangling by trimming.
        while self._turns and self._turns[0]["role"] == "assistant":
            del self._turns[0]
