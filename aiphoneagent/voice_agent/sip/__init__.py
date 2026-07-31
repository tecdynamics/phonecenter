"""SIP/RTP termination. Registers ONE Zadarma extension as a UAC.

The orchestrator talks only to the `base` abstractions, so the concrete
backend (pjsua2 now, LiveKit SIP later) is swappable per spec §4.
"""
