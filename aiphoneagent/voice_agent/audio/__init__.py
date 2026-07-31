"""Audio primitives: G.711 codec, resampling, VAD endpointing.

All in-pipeline audio is mono int16 PCM. Telephony is 8 kHz narrowband
(G.711 a-law/u-law); models (ASR/TTS) work at 16 kHz. Resample at the edges.
"""

SAMPLE_RATE_TELEPHONY = 8000
SAMPLE_RATE_MODEL = 16000
FRAME_MS = 20  # RTP packetization
SAMPLES_PER_FRAME_8K = SAMPLE_RATE_TELEPHONY * FRAME_MS // 1000   # 160
SAMPLES_PER_FRAME_16K = SAMPLE_RATE_MODEL * FRAME_MS // 1000      # 320

# RTP static payload types (RFC 3551)
RTP_PT_PCMU = 0   # G.711 u-law
RTP_PT_PCMA = 8   # G.711 a-law
