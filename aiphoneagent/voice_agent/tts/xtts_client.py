"""XTTS v2 client — TecAI tts-server (OpenAI-compatible TTS).

Wraps the EXISTING XTTS v2 service (`/tec/ai/tts-server`, port 8087). The
contract is OpenAI's TTS shape:

    POST /v1/audio/speech
    Authorization: Bearer <token>
    {"model","input","voice","response_format","language","speed"}

We request `response_format=wav` (self-describing sample rate) and resample to
the rate the SIP bridge consumes. Everything downstream depends only on the
int16 PCM this returns, so the service contract stays isolated to this file.

NOTE: the server's languages do NOT include Greek ('el'); XTTS v2 supports
en/es/fr/de/it/pt/pl/tr/ru/nl/cs/ar/zh-cn/ja/hu/ko/hi. A Greek request returns
422 → `synthesize` raises and the orchestrator logs a TTS failure. Use a
Greek-capable TTS for 'el', or test in a supported language.
"""

from __future__ import annotations

import io
import logging
import wave

import httpx
import numpy as np

from ..audio.resample import resample_to

logger = logging.getLogger(__name__)


class XttsClient:
    def __init__(
        self,
        endpoint: str,
        voices: dict[str, str],
        default_voice: str,
        *,
        api_key: str = "",
        model: str = "tts-1",
        response_format: str = "wav",
        speed: float = 1.0,
        path: str = "/v1/audio/speech/public",   # no-auth route; "/v1/audio/speech" needs api_key
        timeout: float = 15.0,
    ) -> None:
        self._voices = voices
        self._default_voice = default_voice
        self._model = model
        self._response_format = response_format
        self._speed = speed
        # Build the URL explicitly so the endpoint's '/v1' prefix isn't dropped.
        self._url = endpoint.rstrip("/") + "/" + path.lstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # trust_env=False: ignore HTTP(S)_PROXY/ALL_PROXY — localhost service,
        # must be reached directly, never via a proxy.
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _voice(self, language: str) -> str:
        return self._voices.get(language.lower(), self._default_voice)

    async def _synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Return (int16 PCM, sample_rate) from the service."""
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice(language),
            "response_format": self._response_format,
            "language": language,
            "speed": self._speed,
        }
        resp = await self._client.post(self._url, json=payload)
        if resp.status_code >= 400:
            logger.error(
                "TTS %s (voice=%s lang=%s text=%r): %s",
                resp.status_code, payload["voice"], language, text[:60], resp.text[:300],
            )
            resp.raise_for_status()
        return self._parse(resp)

    async def synthesize_16k(self, text: str, language: str) -> np.ndarray:
        """Return int16 PCM at 16 kHz — the rate the SIP bridge consumes."""
        pcm, rate = await self._synthesize(text, language)
        return resample_to(pcm, rate, 16000)

    async def synthesize_8k(self, text: str, language: str) -> np.ndarray:
        """Return int16 PCM at 8 kHz (for a raw-RTP / LiveKit SIP path)."""
        pcm, rate = await self._synthesize(text, language)
        return resample_to(pcm, rate, 8000)

    def _parse(self, resp: httpx.Response) -> tuple[np.ndarray, int]:
        content_type = resp.headers.get("content-type", "")
        data = resp.content
        if "wav" in content_type or data[:4] == b"RIFF":
            with wave.open(io.BytesIO(data), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                pcm = np.frombuffer(frames, dtype=np.int16)
                return pcm, wf.getframerate()
        raise RuntimeError(
            f"Unexpected TTS content-type {content_type!r}; expected WAV "
            "(set XTTS_RESPONSE_FORMAT=wav)."
        )
