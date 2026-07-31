#!/usr/bin/env python3
"""Raw PCM16 streaming for the Chatterbox pilot, matched to VoxCPM's /stream.

Emits little-endian 16-bit PCM mono so existing /stream consumers work unchanged
across a port-swap. Uses the model's native streaming if the installed build
exposes one (generate_stream / generate_streaming); otherwise falls back to full
synthesis then chunked emit (same wire format, but no early-TTFA benefit).
"""
import io
import inspect
import logging
import numpy as np

logger = logging.getLogger("chatterbox-pilot")


def _pcm16(samples) -> bytes:
    s = np.clip(np.asarray(samples, dtype="float32").reshape(-1), -1.0, 1.0)
    return (s * 32767.0).astype("<i2").tobytes()


def _to_mono_float(chunk):
    if hasattr(chunk, "detach"):
        chunk = chunk.detach().cpu().numpy()
    return np.asarray(chunk, dtype="float32").reshape(-1)


def _make_resampler(src: int, dst: int):
    if not dst or src == dst:
        return None
    try:
        import soxr
        return soxr.ResampleStream(src, dst, 1, dtype="float32")
    except Exception as e:
        logger.warning("no soxr (%s); streaming at native %dHz", e, src)
        return None


def _filter(fn, kw: dict) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def _native_stream(stream_fn, gen_kwargs, native_sr, out_rate):
    rs = _make_resampler(native_sr, out_rate)
    for chunk in stream_fn(**_filter(stream_fn, gen_kwargs)):
        s = _to_mono_float(chunk)
        if rs is not None:
            s = rs.resample_chunk(s)
        if len(s):
            yield _pcm16(s)
    if rs is not None:
        tail = rs.resample_chunk(np.zeros(0, dtype="float32"), last=True)
        if len(tail):
            yield _pcm16(tail)


def _fallback_stream(full_synth_fallback):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(full_synth_fallback()), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.reshape(-1)
    step = sr or 8000  # ~1s chunks
    for i in range(0, len(data), step):
        yield _pcm16(data[i:i + step])


def stream_pcm(model, gen_kwargs, native_sr, out_rate, full_synth_fallback):
    """Yield PCM16 chunks: real streaming if available, else full-synth fallback."""
    stream_fn = getattr(model, "generate_stream", None) or getattr(model, "generate_streaming", None)
    if stream_fn is not None:
        logger.info("streaming via model.%s", stream_fn.__name__)
        yield from _native_stream(stream_fn, gen_kwargs, native_sr, out_rate)
    else:
        logger.info("no native streaming; full-synth fallback (no early TTFA)")
        yield from _fallback_stream(full_synth_fallback)
