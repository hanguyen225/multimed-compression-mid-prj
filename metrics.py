"""metrics.py – Signal-to-Noise Ratio (SNR) and latency helpers."""

import wave
import struct
import math
from pathlib import Path


def _read_wav_samples(wav_path: str) -> list[float]:
    """Return a list of normalised float samples from a 16-bit mono WAV."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(
            f"Only 16-bit WAV is supported (got {sampwidth * 8}-bit): {wav_path}"
        )

    total_samples = len(raw) // sampwidth
    samples = list(struct.unpack(f"<{total_samples}h", raw))

    # If stereo, take only the first channel
    if n_channels == 2:
        samples = samples[::2]

    # Normalise to [-1, 1]
    return [s / 32768.0 for s in samples]


def compute_snr(original_wav: str, degraded_wav: str) -> float:
    """Compute the Signal-to-Noise Ratio (dB) between two WAV files.

    SNR = 10 * log10( mean(signal²) / mean(noise²) )
    where noise = original – degraded (truncated to the shorter length).

    Returns float('inf') when the noise power is zero (perfect match).
    Returns float('-inf') when the signal power is zero.
    Returns ``None`` if the degraded file cannot be read or is empty.
    """
    if not Path(degraded_wav).is_file():
        return None

    try:
        orig = _read_wav_samples(original_wav)
        deg = _read_wav_samples(degraded_wav)
    except Exception:
        return None

    if not orig or not deg:
        return None

    # Align lengths
    length = min(len(orig), len(deg))
    orig = orig[:length]
    deg = deg[:length]

    signal_power = sum(s ** 2 for s in orig) / length
    noise_power = sum((o - d) ** 2 for o, d in zip(orig, deg)) / length

    if signal_power == 0.0:
        return float("-inf")
    if noise_power == 0.0:
        return float("inf")

    return 10.0 * math.log10(signal_power / noise_power)


def compute_latency(encoding_time_s: float, delay_ms: float) -> float:
    """Return total latency (ms) = encoding time + simulated network delay."""
    return encoding_time_s * 1000.0 + delay_ms
