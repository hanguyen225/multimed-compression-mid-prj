"""simulate.py – Packet-loss simulation on a binary file or decoded audio."""

import os
import random
import struct
import wave
from pathlib import Path


# Default simulated packet size (bytes).  For real RTP/UDP this is typically
# 20–40 ms of audio per packet, but we keep it configurable.
DEFAULT_PACKET_SIZE = 1400  # bytes


def simulate_packet_loss(
    input_path: str,
    output_path: str,
    packet_loss: float,
    packet_size: int = DEFAULT_PACKET_SIZE,
    seed: int | None = None,
) -> dict:
    """Read *input_path*, split into chunks and randomly drop some of them.

    The surviving chunks are written concatenated to *output_path*.

    Args:
        input_path:   Path to the encoded audio file.
        output_path:  Path to write the degraded binary stream.
        packet_loss:  Fraction of packets to drop (0.0 – 1.0).
        packet_size:  Size of each simulated packet in bytes.
        seed:         Optional RNG seed for reproducibility.

    Returns:
        A dict with keys ``total``, ``dropped``, ``kept``.

    Raises:
        FileNotFoundError: if *input_path* does not exist.
        ValueError: if *packet_loss* is outside [0, 1].
    """
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"Encoded file not found: {input_path}")
    if not 0.0 <= packet_loss <= 1.0:
        raise ValueError(f"packet_loss must be in [0, 1], got {packet_loss}")

    rng = random.Random(seed)

    data = Path(input_path).read_bytes()

    # Split into chunks (last chunk may be smaller)
    chunks = [
        data[i : i + packet_size] for i in range(0, len(data), packet_size)
    ]

    kept_chunks = [c for c in chunks if rng.random() >= packet_loss]

    Path(output_path).write_bytes(b"".join(kept_chunks))

    return {
        "total": len(chunks),
        "dropped": len(chunks) - len(kept_chunks),
        "kept": len(kept_chunks),
    }


def simulate_packet_loss_audio(
    input_wav: str,
    output_wav: str,
    packet_loss: float,
    frame_ms: float = 20.0,
    sample_rate: int = 44100,
    seed: int | None = None,
) -> dict:
    """Simulate packet loss directly in the audio (PCM) domain.

    Randomly zeros out frame-sized segments of *input_wav* to mimic lost
    packets, then writes the result to *output_wav*.  This method always
    produces a valid WAV file and gives a smooth, meaningful SNR curve even
    when the container-level simulation yields an undecodable stream.

    Args:
        input_wav:    Path to the decoded (reference) WAV file.
        output_wav:   Path to write the degraded WAV.
        packet_loss:  Fraction of audio frames to silence (0.0 – 1.0).
        frame_ms:     Duration of one simulated packet in milliseconds.
        sample_rate:  Sample rate assumed for frame-size calculation.
        seed:         Optional RNG seed for reproducibility.

    Returns:
        A dict with keys ``total``, ``dropped``, ``kept``.

    Raises:
        FileNotFoundError: if *input_wav* does not exist.
        ValueError: if *packet_loss* is outside [0, 1].
    """
    if not Path(input_wav).is_file():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")
    if not 0.0 <= packet_loss <= 1.0:
        raise ValueError(f"packet_loss must be in [0, 1], got {packet_loss}")

    rng = random.Random(seed)

    with wave.open(input_wav, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Use actual sample rate from file, overriding the parameter
    sr = framerate
    frame_samples = max(1, int(sr * frame_ms / 1000.0))
    fmt_char = "b" if sampwidth == 1 else "h" if sampwidth == 2 else "i"

    total_samples = len(raw) // sampwidth // n_channels
    samples = list(struct.unpack(
        f"<{total_samples * n_channels}{fmt_char}",
        raw
    ))

    # Build frames (each frame = frame_samples * n_channels values)
    frame_len = frame_samples * n_channels
    frames = [
        samples[i : i + frame_len]
        for i in range(0, len(samples), frame_len)
    ]

    total = len(frames)
    kept = 0
    dropped = 0
    result_samples: list[int] = []

    for frame in frames:
        if rng.random() < packet_loss:
            # Replace with silence (packet concealment: zeros)
            result_samples.extend([0] * len(frame))
            dropped += 1
        else:
            result_samples.extend(frame)
            kept += 1

    # Pack back using same format character determined above
    degraded_raw = struct.pack(
        f"<{len(result_samples)}{fmt_char}", *result_samples
    )

    with wave.open(output_wav, "wb") as wf_out:
        wf_out.setnchannels(n_channels)
        wf_out.setsampwidth(sampwidth)
        wf_out.setframerate(framerate)
        wf_out.writeframes(degraded_raw)

    return {"total": total, "dropped": dropped, "kept": kept}
