"""encode.py – FFmpeg wrapper for encoding WAV to a compressed format."""

import subprocess
import time
from pathlib import Path


SUPPORTED_CODECS = ("opus", "aac")


def encode(input_wav: str, output_path: str, codec: str, bitrate: int) -> float:
    """Encode *input_wav* to *output_path* using *codec* at *bitrate* bps.

    Returns the wall-clock encoding time in seconds.

    Raises:
        FileNotFoundError: if *input_wav* does not exist.
        ValueError: if *codec* is not supported.
        RuntimeError: if FFmpeg exits with a non-zero return code.
    """
    if not Path(input_wav).is_file():
        raise FileNotFoundError(f"Input file not found: {input_wav}")
    if codec not in SUPPORTED_CODECS:
        raise ValueError(
            f"Unsupported codec '{codec}'. Choose from: {SUPPORTED_CODECS}"
        )

    if codec == "opus":
        # libopus works with ogg container
        ffmpeg_codec = "libopus"
    else:
        # aac works with adts / m4a container
        ffmpeg_codec = "aac"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_wav,
        "-c:a", ffmpeg_codec,
        "-b:a", str(bitrate),
        output_path,
    ]

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg encoding failed (exit {result.returncode}):\n"
            + result.stderr.decode(errors="replace")
        )

    return elapsed
