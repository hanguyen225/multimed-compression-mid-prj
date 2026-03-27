"""decode.py – FFmpeg wrapper for decoding a compressed stream back to WAV."""

import subprocess
from pathlib import Path


def decode(input_path: str, output_wav: str) -> None:
    """Decode *input_path* (compressed audio) to *output_wav* (PCM WAV).

    Raises:
        FileNotFoundError: if *input_path* does not exist.
        RuntimeError: if FFmpeg exits with a non-zero return code.
    """
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"Compressed file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "44100",
        "-ac", "1",
        "-sample_fmt", "s16",
        output_wav,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg decoding failed (exit {result.returncode}):\n"
            + result.stderr.decode(errors="replace")
        )
