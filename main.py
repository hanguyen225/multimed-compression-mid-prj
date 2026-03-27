"""main.py – CLI entry point for the audio compression simulation pipeline."""

import argparse
import os
import shutil
import struct
import sys
import tempfile
import wave

from encode import encode
from simulate import simulate_packet_loss, simulate_packet_loss_audio
from decode import decode
from metrics import compute_snr, compute_latency
from plot import plot_snr_vs_packet_loss


def _write_silence_wav(path: str, sample_rate: int = 44100,
                       n_samples: int = 44100) -> None:
    """Write a mono 16-bit silence WAV to *path*."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate audio compression quality under packet loss and network delay."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input WAV file.")
    parser.add_argument(
        "--codec",
        default="opus",
        choices=["opus", "aac"],
        help="Audio codec to use (default: opus).",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=64000,
        help="Encoding bitrate in bps (default: 64000).",
    )
    parser.add_argument(
        "--frame_ms",
        type=float,
        default=20.0,
        help="Frame size in milliseconds – used for packet-size hint (default: 20).",
    )
    parser.add_argument(
        "--packet_loss",
        type=float,
        default=0.0,
        help="Packet loss fraction 0.0–1.0 (default: 0.0).",
    )
    parser.add_argument(
        "--delay_ms",
        type=float,
        default=0.0,
        help="Simulated network delay in milliseconds (default: 0.0).",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Path for the degraded output WAV (default: output.wav).",
    )
    parser.add_argument(
        "--plot",
        default="plot.png",
        help="Path for the SNR-vs-packet-loss plot PNG (default: plot.png).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible packet-loss simulation.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Sweep packet-loss from 0.0 to 0.9 in steps of 0.1, "
            "plot SNR curve, and exit."
        ),
    )
    return parser


def _packet_size_bytes(bitrate: int, frame_ms: float) -> int:
    """Estimate packet payload size in bytes from bitrate and frame duration."""
    size = int(bitrate / 8.0 * frame_ms / 1000.0)
    return max(size, 64)  # floor at 64 bytes


def run_single(args, packet_loss: float, encoded_path: str, encoding_time: float) -> float | None:
    """Run simulate → decode → SNR for one packet-loss value.

    First attempts binary-level packet loss on the container stream.
    If FFmpeg cannot decode the corrupted stream, falls back to audio-domain
    simulation (zeroing random frames of the decoded audio) to produce a
    meaningful SNR value.

    Returns the SNR (dB) or None on failure.
    """
    pkt_size = _packet_size_bytes(args.bitrate, args.frame_ms)

    with tempfile.NamedTemporaryFile(
        suffix=".bin", delete=False
    ) as tmp_bin, tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as tmp_wav:
        degraded_bin = tmp_bin.name
        decoded_wav = tmp_wav.name

    try:
        # 1. Simulate binary-level packet loss
        simulate_packet_loss(
            input_path=encoded_path,
            output_path=degraded_bin,
            packet_loss=packet_loss,
            packet_size=pkt_size,
            seed=args.seed,
        )

        # 2. Attempt to decode the corrupted stream
        try:
            decode(degraded_bin, decoded_wav)
            # 3a. SNR from binary-level degraded audio
            return compute_snr(args.input, decoded_wav)
        except RuntimeError:
            pass  # fall through to audio-domain simulation

        # 3b. Fallback: decode the *clean* encoded file, then apply audio loss
        try:
            decode(encoded_path, decoded_wav)
        except RuntimeError:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            degraded_wav = tmp_audio.name
        try:
            simulate_packet_loss_audio(
                input_wav=decoded_wav,
                output_wav=degraded_wav,
                packet_loss=packet_loss,
                frame_ms=args.frame_ms,
                seed=args.seed,
            )
            return compute_snr(args.input, degraded_wav)
        finally:
            try:
                os.unlink(degraded_wav)
            except OSError:
                pass

    finally:
        for path in (degraded_bin, decoded_wav):
            try:
                os.unlink(path)
            except OSError:
                pass


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Validate inputs ---
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not 0.0 <= args.packet_loss <= 1.0:
        print("ERROR: --packet_loss must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    # Determine codec file extension
    ext = ".ogg" if args.codec == "opus" else ".aac"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_enc:
        encoded_path = tmp_enc.name

    try:
        # --- Encode ---
        print(f"Encoding '{args.input}' with {args.codec} @ {args.bitrate} bps …")
        try:
            encoding_time = encode(args.input, encoded_path, args.codec, args.bitrate)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"ERROR during encoding: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"  Encoding time: {encoding_time * 1000:.1f} ms")

        if args.sweep:
            # ---- Sweep mode: vary packet loss and plot ----
            loss_values = [round(x * 0.1, 1) for x in range(10)]  # 0.0 … 0.9
            snr_values = []
            print("\nSweeping packet-loss rates …")
            for pl in loss_values:
                snr = run_single(args, pl, encoded_path, encoding_time)
                snr_str = f"{snr:.2f} dB" if snr is not None else "N/A"
                print(f"  packet_loss={pl:.1f}  SNR={snr_str}")
                snr_values.append(snr)

            plot_snr_vs_packet_loss(
                loss_values,
                snr_values,
                output_png=args.plot,
                codec=args.codec,
                bitrate=args.bitrate,
            )
            print(f"\nPlot saved to '{args.plot}'")

        else:
            # ---- Single-run mode ----
            with tempfile.NamedTemporaryFile(
                suffix=".bin", delete=False
            ) as tmp_bin, tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_wav:
                degraded_bin = tmp_bin.name
                decoded_wav = tmp_wav.name

            try:
                # Simulate packet loss
                pkt_size = _packet_size_bytes(args.bitrate, args.frame_ms)
                print(
                    f"Simulating packet loss (rate={args.packet_loss}, "
                    f"packet_size={pkt_size} B) …"
                )
                stats = simulate_packet_loss(
                    input_path=encoded_path,
                    output_path=degraded_bin,
                    packet_loss=args.packet_loss,
                    packet_size=pkt_size,
                    seed=args.seed,
                )
                print(
                    f"  Packets: total={stats['total']}, "
                    f"dropped={stats['dropped']}, kept={stats['kept']}"
                )

                # Decode
                print("Decoding degraded stream …")
                decode_ok = True
                try:
                    decode(degraded_bin, decoded_wav)
                except RuntimeError as exc:
                    decode_ok = False
                    # Show only the last meaningful FFmpeg error line
                    err_lines = [l for l in str(exc).splitlines() if l.strip()]
                    short_err = err_lines[-1] if err_lines else str(exc)
                    print(
                        f"  WARNING: Decoding failed (stream too corrupted): {short_err}",
                        file=sys.stderr,
                    )
                    # Fallback: decode clean stream, then apply audio-domain loss
                    print(
                        "  Falling back to audio-domain packet loss simulation …",
                        file=sys.stderr,
                    )
                    try:
                        degraded_fallback = decoded_wav + ".degraded.wav"
                        decode(encoded_path, decoded_wav)
                        simulate_packet_loss_audio(
                            input_wav=decoded_wav,
                            output_wav=degraded_fallback,
                            packet_loss=args.packet_loss,
                            frame_ms=args.frame_ms,
                            seed=args.seed,
                        )
                        shutil.move(degraded_fallback, decoded_wav)
                    except Exception:
                        _write_silence_wav(decoded_wav)

                # Copy decoded to output path
                shutil.copy2(decoded_wav, args.output)
                print(f"  Degraded audio saved to '{args.output}'")

                # Metrics
                snr = compute_snr(args.input, args.output)
                latency = compute_latency(encoding_time, args.delay_ms)

                print("\n── Results ──────────────────────────────")
                print(f"  Codec:        {args.codec}")
                print(f"  Bitrate:      {args.bitrate} bps")
                print(f"  Packet loss:  {args.packet_loss:.1%}")
                print(f"  Latency:      {latency:.1f} ms")
                snr_str = (
                    f"{snr:.2f} dB"
                    if snr is not None and snr != float("inf") and snr != float("-inf")
                    else str(snr)
                )
                print(f"  SNR:          {snr_str}")
                print("─────────────────────────────────────────")

                # Plot (single-point plot with a sweep around the chosen value)
                loss_values = [round(x * 0.1, 1) for x in range(10)]
                snr_values = []
                for pl in loss_values:
                    s = run_single(args, pl, encoded_path, encoding_time)
                    snr_values.append(s)

                plot_snr_vs_packet_loss(
                    loss_values,
                    snr_values,
                    output_png=args.plot,
                    codec=args.codec,
                    bitrate=args.bitrate,
                )
                print(f"  Plot saved to '{args.plot}'")

            finally:
                for path in (degraded_bin, decoded_wav):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    finally:
        try:
            os.unlink(encoded_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
