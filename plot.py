"""plot.py – Visualise SNR vs packet-loss rate and save as PNG."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in headless environments
import matplotlib.pyplot as plt


def plot_snr_vs_packet_loss(
    packet_loss_values: list[float],
    snr_values: list[float],
    output_png: str = "plot.png",
    codec: str = "",
    bitrate: int = 0,
) -> None:
    """Generate and save an SNR-vs-packet-loss line plot.

    Args:
        packet_loss_values: List of packet-loss fractions (x-axis).
        snr_values:         Corresponding SNR values in dB (y-axis).
        output_png:         Destination file path.
        codec:              Codec name (used in title/legend).
        bitrate:            Bitrate in bps (used in title).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Replace infinities for nicer plotting
    finite_snr = [
        (v if v != float("inf") and v != float("-inf") else None)
        for v in snr_values
    ]

    label = f"{codec} @ {bitrate // 1000} kbps" if codec else "SNR"
    ax.plot(
        packet_loss_values,
        finite_snr,
        marker="o",
        linewidth=2,
        label=label,
    )

    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel("SNR (dB)")
    title = "Audio Quality vs Packet Loss"
    if codec:
        title += f"  [{codec}"
        if bitrate:
            title += f", {bitrate // 1000} kbps"
        title += "]"
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=120)
    plt.close(fig)
