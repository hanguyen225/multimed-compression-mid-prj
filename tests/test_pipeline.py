"""tests/test_pipeline.py – Unit and integration tests for the pipeline."""

import math
import os
import struct
import tempfile
import wave
import sys
import unittest

# Make sure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encode import encode, SUPPORTED_CODECS
from simulate import simulate_packet_loss
from decode import decode
from metrics import compute_snr, compute_latency
from plot import plot_snr_vs_packet_loss


# ---------------------------------------------------------------------------
# Helper: write a minimal synthetic 16-bit mono WAV
# ---------------------------------------------------------------------------

def _write_sine_wav(path: str, freq: float = 440.0, duration: float = 1.0,
                    sample_rate: int = 44100) -> None:
    """Write a mono 16-bit sine-wave WAV to *path*."""
    n_samples = int(sample_rate * duration)
    samples = [
        int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n_samples)
    ]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSimulate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make_bin(self, size: int = 10000) -> str:
        path = os.path.join(self.tmp, "data.bin")
        with open(path, "wb") as f:
            f.write(os.urandom(size))
        return path

    def test_no_loss_keeps_all_data(self):
        src = self._make_bin()
        dst = os.path.join(self.tmp, "out.bin")
        stats = simulate_packet_loss(src, dst, packet_loss=0.0, seed=0)
        self.assertEqual(stats["dropped"], 0)
        with open(src, "rb") as f_src, open(dst, "rb") as f_dst:
            self.assertEqual(f_src.read(), f_dst.read())

    def test_full_loss_drops_everything(self):
        src = self._make_bin()
        dst = os.path.join(self.tmp, "out.bin")
        stats = simulate_packet_loss(src, dst, packet_loss=1.0, seed=0)
        self.assertEqual(stats["kept"], 0)
        self.assertEqual(os.path.getsize(dst), 0)

    def test_partial_loss(self):
        src = self._make_bin(14000)
        dst = os.path.join(self.tmp, "out.bin")
        stats = simulate_packet_loss(src, dst, packet_loss=0.5, seed=42)
        self.assertGreater(stats["dropped"], 0)
        self.assertGreater(stats["kept"], 0)
        self.assertEqual(stats["total"], stats["dropped"] + stats["kept"])

    def test_invalid_packet_loss_raises(self):
        src = self._make_bin()
        with self.assertRaises(ValueError):
            simulate_packet_loss(src, "/tmp/x", packet_loss=1.5)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            simulate_packet_loss("/nonexistent/file.bin", "/tmp/x", packet_loss=0.1)


class TestSimulateAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make_wav(self, name: str = "test.wav", duration: float = 1.0) -> str:
        path = os.path.join(self.tmp, name)
        _write_sine_wav(path, duration=duration)
        return path

    def test_no_loss_identical_output(self):
        src = self._make_wav()
        dst = os.path.join(self.tmp, "out.wav")
        from simulate import simulate_packet_loss_audio
        stats = simulate_packet_loss_audio(src, dst, packet_loss=0.0, seed=0)
        self.assertEqual(stats["dropped"], 0)
        with wave.open(src, "rb") as w:
            orig_raw = w.readframes(w.getnframes())
        with wave.open(dst, "rb") as w:
            out_raw = w.readframes(w.getnframes())
        self.assertEqual(orig_raw, out_raw)

    def test_full_loss_produces_silence(self):
        src = self._make_wav()
        dst = os.path.join(self.tmp, "out.wav")
        from simulate import simulate_packet_loss_audio
        stats = simulate_packet_loss_audio(src, dst, packet_loss=1.0, seed=0)
        self.assertEqual(stats["kept"], 0)
        with wave.open(dst, "rb") as w:
            raw = w.readframes(w.getnframes())
        self.assertEqual(set(raw), {0})

    def test_partial_loss_produces_valid_wav(self):
        src = self._make_wav()
        dst = os.path.join(self.tmp, "out.wav")
        from simulate import simulate_packet_loss_audio
        stats = simulate_packet_loss_audio(src, dst, packet_loss=0.5, seed=7)
        self.assertGreater(stats["dropped"], 0)
        self.assertGreater(stats["kept"], 0)
        with wave.open(dst, "rb") as w:
            self.assertGreater(w.getnframes(), 0)

    def test_snr_decreases_with_more_loss(self):
        src = self._make_wav()
        from simulate import simulate_packet_loss_audio
        snrs = []
        for pl in [0.0, 0.3, 0.7]:
            dst = os.path.join(self.tmp, f"out_{pl}.wav")
            simulate_packet_loss_audio(src, dst, packet_loss=pl, seed=42)
            snr = compute_snr(src, dst)
            snrs.append(snr if snr is not None else float("-inf"))
        # More loss → lower (or equal) SNR
        self.assertGreaterEqual(snrs[0], snrs[1])
        self.assertGreaterEqual(snrs[1], snrs[2])


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make_wav(self, name: str = "test.wav") -> str:
        path = os.path.join(self.tmp, name)
        _write_sine_wav(path)
        return path

    def test_snr_identical_files_is_inf(self):
        wav = self._make_wav()
        snr = compute_snr(wav, wav)
        self.assertEqual(snr, float("inf"))

    def test_snr_different_files_is_finite(self):
        orig = self._make_wav("orig.wav")
        other = self._make_wav("other.wav")  # same content
        # Corrupt other by writing silence
        n_samples = 44100
        with wave.open(other, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x00" * n_samples * 2)
        snr = compute_snr(orig, other)
        # SNR should be finite (signal is non-zero, noise is non-zero)
        self.assertNotEqual(snr, None)
        self.assertTrue(math.isfinite(snr))

    def test_snr_missing_file_returns_none(self):
        wav = self._make_wav()
        snr = compute_snr(wav, "/nonexistent/file.wav")
        self.assertIsNone(snr)

    def test_latency_calculation(self):
        latency = compute_latency(0.1, 50.0)  # 100 ms + 50 ms = 150 ms
        self.assertAlmostEqual(latency, 150.0, places=5)


class TestPlot(unittest.TestCase):
    def test_plot_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "plot.png")
            plot_snr_vs_packet_loss(
                [0.0, 0.1, 0.2],
                [30.0, 20.0, 10.0],
                output_png=out,
                codec="opus",
                bitrate=64000,
            )
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 0)


class TestEncodeAndDecode(unittest.TestCase):
    """Integration tests that require FFmpeg and libopus."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wav = os.path.join(self.tmp, "input.wav")
        _write_sine_wav(self.wav, duration=0.5)

    def test_encode_opus(self):
        out = os.path.join(self.tmp, "encoded.ogg")
        elapsed = encode(self.wav, out, "opus", 64000)
        self.assertTrue(os.path.isfile(out))
        self.assertGreater(os.path.getsize(out), 0)
        self.assertGreater(elapsed, 0.0)

    def test_encode_aac(self):
        out = os.path.join(self.tmp, "encoded.aac")
        elapsed = encode(self.wav, out, "aac", 64000)
        self.assertTrue(os.path.isfile(out))
        self.assertGreater(os.path.getsize(out), 0)

    def test_unsupported_codec_raises(self):
        with self.assertRaises(ValueError):
            encode(self.wav, "/tmp/x.mp3", "mp3_unknown", 64000)

    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            encode("/nonexistent/file.wav", "/tmp/x.ogg", "opus", 64000)

    def test_encode_decode_roundtrip_opus(self):
        enc = os.path.join(self.tmp, "enc.ogg")
        dec = os.path.join(self.tmp, "dec.wav")
        encode(self.wav, enc, "opus", 64000)
        decode(enc, dec)
        self.assertTrue(os.path.isfile(dec))
        snr = compute_snr(self.wav, dec)
        # After codec round-trip SNR should be positive (reasonable quality)
        self.assertIsNotNone(snr)
        self.assertGreater(snr, 0.0)

    def test_full_pipeline_opus(self):
        """encode → simulate (no loss) → decode → SNR should be positive."""
        enc = os.path.join(self.tmp, "enc.ogg")
        degraded = os.path.join(self.tmp, "degraded.ogg")
        dec = os.path.join(self.tmp, "dec.wav")

        encode(self.wav, enc, "opus", 64000)
        simulate_packet_loss(enc, degraded, packet_loss=0.0, seed=0)
        decode(degraded, dec)
        snr = compute_snr(self.wav, dec)
        self.assertIsNotNone(snr)
        self.assertGreater(snr, 0.0)


if __name__ == "__main__":
    unittest.main()
