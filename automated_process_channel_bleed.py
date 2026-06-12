"""
Channel bleed reduction via RMS energy gating with a smooth gain ramp.

Processes every conversation under Conversations/<task>/:
  - reads original/*.wav
  - writes automated_processed/*.wav

Gain mapping (dB above per-file noise floor):
  below threshold_db       -> 0
  threshold_db .. high_db  -> smooth ramp 0 -> 1
  above high_threshold_db  -> 1

Usage:
  python automated_process_channel_bleed.py
  python automated_process_channel_bleed.py --conversations path/to/Conversations
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


FRAME_MS = 20.0
HOP_MS = 10.0
NOISE_PERCENTILE = 1.0
THRESHOLD_DB = 2.0
HIGH_THRESHOLD_DB = 20.0
GAIN_EXPONENT = 2.0
SMOOTH_MS = 30.0
DEFAULT_CONVERSATIONS_DIR = Path("Conversations")


def discover_conversations(root: Path) -> list[Path]:
    """Return conversation folders that contain an original/ directory."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "original").is_dir()
    )


def load_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        peak = np.iinfo(data.dtype).max
        audio = data.astype(np.float64) / peak
    else:
        audio = data.astype(np.float64)
    return sr, audio


def frame_rms(audio: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if len(audio) < frame_len:
        return np.array([np.sqrt(np.mean(audio * audio) + 1e-20)])

    n_frames = 1 + (len(audio) - frame_len) // hop_len
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        chunk = audio[i * hop_len : i * hop_len + frame_len]
        rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-20)
    return rms


def rms_to_db_above_floor(rms: np.ndarray, noise_floor: float) -> np.ndarray:
    floor = max(noise_floor, 1e-10)
    return 20.0 * np.log10(np.maximum(rms, 1e-10) / floor)


def compute_frame_gains(
    db_above_floor: np.ndarray,
    *,
    threshold_db: float,
    high_threshold_db: float,
    gain_exponent: float,
) -> np.ndarray:
    if high_threshold_db <= threshold_db:
        raise ValueError("high_threshold_db must be greater than threshold_db")

    width = high_threshold_db - threshold_db
    linear = np.clip((db_above_floor - threshold_db) / width, 0.0, 1.0)
    if gain_exponent != 1.0:
        linear = linear ** gain_exponent
    return linear


def smooth_envelope(gains: np.ndarray, sr: int, hop_ms: float, smooth_ms: float) -> np.ndarray:
    """Attack/release smoothing to reduce gain pumping between frames."""
    if len(gains) == 0 or smooth_ms <= 0:
        return gains

    hop_s = hop_ms / 1000.0
    tau = max(smooth_ms / 1000.0, hop_s)
    attack = float(np.exp(-hop_s / tau))
    release = float(np.exp(-hop_s / (tau * 1.5)))

    smoothed = np.empty_like(gains)
    state = 0.0
    for i, gain in enumerate(gains):
        coef = attack if gain > state else release
        state = coef * state + (1.0 - coef) * gain
        smoothed[i] = state
    return smoothed


def upsample_gains(
    frame_gains: np.ndarray,
    n_samples: int,
    sr: int,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    if len(frame_gains) == 0:
        return np.zeros(n_samples, dtype=np.float64)

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    centers = np.arange(len(frame_gains), dtype=np.float64) * hop_len + frame_len * 0.5
    samples = np.arange(n_samples, dtype=np.float64)
    return np.interp(samples, centers, frame_gains, left=0.0, right=0.0)


def regions_from_gains(
    frame_gains: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    min_gain: float,
) -> list[tuple[float, float]]:
    """Export intervals where gain stays above min_gain (for --write-segments)."""
    if len(frame_gains) == 0:
        return []

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    total_s = (len(frame_gains) - 1) * hop_len / sr + frame_len / sr
    active = frame_gains >= min_gain
    if not active.any():
        return []

    regions: list[tuple[float, float]] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        start_idx = i
        end_idx = i
        i += 1
        while i < n and active[i]:
            end_idx = i
            i += 1
        start_t = (start_idx * hop_len) / sr
        end_t = min((end_idx * hop_len + frame_len) / sr, total_s)
        if end_t > start_t:
            regions.append((start_t, end_t))
    return regions


def apply_gain_ramp(
    audio: np.ndarray,
    sr: int,
    *,
    frame_ms: float,
    hop_ms: float,
    noise_percentile: float,
    threshold_db: float,
    high_threshold_db: float,
    gain_exponent: float,
    smooth_ms: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    rms = frame_rms(audio, sr, frame_ms, hop_ms)
    noise_floor = float(np.percentile(rms, noise_percentile)) if len(rms) else 0.0
    noise_floor = max(noise_floor, 1e-10)

    db = rms_to_db_above_floor(rms, noise_floor)
    frame_gains = compute_frame_gains(
        db,
        threshold_db=threshold_db,
        high_threshold_db=high_threshold_db,
        gain_exponent=gain_exponent,
    )
    frame_gains = smooth_envelope(frame_gains, sr, hop_ms, smooth_ms)
    sample_gains = upsample_gains(frame_gains, len(audio), sr, frame_ms, hop_ms)
    return audio * sample_gains, sample_gains, noise_floor, frame_gains


def write_wav(path: Path, sr: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (clipped * 32767.0).astype(np.int16))


def write_segments_csv(path: Path, regions: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start", "end"])
        for start, end in regions:
            writer.writerow([f"{start:.3f}", f"{end:.3f}"])


def process_conversation(
    task_dir: Path,
    write_segments: bool,
    segment_min_gain: float,
    **kwargs: float,
) -> int:
    input_dir = task_dir / "original"
    output_dir = task_dir / "automated_processed"
    wav_paths = sorted(input_dir.glob("*.wav"))
    if not wav_paths:
        print(f"  no WAV files in {input_dir}", flush=True)
        return 0

    for wav_path in wav_paths:
        print(f"\nProcessing {wav_path.name}", flush=True)
        process_file(
            wav_path,
            output_dir,
            write_segments,
            segment_min_gain,
            **kwargs,
        )
    return len(wav_paths)


def process_conversations(
    conversations_root: Path,
    write_segments: bool,
    segment_min_gain: float,
    **kwargs: float,
) -> int:
    task_dirs = discover_conversations(conversations_root)
    if not task_dirs:
        raise FileNotFoundError(f"No conversation folders with original/ under {conversations_root}")

    print(f"Conversations root: {conversations_root}", flush=True)
    print(f"Found {len(task_dirs)} conversation(s)", flush=True)
    print(
        f"Ramp:   {kwargs['threshold_db']:.1f} dB -> 0, "
        f"{kwargs['high_threshold_db']:.1f} dB -> 1, "
        f"exponent={kwargs['gain_exponent']}, smooth={kwargs['smooth_ms']}ms",
        flush=True,
    )

    total = 0
    for task_dir in task_dirs:
        print(f"\n=== {task_dir.name} ===", flush=True)
        total += process_conversation(
            task_dir,
            write_segments,
            segment_min_gain,
            **kwargs,
        )

    print(f"\nDone. Processed {total} file(s) across {len(task_dirs)} conversation(s)", flush=True)
    return total


def process_file(
    wav_path: Path,
    output_dir: Path,
    write_segments: bool,
    segment_min_gain: float,
    **kwargs: float,
) -> None:
    sr, audio = load_mono(wav_path)
    gated, sample_gains, noise_floor, frame_gains = apply_gain_ramp(audio, sr, **kwargs)

    out_path = output_dir / wav_path.name
    write_wav(out_path, sr, gated)

    total = len(audio) / sr
    weighted = float(sample_gains.sum()) / sr
    full = float((sample_gains >= 0.99).sum()) / sr
    pct_weighted = 100.0 * weighted / total if total else 0.0
    pct_full = 100.0 * full / total if total else 0.0

    if write_segments:
        regions = regions_from_gains(
            frame_gains,
            sr,
            kwargs["frame_ms"],
            kwargs["hop_ms"],
            segment_min_gain,
        )
        write_segments_csv(output_dir / "detected_segments" / f"{wav_path.stem}.csv", regions)

    print(
        f"  {wav_path.name}: weighted {weighted:.1f}s ({pct_weighted:.1f}%), "
        f"full-gain {full:.1f}s ({pct_full:.1f}%) / {total:.1f}s, "
        f"noise={noise_floor:.2e}",
        flush=True,
    )
    print(f"  wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate audio with smooth RMS gain ramp (soft bleed attenuation)"
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DEFAULT_CONVERSATIONS_DIR,
        help="Root folder containing conversation subfolders (default: Conversations)",
    )
    parser.add_argument("--write-segments", action="store_true")
    parser.add_argument(
        "--segment-min-gain",
        type=float,
        default=0.5,
        help="Min gain for detected_segments CSV export (default: 0.5)",
    )
    parser.add_argument("--frame-ms", type=float, default=FRAME_MS)
    parser.add_argument("--hop-ms", type=float, default=HOP_MS)
    parser.add_argument("--noise-percentile", type=float, default=NOISE_PERCENTILE)
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=THRESHOLD_DB,
        help="dB above noise floor where gain reaches 0 (default: 15)",
    )
    parser.add_argument(
        "--high-threshold-db",
        type=float,
        default=HIGH_THRESHOLD_DB,
        help="dB above noise floor where gain reaches 1 (default: 24)",
    )
    parser.add_argument(
        "--gain-exponent",
        type=float,
        default=GAIN_EXPONENT,
        help="Exponent on ramp (>1 suppresses mid-level bleed more, default: 2)",
    )
    parser.add_argument(
        "--smooth-ms",
        type=float,
        default=SMOOTH_MS,
        help="Gain envelope smoothing time constant in ms (default: 30)",
    )
    args = parser.parse_args()

    if args.high_threshold_db <= args.threshold_db:
        print("Error: --high-threshold-db must be greater than --threshold-db", file=sys.stderr)
        sys.exit(1)

    ramp_kwargs = {
        "frame_ms": args.frame_ms,
        "hop_ms": args.hop_ms,
        "noise_percentile": args.noise_percentile,
        "threshold_db": args.threshold_db,
        "high_threshold_db": args.high_threshold_db,
        "gain_exponent": args.gain_exponent,
        "smooth_ms": args.smooth_ms,
    }

    try:
        count = process_conversations(
            args.conversations.resolve(),
            args.write_segments,
            args.segment_min_gain,
            **ramp_kwargs,
        )
        if count == 0:
            sys.exit(1)
    except (ValueError, OSError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
