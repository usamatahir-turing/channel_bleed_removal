"""
Manual channel bleed reduction using Gecko seglst annotations.

For each WAV in manual/Conversations/<task>/original/ that has a matching
<stem>.seglst.json, keeps audio only inside annotated [start_time, end_time]
segments and zeros everything else. WAVs without a seglst file are skipped.

Output: manual/Conversations/<task>/processed/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


DEFAULT_CONVERSATIONS_DIR = Path("manual/Conversations")
DEFAULT_FADE_MS = 10.0
SEGLST_SUFFIX = ".seglst.json"


def discover_conversations(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "original").is_dir()
    )


def parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def load_segments(seglst_path: Path) -> list[tuple[float, float]]:
    with seglst_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{seglst_path}: expected JSON array")

    segments: list[tuple[float, float]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{seglst_path}: item {index} is not an object")
        start = parse_time(item["start_time"])
        end = parse_time(item["end_time"])
        if end < start:
            raise ValueError(
                f"{seglst_path}: item {index} has end ({end:.3f}s) before start ({start:.3f}s)"
            )
        segments.append((start, end))

    if not segments:
        raise ValueError(f"{seglst_path}: no segments")

    segments.sort(key=lambda pair: pair[0])
    return segments


def load_mono_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


def build_mask(
    n_samples: int,
    sr: int,
    segments: list[tuple[float, float]],
    fade_ms: float,
) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=np.float64)
    fade_samples = max(1, int(fade_ms * sr / 1000.0))

    for start_sec, end_sec in segments:
        start = int(round(start_sec * sr))
        end = int(round(end_sec * sr))
        start = max(0, min(start, n_samples))
        end = max(0, min(end, n_samples))
        if end <= start:
            continue

        length = end - start
        rise = min(fade_samples, length // 2 or length)
        fall = min(fade_samples, length - rise)

        if rise:
            mask[start : start + rise] = np.maximum(
                mask[start : start + rise],
                np.linspace(0.0, 1.0, rise, endpoint=False),
            )
        if end - fall > start + rise:
            mask[start + rise : end - fall] = 1.0
        if fall:
            mask[end - fall : end] = np.maximum(
                mask[end - fall : end],
                np.linspace(1.0, 0.0, fall, endpoint=False),
            )

    return mask


def apply_mask(audio: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return (audio.astype(np.float64) * mask).astype(audio.dtype)

    peak = np.iinfo(audio.dtype).max
    scaled = audio.astype(np.float64) / peak
    return np.round(scaled * mask * peak).astype(audio.dtype)


def write_wav(path: Path, sr: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(audio.dtype, np.floating):
        clipped = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, sr, (clipped * 32767.0).astype(np.int16))
    else:
        wavfile.write(path, sr, audio)


def process_wav(wav_path: Path, seglst_path: Path, output_dir: Path, fade_ms: float) -> bool:
    segments = load_segments(seglst_path)
    sr, audio = load_mono_wav(wav_path)
    mask = build_mask(len(audio), sr, segments, fade_ms)
    processed = apply_mask(audio, mask)

    kept_seconds = float(mask.sum()) / sr
    total_seconds = len(audio) / sr
    kept_pct = 100.0 * kept_seconds / total_seconds if total_seconds else 0.0

    output_path = output_dir / wav_path.name
    write_wav(output_path, sr, processed)

    print(
        f"  {wav_path.name}: {len(segments)} segment(s), "
        f"kept {kept_seconds:.1f}s / {total_seconds:.1f}s ({kept_pct:.1f}%)",
        flush=True,
    )
    print(f"  wrote {output_path}", flush=True)
    return True


def process_conversation(task_dir: Path, fade_ms: float) -> int:
    input_dir = task_dir / "original"
    output_dir = task_dir / "processed"
    wav_paths = sorted(input_dir.glob("*.wav"))
    if not wav_paths:
        print(f"  no WAV files in {input_dir}", flush=True)
        return 0

    processed = 0
    for wav_path in wav_paths:
        seglst_path = input_dir / f"{wav_path.stem}{SEGLST_SUFFIX}"
        if not seglst_path.is_file():
            print(f"  skip {wav_path.name}: no {seglst_path.name}", flush=True)
            continue
        print(f"\nProcessing {wav_path.name}", flush=True)
        process_wav(wav_path, seglst_path, output_dir, fade_ms)
        processed += 1
    return processed


def process_conversations(conversations_root: Path, fade_ms: float) -> int:
    task_dirs = discover_conversations(conversations_root)
    if not task_dirs:
        raise FileNotFoundError(f"No conversation folders with original/ under {conversations_root}")

    print(f"Conversations root: {conversations_root}", flush=True)
    print(f"Fade:             {fade_ms:.1f} ms", flush=True)
    print(f"Found {len(task_dirs)} conversation(s)", flush=True)

    total = 0
    for task_dir in task_dirs:
        print(f"\n=== {task_dir.name} ===", flush=True)
        total += process_conversation(task_dir, fade_ms)

    print(f"\nDone. Processed {total} file(s) across {len(task_dirs)} conversation(s)", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate audio to Gecko seglst segments (zero outside annotations)"
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DEFAULT_CONVERSATIONS_DIR,
        help="Root folder containing manual conversation subfolders (default: manual/Conversations)",
    )
    parser.add_argument(
        "--fade-ms",
        type=float,
        default=DEFAULT_FADE_MS,
        help=f"Fade in/out at segment edges in ms (default: {DEFAULT_FADE_MS})",
    )
    args = parser.parse_args()

    try:
        count = process_conversations(args.conversations.resolve(), args.fade_ms)
        if count == 0:
            sys.exit(1)
    except (ValueError, OSError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
