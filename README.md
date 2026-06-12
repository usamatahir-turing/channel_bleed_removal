# Channel Bleed

Post-processing for Riverside-style multitrack recordings to reduce **channel bleed** (one speaker's voice appearing on another speaker's track).

Two approaches:

| Script | Method |
|--------|--------|
| `automated_process_channel_bleed.py` | RMS energy gating with smooth gain ramp |
| `manually_process_channel_bleed.py` | Gecko `seglst.json` segment annotations |

## Problem

In multitrack conversation recordings, a speaker's isolated track sometimes contains low-level audio from other participants, typically from headphone leakage or room pickup. This tool attenuates or removes that bleed while preserving the intended speaker's voice.

## How it works

For each WAV file, the script:

1. Computes a short-term **RMS energy envelope** (20 ms frames, 10 ms hop).
2. Estimates a per-file **noise floor** from the quietest frames (15th percentile by default).
3. Converts frame energy to **dB above the noise floor**.
4. Maps dB to a **continuous gain** (smooth ramp):
   - below `threshold_db` → gain **0**
   - between `threshold_db` and `high_threshold_db` → ramp **0 → 1** (with optional exponent to suppress mid-level bleed)
   - above `high_threshold_db` → gain **1**
5. **Smooths** the gain envelope to avoid pumping artifacts.
6. Applies gain to the audio and writes the result.

Bleed is often in the middle energy range (above silence, below full speech). The smooth ramp attenuates those regions instead of using a hard on/off gate.

## Project layout

```
channel_bleed/
  automated_process_channel_bleed.py
  requirements.txt
  Conversations/
    NV-GR-SS03-CONVO07/
      original/              ← input WAVs (one per speaker)
        speaker@turing.com.wav
      automated_processed/   ← output WAVs (created by script)
    NV-AR-SS05-CONVO13/
      original/
      automated_processed/
    ...
```

Every subfolder under `Conversations/` that contains an `original/` directory is processed automatically. Output is written to `automated_processed/` inside the same conversation folder.

## Setup

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`.

## Usage

From the project root:

```bash
python automated_process_channel_bleed.py
```

Custom conversations root:

```bash
python automated_process_channel_bleed.py --conversations path/to/Conversations
```

Optional: export detected high-gain intervals as CSV for inspection (not run by default):

```bash
python automated_process_channel_bleed.py --write-segments
```

With that flag, CSV files are written to `automated_processed/detected_segments/` with `start` and `end` times in seconds. Regions are frames where gain ≥ `--segment-min-gain` (default 0.5).

## Output stats

For each file the script prints:

- **weighted** — effective duration after partial attenuation (gain-weighted)
- **full-gain** — duration at ≥ 99% gain (confident speech)

For a balanced 3-speaker conversation, full-gain time is often around **~30–35%** of the file per channel. If full-gain or weighted percentages are much higher, bleed may still be getting through — try tightening the parameters below.

## Parameters

Defaults are set at the top of `automated_process_channel_bleed.py` and can be overridden via CLI flags.

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold-db` | 30 | dB above noise floor where gain reaches 0 |
| `--high-threshold-db` | 40 | dB above noise floor where gain reaches 1 |
| `--gain-exponent` | 2 | Exponent on the ramp; >1 suppresses mid-level bleed more |
| `--noise-percentile` | 15 | Percentile of frame RMS used as noise floor |
| `--smooth-ms` | 30 | Gain envelope smoothing (ms) |
| `--frame-ms` | 20 | RMS analysis frame length (ms) |
| `--hop-ms` | 10 | RMS hop size (ms) |
| `--segment-min-gain` | 0.5 | Min gain for `--write-segments` CSV export |

### Tuning examples

More aggressive bleed reduction:

```bash
python automated_process_channel_bleed.py --threshold-db 32 --high-threshold-db 42 --gain-exponent 3
```

Narrower transition band:

```bash
python automated_process_channel_bleed.py --threshold-db 28 --high-threshold-db 38
```

## Limitations

- **Energy-based only** — the script uses loudness, not speaker identity. It cannot distinguish two voices at similar levels.
- **Overlap** — bleed during simultaneous speech may remain if combined energy stays in the full-gain zone.
- **Quiet speech** — aggressive thresholds can attenuate soft backchannels or trailing syllables.
- **Per-channel** — each track is processed independently; cross-channel logic is not used.

For heavy bleed where transcript accuracy is critical, listen to outputs and adjust thresholds per language/recording setup.

---

## Manual processing (seglst)

Use when segments have been annotated in Gecko. Keeps audio **only** inside annotated `[start_time, end_time]` ranges and zeros everything else.

### Layout

```
manual/Conversations/
  NV-EN-SS08-CONVO20/
    original/
      speaker@turing.com.wav
      speaker@turing.com.seglst.json
    processed/              ← output (created by script)
```

Each WAV must have a matching `<stem>.seglst.json` in the same `original/` folder. WAVs without a seglst file are skipped.

### Usage

```bash
python manually_process_channel_bleed.py
```

Custom root:

```bash
python manually_process_channel_bleed.py --conversations path/to/manual/Conversations
```

Optional fade at segment edges (default 10 ms):

```bash
python manually_process_channel_bleed.py --fade-ms 10
```
