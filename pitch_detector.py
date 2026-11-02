import librosa
import numpy as np

SAMPLE_RATE = 22050
HOP_LENGTH = 256
FRAME_LENGTH = 2048


def analyze_recording(audio_file, target_note="A4", tolerance=10):
    """Analyze a short recording once and return reusable results."""
    if hasattr(audio_file, "seek"):
        audio_file.seek(0)

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True,
        duration=60.1,
    )

    if audio.size == 0:
        raise ValueError("The recording is empty.")

    if len(audio) / sr > 60:
        raise ValueError("Please use a recording under 60 seconds.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The recording contains invalid samples.")

    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError("The recording is silent.")

    pitches, voiced, confidence = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("A7"),
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        resolution=0.1,
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]

    times = librosa.times_like(
        pitches,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    duration = len(audio) / sr

    # Exclude padded frames beyond the actual recording.
    inside = times < duration
    pitches = pitches[inside]
    voiced = voiced[inside]
    confidence = confidence[inside]
    rms = rms[inside]
    times = times[inside]

    energy_threshold = max(1e-4, float(rms.max()) * 0.05)
    audible = rms >= energy_threshold

    valid = (
        voiced
        & np.isfinite(pitches)
        & (confidence >= 0.5)
        & audible
    )

    filtered = np.where(valid, pitches, np.nan)

    # Frame durations make coverage duration-weighted.
    frame_durations = np.minimum(
        HOP_LENGTH / sr,
        duration - times,
    )

    accepted_seconds = float(frame_durations[valid].sum())
    audible_seconds = float(frame_durations[audible].sum())

    target_hz = float(librosa.note_to_hz(target_note))
    cents = np.full(filtered.shape, np.nan)

    cents[valid] = 1200 * np.log2(
        filtered[valid] / target_hz
    )

    segments = _group_notes(times, filtered, duration, tolerance)

    summary = {
        "duration": duration,
        "coverage": 100 * accepted_seconds / duration,
        "audible_coverage": (
            100 * accepted_seconds / audible_seconds
            if audible_seconds > 0
            else 0.0
        ),
        "accepted_seconds": accepted_seconds,
        "frequency": None,
        "note": None,
        "bias": None,
        "median_error": None,
        "spread": None,
        "in_tune": None,
    }

    if valid.any():
        accepted_cents = cents[valid]
        frequency = float(np.median(filtered[valid]))

        summary.update({
            "frequency": frequency,
            "note": librosa.hz_to_note(frequency),
            "bias": float(np.median(accepted_cents)),
            "median_error": float(
                np.median(np.abs(accepted_cents))
            ),
            # Robust spread: central 80% of accepted pitches.
            "spread": float(
                np.percentile(accepted_cents, 90)
                - np.percentile(accepted_cents, 10)
            ),
            "in_tune": float(
                100
                * frame_durations[
                    valid & (np.abs(cents) <= tolerance)
                ].sum()
                / accepted_seconds
            ),
        })

    return {
        "target": target_note,
        "target_hz": target_hz,
        "tolerance": tolerance,
        "summary": summary,
        "segments": segments,
        "frames": {
            "Time (s)": times,
            "Frequency (Hz)": filtered,
            "Deviation from target (cents)": cents,
            "Accepted": valid,
            "Voicing probability": confidence,
            "RMS": rms,
        },
    }


def _group_notes(times, pitches, duration, tolerance):
    """Group consecutive equal note labels, preserving silent gaps."""
    labels = [
        librosa.hz_to_note(float(f)) if np.isfinite(f) else None
        for f in pitches
    ]

    segments = []
    start = 0

    for end in range(1, len(labels) + 1):
        if end < len(labels) and labels[end] == labels[start]:
            continue

        label = labels[start]
        end_time = (
            float(times[end])
            if end < len(times)
            else duration
        )

        segment_duration = end_time - float(times[start])

        if label is not None and segment_duration >= 0.12:
            values = pitches[start:end]
            frequency = float(np.median(values))
            reference = float(librosa.note_to_hz(label))
            deviations = 1200 * np.log2(values / reference)
            deviation = float(np.median(deviations))

            if abs(deviation) <= tolerance:
                status = "In tune"
            elif deviation > 0:
                status = "Sharp"
            else:
                status = "Flat"

            segments.append({
                "Start (s)": round(float(times[start]), 2),
                "Duration (s)": round(segment_duration, 2),
                "Note": label,
                "Frequency (Hz)": round(frequency, 2),
                "Deviation (cents)": round(deviation, 1),
                "Spread (cents)": round(
                    float(
                        np.percentile(deviations, 90)
                        - np.percentile(deviations, 10)
                    ),
                    1,
                ),
                "Status": status,
            })

        start = end

    return segments