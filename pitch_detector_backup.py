import librosa
import numpy as np

SAMPLE_RATE = 22050
HOP_LENGTH = 256
FRAME_LENGTH = 2048


def _get_pitch_track(audio_file):
    if hasattr(audio_file, "seek"):
        audio_file.seek(0)

    audio, sample_rate = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("The recording is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The recording contains invalid audio data.")

    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError("The recording is silent.")

    # This range includes lower humming notes and violin notes.
    pitches, voiced, probabilities = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("A7"),
        sr=sample_rate,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        resolution=0.1,
    )

    loudness = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]

    times = librosa.times_like(
        pitches,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    threshold = max(
        1e-4,
        float(np.max(loudness)) * 0.05,
    )

    valid = (
        voiced
        & np.isfinite(pitches)
        & (probabilities >= 0.5)
        & (loudness >= threshold)
    )

    pitches = np.where(valid, pitches, np.nan)

    return times, pitches


def detect_pitch(audio_file):
    _, pitches = _get_pitch_track(audio_file)

    valid_pitches = pitches[np.isfinite(pitches)]

    if valid_pitches.size == 0:
        raise ValueError(
            "No clear pitch detected. Try a steady note "
            "in a quiet room."
        )

    frequency = float(np.median(valid_pitches))
    note = librosa.hz_to_note(frequency)

    return frequency, note


def analyze_intonation(frequency, tolerance=10):
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(
            "Frequency must be a positive, finite number."
        )

    midi_note = float(librosa.hz_to_midi(frequency))
    nearest_note = round(midi_note)

    cents = (midi_note - nearest_note) * 100

    if abs(cents) <= tolerance:
        status = "In tune"
    elif cents > 0:
        status = "Sharp"
    else:
        status = "Flat"

    return cents, status


def detect_multiple_pitches(audio_file):
    times, pitches = _get_pitch_track(audio_file)

    detected_notes = []

    for time, frequency in zip(times, pitches):
        if np.isfinite(frequency):
            note = librosa.hz_to_note(float(frequency))
        else:
            note = None

        detected_notes.append(
            (
                float(time),
                float(frequency),
                note,
            )
        )

    return detected_notes


def group_notes(detected_notes, minimum_duration=0.12):
    grouped_notes = []

    current_note = None
    start_time = None
    frequencies = []

    frame_duration = HOP_LENGTH / SAMPLE_RATE

    def save_note(end_time):
        if current_note is None or not frequencies:
            return

        duration = end_time - start_time

        if duration < minimum_duration:
            return

        frequency = float(np.median(frequencies))
        cents, status = analyze_intonation(frequency)

        grouped_notes.append(
            {
                "Start (s)": round(start_time, 2),
                "Duration (s)": round(duration, 2),
                "Note": current_note,
                "Frequency (Hz)": round(frequency, 2),
                "Deviation (cents)": round(cents, 1),
                "Status": status,
            }
        )

    for time, frequency, note in detected_notes:
        if note != current_note:
            save_note(time)

            current_note = note
            start_time = time
            frequencies = []

        if note is not None and np.isfinite(frequency):
            frequencies.append(frequency)

    if detected_notes:
        save_note(
            detected_notes[-1][0] + frame_duration
        )

    return grouped_notes