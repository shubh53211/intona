# 🎻 Intona — Violin Practice Coach

Intona is a Python and Streamlit prototype that analyzes
solo recordings and provides pitch and intonation feedback.

## Features

- Upload WAV, MP3, or M4A audio
- Record directly through your microphone
- Listen to reference tones
- Compare one sustained note with a selected target
- Detect sequential notes and display a pitch graph
- View sharp/flat practice feedback
- Download note analysis as CSV

## How it works

Intona uses librosa's probabilistic YIN (pYIN) algorithm
to estimate fundamental frequency over time. It filters
quiet or uncertain measurements and groups consecutive
pitch estimates into note segments.

Single-note mode compares the median accepted frequency
with the selected target, including its octave.
Sequence mode compares each retained segment with its
nearest equal-tempered note.

Tuning uses A4 = 440 Hz. The current pitch-bin resolution
is approximately 10 cents, with an in-tune tolerance of
±10 cents. Resolution does not guarantee measurement accuracy.

## Run locally

Use a Python environment compatible with the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

If your environment already exists, activate it and run
the app using the final two commands.

## Try it

1. Choose a reference note and listen to its tone.
2. Stop playback before recording.
3. Record one steady note or upload a short solo recording.
4. Choose single-note or sequence analysis.
5. Click "Analyze recording."

Start with WAV files. MP3 and M4A decoding depends on
the available audio decoder.

## Validation so far

- Tested note detection and sharp/flat labels using
  synthetic recordings with known pitches.
- Tested microphone capture and pitch detection with humming.
- Real violin accuracy has not yet been validated.

## Limitations

- Analyzes recordings after capture; this is not a live tuner.
- Supports one note at a time, not chords or accompaniment.
- Noise, vibrato, and bow changes can affect detection.
- Repeated notes at the same pitch may merge.
- A median pitch can misrepresent recordings with changing notes.
- Sequence percentages describe retained segments only;
  they do not measure melody correctness or all recorded audio.
- Practice advice uses rules based on detected pitch deviations.

## Technology

Python · Streamlit · librosa · NumPy · pandas · SoundFile