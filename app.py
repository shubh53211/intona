import hashlib
import io

import librosa
import numpy as np
import pandas as pd
import streamlit as st

from pitch_detector import analyze_recording


st.set_page_config(
    page_title="Intona | Practice Lab",
    page_icon="🎻",
    layout="wide",
)

st.title("🎻 Intona")
st.subheader("Listen. Practice. Measure your progress.")

st.write(
    "Analyze solo recordings, explore tuning and steadiness, "
    "and compare practice takes."
)

st.caption(
    "Prototype for one note at a time. No chords or accompaniment. "
    "Equal temperament, A4 = 440 Hz. Analysis runs after recording."
)

if "baseline" not in st.session_state:
    st.session_state.baseline = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# SETTINGS
with st.sidebar:
    st.header("Practice settings")

    target = st.selectbox(
        "Target note",
        [
            "G3", "A3", "B3", "C4", "C#4", "D4", "E4",
            "F4", "G4", "A4", "B4", "C#5", "D5", "E5",
        ],
        index=9,
    )

    mode = st.radio(
        "Analysis mode",
        ["One sustained note", "A sequence of notes"],
    )

    tolerance = st.select_slider(
        "In-tune tolerance",
        options=[10, 20, 30],
        value=10,
        format_func=lambda value: f"±{value} cents",
    )

    st.caption(
        "Pitch estimates use approximately 10-cent bins. "
        "This is resolution, not guaranteed accuracy."
    )

    st.caption(
        "Use recordings under 60 seconds. "
        "Real violin accuracy still needs broader validation."
    )


# REFERENCE TONE
target_hz = float(librosa.note_to_hz(target))

with st.expander(
    f"🎵 Hear {target} — {target_hz:.2f} Hz"
):
    sr = 44100
    t = np.arange(sr * 2) / sr
    tone = 0.25 * np.sin(2 * np.pi * target_hz * t)

    fade = int(0.02 * sr)
    envelope = np.ones(len(tone))
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)

    st.audio(
        (tone * envelope).astype(np.float32),
        sample_rate=sr,
    )

    st.caption("Stop reference playback before recording.")


# INPUT
source = st.radio(
    "Choose your audio source",
    ["Upload a file", "Record with microphone"],
    horizontal=True,
)

if source == "Upload a file":
    recording = st.file_uploader(
        "Upload solo audio",
        type=["wav", "mp3", "m4a"],
    )
else:
    recording = st.audio_input(
        "Record your practice take",
        sample_rate=44100,
    )

audio_bytes = (
    recording.getvalue()
    if recording is not None
    else None
)

signature = None

if audio_bytes is not None:
    st.audio(audio_bytes)

    signature = (
        hashlib.sha256(audio_bytes).hexdigest(),
        target,
        tolerance,
        mode,
    )

    if st.button("Analyze recording", type="primary"):
        # Clear old results before attempting new analysis.
        st.session_state.analysis = None

        with st.spinner("Measuring pitch and recording quality..."):
            try:
                result = analyze_recording(
                    io.BytesIO(audio_bytes),
                    target_note=target,
                    tolerance=tolerance,
                )

                st.session_state.analysis = {
                    "signature": signature,
                    "result": result,
                }

            except Exception as error:
                st.error(f"Could not analyze recording: {error}")
                st.info(
                    "Try a short WAV recording. "
                    "MP3 and M4A decoding depends on "
                    "your installed audio decoder."
                )


# DISPLAY SAVED RESULTS ONLY WHEN INPUT AND SETTINGS MATCH
saved = st.session_state.analysis

if saved is not None and saved["signature"] == signature:
    result = saved["result"]
    summary = result["summary"]
    frames = pd.DataFrame(result["frames"])

    st.subheader("Recording quality")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Recording length",
        f"{summary['duration']:.1f} s",
    )
    c2.metric(
        "Usable pitch / whole recording",
        f"{summary['coverage']:.0f}%",
    )
    c3.metric(
        "Usable pitch / audible audio",
        f"{summary['audible_coverage']:.0f}%",
    )

    st.caption(
        "Usable pitch means frames passing voicing, probability, "
        "and volume filters. Coverage is not an accuracy score. "
        "The audible-audio denominator uses a volume threshold."
    )

    if summary["frequency"] is None:
        st.warning(
            "No usable pitch estimates were found. "
            "Check microphone input and try a steady note."
        )

    else:
        if summary["coverage"] < 30:
            st.warning(
                "Less than 30% of the recording produced usable "
                "pitch estimates. Interpret the results cautiously."
            )

        if mode == "One sustained note":
            st.subheader(f"Tuning against {target}")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Median detected note",
                summary["note"],
            )
            c2.metric(
                "Median frequency",
                f"{summary['frequency']:.2f} Hz",
            )
            c3.metric(
                "Median target deviation",
                f"{summary['bias']:+.1f} cents",
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Accepted time within tolerance",
                f"{summary['in_tune']:.0f}%",
            )
            c2.metric(
                "Median absolute target error",
                f"{summary['median_error']:.1f} cents",
            )
            c3.metric(
                "Pitch spread",
                f"{summary['spread']:.1f} cents",
            )

            st.caption(
                "Pitch spread is the distance between the 10th "
                "and 90th percentiles of accepted pitches. "
                "Lower spread indicates greater steadiness "
                "for a held note without vibrato. "
                "Vibrato and intentional pitch changes increase it."
            )

            st.subheader("🎯 Practice feedback")

            bias = summary["bias"]

            if abs(bias) >= 100:
                direction = "above" if bias > 0 else "below"
                st.warning(
                    f"The median estimate is "
                    f"{abs(bias) / 100:.1f} semitones "
                    f"{direction} {target}. "
                    "Check the intended note and octave."
                )
            elif summary["median_error"] <= tolerance:
                st.success(
                    "The median absolute error is within "
                    "your selected tolerance."
                )
            elif abs(bias) <= tolerance:
                st.info(
                    "Your median pitch is near the target, "
                    "but individual measurements wander outside "
                    "the tolerance. Practice a steadier held note."
                )
            elif bias > 0:
                st.warning(
                    "Your median pitch is sharp. "
                    "Compare with the reference and aim lower."
                )
            else:
                st.warning(
                    "Your median pitch is flat. "
                    "Compare with the reference and aim higher."
                )

            if summary["spread"] > 40:
                st.info(
                    "The central pitch range exceeds 40 cents. "
                    "For a steadiness exercise, hold one note "
                    "without vibrato and use an even sound."
                )

            st.subheader("Target deviation over time")

            chart = frames.set_index("Time (s)")[
                ["Deviation from target (cents)"]
            ].copy()

            chart["Target"] = 0.0
            chart["Upper tolerance"] = float(tolerance)
            chart["Lower tolerance"] = float(-tolerance)

            st.line_chart(chart)

            st.caption(
                "Gaps represent rejected pitch estimates. "
                "A median can hide changing notes; use this "
                "mode only for a sustained target."
            )

            # SAME-TARGET PRACTICE COMPARISON
            st.subheader("📈 Compare practice takes")

            if st.button("Save this take as baseline"):
                st.session_state.baseline = {
                    "target": target,
                    "tolerance": tolerance,
                    "audio_hash": signature[0],
                    "summary": summary.copy(),
                }

            baseline = st.session_state.baseline

            if baseline is None:
                st.info(
                    "Save a baseline, record another take of "
                    "the same note, then analyze it."
                )

            elif (
                baseline["target"] != target
                or baseline["tolerance"] != tolerance
            ):
                st.info(
                    "Baseline settings differ. Return to its "
                    "target and tolerance, or save a new baseline."
                )

            elif baseline["audio_hash"] == signature[0]:
                st.info(
                    "Baseline saved. Record or upload a different "
                    "take to compare."
                )

            else:
                previous = baseline["summary"]

                comparison = pd.DataFrame([
                    {
                        "Measurement": "Median absolute error (cents)",
                        "Baseline": previous["median_error"],
                        "Current": summary["median_error"],
                    },
                    {
                        "Measurement": "Pitch spread (cents)",
                        "Baseline": previous["spread"],
                        "Current": summary["spread"],
                    },
                    {
                        "Measurement": "Accepted time in tune (%)",
                        "Baseline": previous["in_tune"],
                        "Current": summary["in_tune"],
                    },
                    {
                        "Measurement": "Whole-recording coverage (%)",
                        "Baseline": previous["coverage"],
                        "Current": summary["coverage"],
                    },
                ])

                st.dataframe(
                    comparison.round(1),
                    hide_index=True,
                    use_container_width=True,
                )

                improvement = (
                    previous["median_error"]
                    - summary["median_error"]
                )

                if improvement > 0:
                    st.success(
                        f"Median absolute target error decreased "
                        f"by {improvement:.1f} cents."
                    )
                elif improvement < 0:
                    st.info(
                        f"Median absolute target error increased "
                        f"by {abs(improvement):.1f} cents."
                    )
                else:
                    st.info(
                        "Median absolute target error is unchanged."
                    )

                st.caption(
                    "Compare similar recordings with similar "
                    "coverage. These are descriptive measurements, "
                    "not proof of skill improvement."
                )

            if baseline is not None:
                if st.button("Clear baseline"):
                    st.session_state.baseline = None
                    st.rerun()

            st.caption(
                "Baseline measurements last only for this "
                "browser session."
            )

        else:
            st.subheader("🎵 Note sequence")

            segments = pd.DataFrame(result["segments"])

            if segments.empty:
                st.warning(
                    "Pitch was detected, but no note segment "
                    "lasted at least 0.12 seconds."
                )

            else:
                st.dataframe(
                    segments,
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "Segment tuning is measured against each "
                    "segment's nearest note, not the selected target. "
                    "Short segments are discarded. Repeated notes "
                    "without a gap may merge."
                )

                flagged = segments[
                    segments["Status"] != "In tune"
                ]

                st.subheader("🎯 Practice priorities")

                if flagged.empty:
                    st.success(
                        "All retained segment medians are within "
                        "tolerance of their nearest notes."
                    )
                else:
                    priorities = flagged.copy()
                    priorities["Absolute error"] = (
                        priorities["Deviation (cents)"].abs()
                    )

                    priorities = priorities.sort_values(
                        "Absolute error",
                        ascending=False,
                    ).head(3)

                    for _, row in priorities.iterrows():
                        st.write(
                            f"• **{row['Note']}** at "
                            f"**{row['Start (s)']:.2f} s**: "
                            f"{row['Deviation (cents)']:+.1f} cents "
                            f"({row['Status'].lower()})."
                        )

                st.download_button(
                    "Download note segments",
                    data=segments.to_csv(index=False),
                    file_name="intona_segments.csv",
                    mime="text/csv",
                    on_click="ignore",
                )

            st.subheader("Pitch over time")

            st.line_chart(
                frames.set_index("Time (s)")[
                    ["Frequency (Hz)"]
                ]
            )

        with st.expander("Inspect frame measurements"):
            st.dataframe(
                frames,
                hide_index=True,
                use_container_width=True,
            )

        st.download_button(
            "Download frame measurements",
            data=frames.to_csv(index=False),
            file_name="intona_frames.csv",
            mime="text/csv",
            on_click="ignore",
        )

else:
    st.info(
        "Choose your settings, then record or upload "
        "audio and click Analyze recording."
    )