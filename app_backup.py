import pandas as pd
import streamlit as st

from pitch_detector import (
    detect_pitch,
    analyze_intonation,
    detect_multiple_pitches,
    group_notes,
)

st.set_page_config(
    page_title="Intona | Violin Coach",
    page_icon="🎻",
    layout="wide",
)

st.title("🎻 Intona")
st.subheader("Your Personal Violin Practice Coach")

st.write(
    "Upload a solo violin recording to explore its notes "
    "and see whether they tend to be sharp or flat."
)

st.caption(
    "Use a short recording with one note at a time. "
    "Chords and accompaniment are not supported yet. "
    "Intonation is measured against equal temperament with A4 = 440 Hz."
)

audio_source = st.radio(
    "Choose your audio source",
    ["Upload a file", "Record with microphone"],
    horizontal=True,
)

if audio_source == "Upload a file":
    uploaded_file = st.file_uploader(
        "Upload your violin recording",
        type=["wav", "mp3", "m4a"],
    )
else:
    uploaded_file = st.audio_input(
        "Record your violin",
        sample_rate=44100,
    )

if uploaded_file is not None:
    st.audio(uploaded_file)

    mode = st.radio(
        "What would you like to analyze?",
        ["One sustained note", "A sequence of notes"],
        horizontal=True,
    )

    if st.button("Analyze recording", type="primary"):
        with st.spinner("Listening to your violin..."):
            try:
                uploaded_file.seek(0)

                if mode == "One sustained note":
                    frequency, note = detect_pitch(uploaded_file)
                    cents, status = analyze_intonation(frequency)

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Detected note", note)
                    col2.metric("Frequency", f"{frequency:.2f} Hz")
                    col3.metric("Deviation", f"{cents:+.1f} cents")

                    if status == "In tune":
                        st.success(
                            "🎯 Within 10 cents of the nearest note."
                        )
                    elif status == "Sharp":
                        st.warning(
                            f"⬆️ About {abs(cents):.1f} cents sharp."
                        )
                    else:
                        st.warning(
                            f"⬇️ About {abs(cents):.1f} cents flat."
                        )

                    st.caption(
                        "This result summarizes the recording. "
                        "Use sequence mode if you played different notes."
                    )

                else:
                    notes = detect_multiple_pitches(uploaded_file)
                    grouped = group_notes(notes)

                    if not grouped:
                        st.warning(
                            "No stable notes found. Try a short, clear "
                            "recording with each note held for about "
                            "half a second or longer."
                        )
                    else:
                        results = pd.DataFrame(grouped)

                        col1, col2, col3 = st.columns(3)

                        col1.metric("Note segments", len(results))

                        in_tune = (
                            results["Status"] == "In tune"
                        ).sum()

                        col2.metric(
                            "Segments within ±10 cents",
                            f"{100 * in_tune / len(results):.0f}%",
                        )

                        col3.metric(
                            "Median absolute deviation",
                            f"{results['Deviation (cents)'].abs().median():.1f} cents",
                        )

                        st.subheader("🎵 Detected notes")

                        st.dataframe(
                            results,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.subheader("🎯 Practice feedback")

                        sharp_notes = results.loc[
                            results["Status"] == "Sharp",
                            "Note",
                        ].unique().tolist()

                        flat_notes = results.loc[
                            results["Status"] == "Flat",
                            "Note",
                        ].unique().tolist()

                        if not sharp_notes and not flat_notes:
                            st.success(
                                "All detected segments were within "
                                "±10 cents of their nearest reference note."
                            )
                        else:
                            if sharp_notes:
                                st.warning(
                                    "Sharp segments detected on: "
                                    + ", ".join(sharp_notes)
                                    + ". Practice these slowly, listening "
                                    "for a slightly lower pitch."
                                )

                            if flat_notes:
                                st.warning(
                                    "Flat segments detected on: "
                                    + ", ".join(flat_notes)
                                    + ". Practice these slowly, listening "
                                    "for a slightly higher pitch."
                                )

                            st.info(
                                "Practice tip: choose one flagged note. "
                                "Listen to a reference tone, hold the note "
                                "steadily without vibrato, then record again."
                            )

                        st.subheader("Pitch over time")

                        pitch_data = pd.DataFrame(
                            [
                                {
                                    "Time (s)": time,
                                    "Frequency (Hz)": frequency,
                                }
                                for time, frequency, note in notes
                            ]
                        ).set_index("Time (s)")

                        st.line_chart(pitch_data)

                        st.download_button(
                            "Download note analysis",
                            data=results.to_csv(index=False),
                            file_name="intona_analysis.csv",
                            mime="text/csv",
                        )

                        st.caption(
                            "Note segments are estimates. Vibrato, bow "
                            "changes, and repeated notes at the same "
                            "pitch can affect segmentation."
                        )

            except Exception as error:
                st.error(f"Could not analyze this recording: {error}")
                st.info(
                    "Try a short WAV recording first. "
                    "MP3 and M4A support depends on your audio decoder."
                )

else:
    st.info("Upload a recording to begin.")