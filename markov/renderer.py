"""
MIDI file writing and audio synthesis
- Convert a generated (chord, [notes]) sequence into a music21 Stream
- Write the Stream to a MIDI file in outputs/
- Call FluidSynth to synthesize MIDI to WAV using the configured soundfont
- Return the path to the WAV file for Streamlit's st.audio() widget
- Handle FluidSynth availability gracefully (fallback: MIDI-only mode)
"""