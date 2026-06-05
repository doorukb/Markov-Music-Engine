from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from markov.generator import CompositionResult
from markov.renderer import render_midi, render_wav

__all__ = ["PreparedAudio", "prepare_audio", "audio_widget"]

# a dataclass to hold the paths produced by prepare_audio()
# paths produced by prepare_audio()
@dataclass(frozen=True)
class PreparedAudio:
    midi_path: Path
    wav_path: Path | None

    # WAV path when FluidSynth synthesis succeeded, else None
    @property
    def wav(self) -> Path | None:
        return self.wav_path

# render MIDI, then synthesize WAV via FluidSynth
# returns both paths, wav_path is None when FluidSynth is unavailable or synthesis fails
def prepare_audio(composition_result: CompositionResult, output_stem: str) -> PreparedAudio:
    midi_path = render_midi(composition_result, f"{output_stem}.mid")
    try:
        wav_path = render_wav(midi_path, f"{output_stem}.wav")
    except RuntimeError:
        return PreparedAudio(midi_path=midi_path, wav_path=None)
    return PreparedAudio(midi_path=midi_path, wav_path=wav_path)

# render a WAV audio player and MIDI download button
# wav_path: the path to the WAV file (may be None if FluidSynth is unavailable or synthesis fails)
# label: the heading shown above the player or download control
# midi_path: the path to the MIDI file (may be None if wav_path is provided)
def audio_widget(wav_path: Path | None, label: str, *, midi_path: Path | None = None) -> None:
    # render the heading
    st.markdown(f"**{label}**")
    resolved_midi = midi_path
    if resolved_midi is None and wav_path is not None:
        resolved_midi = wav_path.with_suffix(".mid")

    if wav_path is not None and wav_path.is_file():
        st.audio(wav_path.read_bytes(), format="audio/wav")
        return

    if resolved_midi is not None and resolved_midi.is_file():
        st.warning("FluidSynth is not available or audio synthesis failed. Download the MIDI file and play it in your own player.")
        st.download_button(
            label="Download MIDI",
            data=resolved_midi.read_bytes(),
            file_name=resolved_midi.name,
            mime="audio/midi",
            key=f"midi_download_{resolved_midi.name}_{label}",
        )
        return
    st.warning("No audio could be prepared for this composition.")