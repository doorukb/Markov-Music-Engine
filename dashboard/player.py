"""In-browser audio playback for generated compositions (MIDI → WAV → st.audio)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from markov.generator import CompositionResult
from markov.renderer import render_midi, render_wav

__all__ = ["PreparedAudio", "prepare_audio", "audio_widget"]

# MIDI path always set; wav_path set when FluidSynth synthesis succeeds.
@dataclass(frozen=True)
class PreparedAudio:
    midi_path: Path
    wav_path: Path | None

# render MIDI, then synthesize WAV via FluidSynth.
def prepare_audio(composition_result: CompositionResult, output_stem: str) -> PreparedAudio:
    midi_path = render_midi(composition_result, f"{output_stem}.mid")
    try:
        wav_path = render_wav(midi_path, f"{output_stem}.wav")
    except RuntimeError:
        return PreparedAudio(midi_path=midi_path, wav_path=None)
    return PreparedAudio(midi_path=midi_path, wav_path=wav_path)

# play WAV in-browser, or offer a MIDI download when synthesis is unavailable
def audio_widget(wav_path: Path | None, label: str, *, midi_path: Path | None = None) -> None:
    st.markdown(f"**{label}**")
    resolved_midi = midi_path
    if resolved_midi is None and wav_path is not None:
        # prepare_audio uses the same stem for .mid and .wav (see render_midi / render_wav).
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