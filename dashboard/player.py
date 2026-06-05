from __future__ import annotations
import base64
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from markov.audio_setup import check_audio_setup, ensure_soundfont
from markov.generator import CompositionResult
from markov.playback import export_original_midi, midi_file_duration_seconds
from markov.renderer import render_midi, render_wav

__all__ = [
    "PreparedAudio",
    "prepare_audio",
    "prepare_original_audio",
    "prepare_midi_wav",
    "audio_widget",
    "get_last_synthesis_error",
    "set_last_synthesis_error",
]

_LAST_ERROR_KEY = "_last_synthesis_error"

# MIDI path always set- wav_path set when FluidSynth synthesis succeeds
@dataclass(frozen=True)
class PreparedAudio:
    midi_path: Path
    wav_path: Path | None
    duration_seconds: float = 0.0


def set_last_synthesis_error(message: str | None) -> None:
    st.session_state[_LAST_ERROR_KEY] = message

def get_last_synthesis_error() -> str | None:
    return st.session_state.get(_LAST_ERROR_KEY)

# synthesize WAV for an existing MIDI file
def prepare_midi_wav(midi_path: Path, wav_stem: str) -> PreparedAudio:
    set_last_synthesis_error(None)
    try:
        ensure_soundfont()
        wav_path = render_wav(midi_path, f"{wav_stem}.wav")
        duration = midi_file_duration_seconds(wav_path)
        return PreparedAudio(midi_path=midi_path, wav_path=wav_path, duration_seconds=duration)
    except Exception as exc:
        set_last_synthesis_error(str(exc))
        duration = midi_file_duration_seconds(midi_path)
        return PreparedAudio(midi_path=midi_path, wav_path=None, duration_seconds=duration)

# render MIDI and synthesize WAV via FluidSynth
def prepare_audio(composition_result: CompositionResult, output_stem: str) -> PreparedAudio:
    midi_path = render_midi(composition_result, f"{output_stem}.mid")
    assets = prepare_midi_wav(midi_path, output_stem)
    if assets.duration_seconds <= 0:
        from markov.playback import composition_duration_seconds

        return PreparedAudio(
            midi_path=assets.midi_path,
            wav_path=assets.wav_path,
            duration_seconds=composition_duration_seconds(composition_result),
        )
    return assets

# export the training source piece to MIDI and synthesize WAV when possible
def prepare_original_audio(source_path: Path, output_stem: str) -> PreparedAudio:
    midi_path, duration = export_original_midi(source_path, output_stem)
    assets = prepare_midi_wav(midi_path, output_stem)
    if assets.duration_seconds <= 0:
        return PreparedAudio(midi_path=midi_path, wav_path=assets.wav_path, duration_seconds=duration)
    return assets

# play WAV in-browser, or offer a MIDI download when synthesis is unavailable
def audio_widget(wav_path: Path | None, label: str, *, midi_path: Path | None = None) -> None:
    st.markdown(f"**{label}**")
    resolved_midi = midi_path
    if resolved_midi is None and wav_path is not None:
        # prepare_audio uses the same stem for .mid and .wav (see render_midi / render_wav)
        resolved_midi = wav_path.with_suffix(".mid")
    # play WAV in-browser
    if wav_path is not None and wav_path.is_file():
        st.audio(wav_path.read_bytes(), format="audio/wav")
        if resolved_midi is not None and resolved_midi.is_file():
            st.download_button(
                "Download MIDI",
                data=resolved_midi.read_bytes(),
                file_name=resolved_midi.name,
                mime="audio/midi",
                key=f"midi_dl_inline_{resolved_midi.name}_{label}",
            )
        return

    # check if the audio setup is available
    # offer a MIDI download when synthesis is unavailable
    setup = check_audio_setup()
    err = get_last_synthesis_error()
    if err:
        st.warning(f"WAV synthesis failed: {err}")
    elif not setup["soundfont_present"]:
        st.warning("Soundfont missing — the app will download one on the next generate.")
    elif not setup["can_synthesize_wav"]:
        st.warning("No WAV synthesizer available (install FluidSynth or use pyfluidsynth).")

    if resolved_midi is not None and resolved_midi.is_file():
        st.download_button(
            label="Download MIDI",
            data=resolved_midi.read_bytes(),
            file_name=resolved_midi.name,
            mime="audio/midi",
            key=f"midi_download_{resolved_midi.name}_{label}",
        )
        return
    st.warning("No audio could be prepared for this composition.")

# convert a WAV file to a base64-encoded string
def wav_to_base64(wav_path: Path) -> str:
    return base64.b64encode(wav_path.read_bytes()).decode("ascii")