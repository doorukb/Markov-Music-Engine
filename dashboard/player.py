from __future__ import annotations
import base64
import logging
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from markov.audio_setup import ensure_fluidsynth_binary, ensure_soundfont
from markov.generator import CompositionResult
from markov.playback import export_original_midi, midi_file_duration_seconds
from markov.renderer import render_midi, render_wav

__all__ = [
    "PreparedAudio",
    "prepare_audio",
    "prepare_original_audio",
    "prepare_midi_wav",
    "get_last_synthesis_error",
    "set_last_synthesis_error",
    "wav_to_base64",
]

logger = logging.getLogger(__name__)
_LAST_ERROR_KEY = "_last_synthesis_error"
_USER_SYNTH_ERROR = ("WAV synthesis failed. Run `make setup-audio` or restart the dashboard after first-run download.")

# prepared audio data class
@dataclass(frozen=True)
class PreparedAudio:
    midi_path: Path # the path to the MIDI file
    wav_path: Path | None # the path to the WAV file
    duration_seconds: float = 0.0 # the duration of the audio in seconds

# set the last synthesis error
def set_last_synthesis_error(message: str | None) -> None:
    st.session_state[_LAST_ERROR_KEY] = message

# get the last synthesis error
def get_last_synthesis_error() -> str | None:
    return st.session_state.get(_LAST_ERROR_KEY)

# generate the user facing synthesis error
def _user_facing_synthesis_error(exc: Exception) -> str:
    logger.warning("WAV synthesis failed: %s", exc)
    return _USER_SYNTH_ERROR

# prepare the MIDI and WAV files
def prepare_midi_wav(midi_path: Path, wav_stem: str) -> PreparedAudio:
    set_last_synthesis_error(None)
    try:
        ensure_soundfont()
        ensure_fluidsynth_binary()
        wav_path = render_wav(midi_path, f"{wav_stem}.wav")
        duration = midi_file_duration_seconds(wav_path)
        return PreparedAudio(midi_path=midi_path, wav_path=wav_path, duration_seconds=duration)
    except Exception as exc:
        set_last_synthesis_error(_user_facing_synthesis_error(exc))
        duration = midi_file_duration_seconds(midi_path)
        return PreparedAudio(midi_path=midi_path, wav_path=None, duration_seconds=duration)

# prepare the audio for the composition result
def prepare_audio(composition_result: CompositionResult, output_stem: str) -> PreparedAudio:
    midi_path = render_midi(composition_result, f"{output_stem}.mid")
    assets = prepare_midi_wav(midi_path, output_stem)
    if assets.duration_seconds <= 0:
        from markov.playback import composition_duration_seconds
        # return the prepared audio with the duration of the composition
        return PreparedAudio(
            midi_path=assets.midi_path,
            wav_path=assets.wav_path,
            duration_seconds=composition_duration_seconds(composition_result),
        )
    return assets

# prepare the audio for the original audio
def prepare_original_audio(source_path: Path, output_stem: str) -> PreparedAudio:
    midi_path, duration = export_original_midi(source_path, output_stem)
    assets = prepare_midi_wav(midi_path, output_stem)
    if assets.duration_seconds <= 0:
        return PreparedAudio(midi_path=midi_path, wav_path=assets.wav_path, duration_seconds=duration)
    return assets

# convert the WAV file to a base64-encoded string
def wav_to_base64(wav_path: Path) -> str:
    return base64.b64encode(wav_path.read_bytes()).decode("ascii")