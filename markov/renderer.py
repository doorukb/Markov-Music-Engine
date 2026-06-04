"""
MIDI file writing and audio synthesis
- Convert a generated (chord, [notes]) sequence into a music21 Stream
- Write the Stream to a MIDI file in outputs/
- Call FluidSynth to synthesize MIDI to WAV using the configured soundfont
- Return the path to the WAV file for Streamlit's st.audio() widget
- Handle FluidSynth availability gracefully (fallback: MIDI-only mode)
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from music21 import meter, note, stream, tempo
from config import AUDIO_FORMAT, OUTPUTS_DIR, SAMPLE_RATE, SOUNDFONT_PATH
from markov.generator import CompositionResult

__all__ = ["render_midi", "render_wav", "composition_to_stream"]

MEASURE_QUARTER_LENGTH = 4.0

# resolve the output path for a MIDI or WAV file
def _resolve_output_path(output_path: str | Path, suffix: str) -> Path:
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(suffix)
    elif path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)

    if not path.is_absolute():
        path = OUTPUTS_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path

# build a music21 Score from a CompositionResult
def composition_to_stream(composition_result: CompositionResult) -> stream.Score:
    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=composition_result.tempo_bpm))

    measure_offset = 0.0
    for measure_number, (_chord_index, note_indices) in enumerate(
        composition_result.composition, start=1
    ):
        measure = stream.Measure(number=measure_number)
        measure.insert(0, meter.TimeSignature("4/4"))

        note_count = len(note_indices)
        if note_count > 0:
            step = MEASURE_QUARTER_LENGTH / note_count
            for i, midi_pitch in enumerate(note_indices):
                event = note.Note(midi=int(midi_pitch), quarterLength=step)
                measure.insert(i * step, event)

        part.insert(measure_offset, measure)
        measure_offset += MEASURE_QUARTER_LENGTH

    score.insert(0, part)
    return score

# render a CompositionResult to a MIDI file
# each harmony step occupies one 4/4 measure- melody notes are evenly spaced within that measure at composition_result.tempo_bpm
def render_midi(
    composition_result: CompositionResult,
    output_path: str | Path,
) -> Path:
    midi_path = _resolve_output_path(output_path, ".mid")
    score = composition_to_stream(composition_result)
    score.write("midi", str(midi_path))
    return midi_path

# render a MIDI file to a WAV file using FluidSynth
# raise RuntimeError if FluidSynth is not on PATH or the soundfont file is missing
def render_wav(midi_path: str | Path, output_path: str | Path) -> Path:
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    wav_path = _resolve_output_path(output_path, f".{AUDIO_FORMAT}")

    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is None:
        raise RuntimeError("FluidSynth is not available: 'fluidsynth' was not found on PATH. Install FluidSynth and ensure the executable is available to render WAV audio.")

    if not SOUNDFONT_PATH.is_file():
        raise RuntimeError(f"Soundfont not found at {SOUNDFONT_PATH}. Add a .sf2 file at that path or update config.SOUNDFONT_PATH.")

    command = [
        fluidsynth,
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        str(SAMPLE_RATE),
        str(SOUNDFONT_PATH),
        str(midi_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"FluidSynth failed to render {midi_path} to WAV: {stderr or exc}") from exc

    if not wav_path.is_file():
        raise RuntimeError(f"FluidSynth did not produce expected WAV file: {wav_path}")
    return wav_path