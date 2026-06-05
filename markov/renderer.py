from __future__ import annotations
import logging
import shutil
import subprocess
import time
from pathlib import Path
from music21 import meter, note, stream, tempo
from config import AUDIO_FORMAT, OUTPUTS_DIR, SAMPLE_RATE, SOUNDFONT_PATH
from markov.audio_setup import ensure_soundfont
from markov.generator import CompositionResult

__all__ = ["render_midi", "render_wav", "composition_to_stream"]

logger = logging.getLogger(__name__)

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
    time_sig = meter.TimeSignature("4/4")
    for measure_number, (_chord_index, note_indices) in enumerate(composition_result.composition, start=1):
        measure = stream.Measure(number=measure_number)
        if measure_number == 1:
            measure.insert(0, time_sig)

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
def render_midi(composition_result: CompositionResult, output_path: str | Path) -> Path:
    midi_path = _resolve_output_path(output_path, ".mid")
    score = composition_to_stream(composition_result)
    score.write("midi", str(midi_path))
    logger.info("MIDI written: %s", midi_path)
    return midi_path

# render a MIDI file to a WAV file using the CLI fluidsynth
def _render_wav_cli(midi_path: Path, wav_path: Path) -> None:
    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is None:
        raise RuntimeError("fluidsynth executable not on PATH")

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
    subprocess.run(command, check=True, capture_output=True, text=True)

# render a MIDI file to a WAV file using the pyfluidsynth library
def _render_wav_pyfluidsynth(midi_path: Path, wav_path: Path) -> None:
    import fluidsynth

    fs = fluidsynth.Synth(samplerate=float(SAMPLE_RATE))
    try:
        fs.start(driver="file", filename=str(wav_path), filetype="wav")
        sfid = fs.sfload(str(SOUNDFONT_PATH))
        if sfid == -1:
            raise RuntimeError(f"pyfluidsynth failed to load soundfont at {SOUNDFONT_PATH}")
        fs.program_select(0, sfid, 0, 0)
        status = fs.midi_file_play(str(midi_path))
        if status != 0:
            raise RuntimeError(f"pyfluidsynth midi_file_play returned {status}")
        deadline = time.time() + max(120.0, midi_path.stat().st_size / 100.0)
        while time.time() < deadline:
            if hasattr(fs, "get_playing") and not fs.get_playing():
                break
            time.sleep(0.05)
    finally:
        fs.delete()

# render a MIDI file to a WAV file using either the CLI fluidsynth or the pyfluidsynth library
def render_wav(midi_path: str | Path, output_path: str | Path) -> Path:
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    ensure_soundfont()
    if not SOUNDFONT_PATH.is_file():
        raise RuntimeError(f"Soundfont not found at {SOUNDFONT_PATH}")

    wav_path = _resolve_output_path(output_path, f".{AUDIO_FORMAT}")
    errors: list[str] = []

    if shutil.which("fluidsynth"):
        try:
            _render_wav_cli(midi_path, wav_path)
            if wav_path.is_file():
                logger.info("WAV written (CLI): %s", wav_path)
                return wav_path
        except Exception as exc:
            errors.append(f"CLI FluidSynth: {exc}")
            wav_path.unlink(missing_ok=True)

    try:
        _render_wav_pyfluidsynth(midi_path, wav_path)
    except Exception as exc:
        errors.append(f"pyfluidsynth: {exc}")
        wav_path.unlink(missing_ok=True)

    if wav_path.is_file():
        logger.info("WAV written (pyfluidsynth): %s", wav_path)
        return wav_path

    detail = "; ".join(errors) if errors else "no synthesizer available"
    raise RuntimeError(f"Could not render WAV for {midi_path}: {detail}")