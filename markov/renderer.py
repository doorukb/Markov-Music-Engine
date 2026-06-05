from __future__ import annotations
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from music21 import meter, note, stream, tempo
from config import AUDIO_FORMAT, FLUIDSYNTH_BIN_DIR, OUTPUTS_DIR, SAMPLE_RATE, SOUNDFONT_PATH
from markov.audio_setup import (
    add_fluidsynth_dll_directory,
    ensure_fluidsynth_binary,
    ensure_soundfont,
    get_fluidsynth_executable,
)
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
# each harmony step occupies one 4/4 measure- melody notes are evenly spaced within that measure at composition_result.tempo_bpm
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

# build the environment for the fluidsynth subprocess
def _fluidsynth_subprocess_env(exe: Path) -> tuple[dict[str, str], str | None]:
    env = os.environ.copy()
    cwd: str | None = None
    bin_dir = FLUIDSYNTH_BIN_DIR if FLUIDSYNTH_BIN_DIR.is_dir() else exe.parent

    if sys.platform == "win32" and bin_dir.is_dir():
        env["PATH"] = str(bin_dir.resolve()) + os.pathsep + env.get("PATH", "")
        cwd = str(bin_dir.resolve())
    return env, cwd

# render a MIDI file to a WAV file using the CLI fluidsynth
def _render_wav_cli(midi_path: Path, wav_path: Path) -> None:
    fluidsynth = get_fluidsynth_executable()
    if fluidsynth is None:
        raise RuntimeError("fluidsynth executable not available")
    command = [
        str(fluidsynth),
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        str(SAMPLE_RATE),
        str(SOUNDFONT_PATH),
        str(midi_path),
    ]

    env, cwd = _fluidsynth_subprocess_env(fluidsynth)
    subprocess.run(command, check=True, capture_output=True, text=True, env=env, cwd=cwd)

# render a MIDI file to a WAV file using the pyfluidsynth library
# add the fluidsynth DLL directory to the PATH
def _render_wav_pyfluidsynth(midi_path: Path, wav_path: Path) -> None:
    add_fluidsynth_dll_directory()
    import fluidsynth
    fs = fluidsynth.Synth(samplerate=float(SAMPLE_RATE))

    try:
        fs.start(driver="file", filename=str(wav_path), filetype="wav")
        sfid = fs.sfload(str(SOUNDFONT_PATH))
        if sfid == -1:
            raise RuntimeError("pyfluidsynth failed to load soundfont")
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
# ensure the soundfont and fluidsynth binary are available
def render_wav(midi_path: str | Path, output_path: str | Path) -> Path:
    midi_path = Path(midi_path)

    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path.name}")

    ensure_soundfont()
    ensure_fluidsynth_binary()

    if not SOUNDFONT_PATH.is_file():
        raise RuntimeError("Soundfont not found — run make setup-audio")

    wav_path = _resolve_output_path(output_path, f".{AUDIO_FORMAT}")
    errors: list[str] = []

    if get_fluidsynth_executable() is not None:
        try:
            _render_wav_cli(midi_path, wav_path)
            if wav_path.is_file():
                logger.info("WAV written (CLI): %s", wav_path)
                # return the WAV path if it was written successfully
                return wav_path

        except Exception as exc:
            errors.append(f"CLI FluidSynth: {exc}")
            wav_path.unlink(missing_ok=True)

    try:
        _render_wav_pyfluidsynth(midi_path, wav_path)
        if wav_path.is_file():
            logger.info("WAV written (pyfluidsynth): %s", wav_path)
            # return the WAV path if it was written successfully
            return wav_path
    except Exception as exc:
        errors.append(f"pyfluidsynth: {exc}")
        wav_path.unlink(missing_ok=True)

    if wav_path.is_file():
        logger.info("WAV written (pyfluidsynth): %s", wav_path)
        return wav_path

    detail = "; ".join(errors) if errors else "no synthesizer available"
    raise RuntimeError(f"Could not render WAV: {detail}")