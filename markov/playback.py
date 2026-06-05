from __future__ import annotations
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Sequence
from config import DEFAULT_TEMPO_BPM, OUTPUTS_DIR, SUPPORTED_STYLES
from markov.data import load_corpus
from markov.encoder import ChordToken, chord_vocabulary_inverse
from markov.generator import CompositionResult
from markov.matrix import HierarchicalMarkovModel

__all__ = [
    "MEASURE_QUARTER_LENGTH",
    "CHORD_VOCAB_FILE",
    "seconds_to_n_chords",
    "composition_duration_seconds",
    "midi_file_duration_seconds",
    "resolve_source",
    "export_original_midi",
    "load_model_bundle",
    "save_model_bundle",
    "load_models_for_orders",
]

logger = logging.getLogger(__name__)

MEASURE_QUARTER_LENGTH = 4.0
CHORD_VOCAB_FILE = "chord_vocab.json"

# convert seconds to the number of chords at a given tempo
def seconds_to_n_chords(seconds: float, tempo_bpm: int) -> int:
    return max(1, round(seconds * tempo_bpm / (MEASURE_QUARTER_LENGTH * 60.0)))

# convert a CompositionResult to the number of seconds it takes to play
def composition_duration_seconds(result: CompositionResult) -> float:
    return len(result.composition) * MEASURE_QUARTER_LENGTH * 60.0 / result.tempo_bpm

# convert a MIDI file to the number of seconds it takes to play
def midi_file_duration_seconds(path: Path, *, tempo_bpm: int = DEFAULT_TEMPO_BPM) -> float:
    from music21 import converter

    if not path.is_file():
        return 60.0
    try:
        score = converter.parse(str(path))
    except Exception:
        return 60.0
    seconds = getattr(score, "seconds", None)
    if seconds and seconds > 0 and not (isinstance(seconds, float) and seconds != seconds):
        return float(seconds)
    return float(score.duration.quarterLength) * 60.0 / tempo_bpm

# resolve the path to the source piece
def resolve_source(source_arg: str | None, style: str, corpus_paths: Sequence[Path]) -> Path:
    if not source_arg or not str(source_arg).strip():
        return random.choice(list(corpus_paths))
    text = str(source_arg).strip()
    p = Path(text)
    if p.is_file():
        return p
    if text in SUPPORTED_STYLES:
        return random.choice(load_corpus(text))
    logger.warning("source %r is not a file or known style; using a random %s piece.", text, style)
    return random.choice(list(corpus_paths))

# resolve the path to the source piece
def _resolve_playback_score(source_path: Path):
    from music21 import converter, stream

    parsed = converter.parse(str(source_path))
    if isinstance(parsed, stream.Opus):
        if not parsed.scores:
            raise RuntimeError(f"No scores found in opus file: {source_path}")
        return parsed.scores[0]
    return parsed

# export the original MIDI file
def export_original_midi(source_path: Path, output_stem: str) -> tuple[Path, float]:
    from music21.exceptions21 import StreamException

    midi_path = OUTPUTS_DIR / f"{output_stem}.mid"
    if source_path.suffix.lower() in {".mid", ".midi"} and source_path.is_file():
        shutil.copy2(source_path, midi_path)
    else:
        score = _resolve_playback_score(source_path)
        try:
            score.write("midi", str(midi_path))
        except StreamException:
            if score.parts:
                score.parts[0].write("midi", str(midi_path))
            else:
                score.flatten().write("midi", str(midi_path))
    if not midi_path.is_file():
        raise RuntimeError(f"Failed to write playback MIDI: {midi_path}")
    return midi_path, midi_file_duration_seconds(midi_path)

# save the chord vocabulary to a file
def _save_chord_vocab(chord_to_index: dict[ChordToken, int], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHORD_VOCAB_FILE).write_text(json.dumps(chord_to_index, indent=2), encoding="utf-8")

# load the chord vocabulary from a file
def _load_index_to_chord(directory: Path, vocab_size: int) -> list[ChordToken]:
    vocab_path = directory / CHORD_VOCAB_FILE
    if vocab_path.is_file():
        chord_to_index: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        return chord_vocabulary_inverse(chord_to_index)
    return [f"chord_{i}" for i in range(vocab_size)]

# load a model bundle from a directory
def load_model_bundle(directory: Path) -> tuple[HierarchicalMarkovModel, list[ChordToken]]:
    model = HierarchicalMarkovModel.load(directory)
    if model.harmony.vocab_size is None:
        raise RuntimeError(f"Loaded model at {directory} has no harmony vocabulary size.")
    index_to_chord = _load_index_to_chord(directory, model.harmony.vocab_size)
    return model, index_to_chord

# save a model bundle to a directory
def save_model_bundle(model: HierarchicalMarkovModel, directory: Path, chord_to_index: dict[ChordToken, int] | None) -> None:
    model.save(directory)
    if chord_to_index is not None:
        _save_chord_vocab(chord_to_index, directory)

# load models for multiple orders
def load_models_for_orders(load_model: Path, orders: Sequence[int]) -> tuple[dict[int, HierarchicalMarkovModel], list[ChordToken]]:
    models: dict[int, HierarchicalMarkovModel] = {}
    index_to_chord: list[ChordToken] | None = None
    for order in orders:
        order_dir = load_model / f"order{order}"
        if not order_dir.is_dir():
            raise FileNotFoundError(f"multi-order load expects {order_dir}")
        model, loaded_index = load_model_bundle(order_dir)
        if model.melody.order != order:
            raise ValueError(f"loaded model at {order_dir} has melody order {model.melody.order}, expected {order}")
        models[order] = model
        if index_to_chord is None:
            index_to_chord = loaded_index
    if index_to_chord is None:
        raise ValueError("load_models_for_orders() requires at least one order.")
    return models, index_to_chord