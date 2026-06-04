"""
main entry point for the Markov Music Engine

Usage :
    python main.py --style classical --order 1 --n-chords 16 --tempo 120
    python main.py --style jazz --order 2 --compare

"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence
from config import (
    DEFAULT_N_CHORDS,
    DEFAULT_ORDER,
    DEFAULT_TEMPO_BPM,
    OUTPUTS_DIR,
    SUPPORTED_ORDERS,
    SUPPORTED_STYLES,
)
from markov.analysis import summarise
from markov.data import load_corpus
from markov.encoder import (
    ChordToken,
    build_chord_vocabulary,
    build_note_vocabulary,
    chord_vocabulary_inverse,
    encode_chords,
)
from markov.generator import Composer
from markov.harmony import ChordChain
from markov.matrix import HierarchicalMarkovModel
from markov.melody import MelodyChain
from markov.parser import ParseError, parse_midi
from markov.renderer import render_midi, render_wav

logger = logging.getLogger(__name__)

_CHORD_VOCAB_FILE = "chord_vocab.json"
_TOP_STATIONARY_ROWS = 10

# build the command-line parser
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a hierarchical Markov model, generate music, and export MIDI/WAV.")
    parser.add_argument(
        "--style",
        required=True,
        choices=SUPPORTED_STYLES,
        help="Corpus style (classical, jazz, or pop).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=DEFAULT_ORDER,
        choices=SUPPORTED_ORDERS,
        help=f"Melody Markov order (default: {DEFAULT_ORDER}). Ignored when --compare is set.",
    )
    parser.add_argument(
        "--n-chords",
        type=int,
        default=DEFAULT_N_CHORDS,
        dest="n_chords",
        metavar="N",
        help=f"Number of chord steps to generate (default: {DEFAULT_N_CHORDS}).",
    )
    parser.add_argument(
        "--notes-per-chord",
        type=int,
        default=4,
        dest="notes_per_chord",
        metavar="N",
        help="Melody notes sampled per chord (default: 4).",
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=DEFAULT_TEMPO_BPM,
        metavar="BPM",
        help=f"Tempo in beats per minute (default: {DEFAULT_TEMPO_BPM}).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate order-1 and order-2 outputs side by side (MIDI and WAV).",
    )
    parser.add_argument(
        "--save-model",
        metavar="DIR",
        dest="save_model",
        type=Path,
        help="Save the trained model under DIR (skips retraining when used with --load-model).",
    )
    parser.add_argument(
        "--load-model",
        metavar="DIR",
        dest="load_model",
        type=Path,
        help="Load a previously saved model from DIR instead of retraining.",
    )
    return parser

# save the chord vocabulary to a file
def _save_chord_vocab(chord_to_index: dict[ChordToken, int], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _CHORD_VOCAB_FILE).write_text(
        json.dumps(chord_to_index, indent=2),
        encoding="utf-8",
    )

# load the chord vocabulary from a file
def _load_index_to_chord(directory: Path, vocab_size: int) -> list[ChordToken]:
    vocab_path = directory / _CHORD_VOCAB_FILE
    if vocab_path.is_file():
        chord_to_index: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        return chord_vocabulary_inverse(chord_to_index)
    return [f"chord_{i}" for i in range(vocab_size)]

# collect chord sequences from a list of MIDI files
def _collect_chord_sequences(paths: Sequence[Path]) -> list[list[ChordToken]]:
    sequences: list[list[ChordToken]] = []
    for path in paths:
        try:
            chord_sequence, _ = parse_midi(path)
            sequences.append(chord_sequence)
        except ParseError as exc:
            logger.warning("skipping %s: %s", path, exc)
    return sequences

# train the model on a list of MIDI files
def train_model(paths: Sequence[Path], order: int) -> tuple[HierarchicalMarkovModel, list[ChordToken], dict[ChordToken, int]]:
    chord_sequences = _collect_chord_sequences(paths)
    if not chord_sequences:
        raise RuntimeError("No chord sequences could be parsed from the corpus.")

    chord_to_index = build_chord_vocabulary(chord_sequences)
    index_to_chord = chord_vocabulary_inverse(chord_to_index)
    note_to_index = build_note_vocabulary()

    harmony = ChordChain(vocab_size=len(chord_to_index))
    melody = MelodyChain(order=order)
    model = HierarchicalMarkovModel(harmony=harmony, melody=melody)
    model.train(paths, parse_midi, encode_chords, chord_to_index, note_to_index)
    return model, index_to_chord, chord_to_index

# load a model bundle from a directory
def load_model_bundle(directory: Path) -> tuple[HierarchicalMarkovModel, list[ChordToken]]:
    model = HierarchicalMarkovModel.load(directory)
    assert model.harmony.vocab_size is not None
    index_to_chord = _load_index_to_chord(directory, model.harmony.vocab_size)
    return model, index_to_chord

# save a model bundle to a directory
def save_model_bundle(model: HierarchicalMarkovModel, directory: Path, chord_to_index: dict[ChordToken, int] | None) -> None:
    model.save(directory)
    if chord_to_index is not None:
        _save_chord_vocab(chord_to_index, directory)

# print the analysis of a chord transition matrix
def print_analysis(summary: dict[str, object], *, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"Dominant chord:      {summary['dominant_chord']} ({summary['dominant_chord_pct']:.1f}%)")
    print(f"Chain entropy:       {summary['entropy_bits']:.3f} bits")
    print(f"Mixing time (est.):  {summary['mixing_time_steps']} steps")

    distribution: dict[str, float] = summary["stationary_distribution"]  # type: ignore[assignment]
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    print(f"\nStationary distribution (top {min(_TOP_STATIONARY_ROWS, len(ranked))}):")
    for chord, probability in ranked[:_TOP_STATIONARY_ROWS]:
        print(f"  {chord:<24} {probability:.4f}")
    if len(ranked) > _TOP_STATIONARY_ROWS:
        print(f"  ... ({len(ranked) - _TOP_STATIONARY_ROWS} more chord(s) omitted)")

# run the generation pipeline
def run_generation(
    model: HierarchicalMarkovModel,
    index_to_chord: Sequence[ChordToken],
    *,
    style: str,
    order: int,
    n_chords: int,
    notes_per_chord: int,
    tempo_bpm: int,
    output_stem: str,
) -> None:
    composer = Composer(model)
    result = composer.compose(
        style=style,
        n_chords=n_chords,
        order=order,
        notes_per_chord=notes_per_chord,
        tempo_bpm=tempo_bpm,
    )

    midi_path = render_midi(result, f"{output_stem}.mid")
    print(f"MIDI written: {midi_path}")

    try:
        wav_path = render_wav(midi_path, f"{output_stem}.wav")
        print(f"WAV written:  {wav_path}")
    except RuntimeError as exc:
        print(f"WAV skipped:  {exc}", file=sys.stderr)

    matrix = model.harmony.transition_matrix
    if matrix is None:
        raise RuntimeError("Harmony transition matrix is missing after training.")
    print_analysis(summarise(matrix, index_to_chord), title=f"Harmony analysis - {style}, order {order}")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.n_chords < 1:
        print("error: --n-chords must be at least 1", file=sys.stderr)
        return 2
    if args.notes_per_chord < 1:
        print("error: --notes-per-chord must be at least 1", file=sys.stderr)
        return 2
    if args.tempo <= 0:
        print("error: --tempo must be positive", file=sys.stderr)
        return 2

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    orders = [1, 2] if args.compare else [args.order]
    corpus_paths = load_corpus(args.style)

    # train or load the model for each order
    for order in orders:
        if args.load_model is not None:
            model_dir = args.load_model / f"order{order}" if args.compare else args.load_model
            if not model_dir.is_dir():
                print(f"error: model directory not found: {model_dir}", file=sys.stderr)
                return 1
            logger.info("Loading model from %s", model_dir)
            model, index_to_chord = load_model_bundle(model_dir)
            chord_to_index = None
        else:
            logger.info("Training model (order=%s) on %s file(s)", order, len(corpus_paths))
            model, index_to_chord, chord_to_index = train_model(corpus_paths, order)
            if args.save_model is not None:
                save_dir = args.save_model / f"order{order}" if args.compare else args.save_model
                logger.info("Saving model to %s", save_dir)
                save_model_bundle(model, save_dir, chord_to_index)

        if model.melody.order != order:
            print(f"error: loaded melody order is {model.melody.order}, but run requested order {order}", file=sys.stderr)
            return 1

        output_stem = f"{args.style}_order{order}"
        run_generation(
            model,
            index_to_chord,
            style=args.style,
            order=order,
            n_chords=args.n_chords,
            notes_per_chord=args.notes_per_chord,
            tempo_bpm=args.tempo,
            output_stem=output_stem,
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())