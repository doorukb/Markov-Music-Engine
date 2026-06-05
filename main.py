"""
main entry point for the Markov Music Engine

Example Usage :
    python main.py --style classical --order 1 --n-chords 16 --tempo 120
    python main.py --style jazz --order 2 --compare --play
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
from markov.generator import ComparisonResult, Composer, CompositionResult
from markov.harmony import ChordChain
from markov.matrix import HierarchicalMarkovModel
from markov.melody import MelodyChain
from markov.parser import ParseError, parse_midi
from markov.renderer import render_midi, render_wav

logger = logging.getLogger(__name__)

_CHORD_VOCAB_FILE = "chord_vocab.json"
_TOP_STATIONARY_ROWS = 10
_MEASURE_QUARTER_LENGTH = 4.0

# calculate the duration of a composition in seconds
def _composition_duration_seconds(result: CompositionResult) -> float:
    return len(result.composition) * _MEASURE_QUARTER_LENGTH * 60.0 / result.tempo_bpm

# play a MIDI file with progress bars
def play_midi_with_progress(path: Path, label: str, duration_seconds: float) -> None:
    import time
    import pygame
    from tqdm import tqdm

    print(f"\nNow playing: {label}")
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()

    bar_format = "{l_bar}{bar}| {n:.1f}/{total:.1f}s"
    with tqdm(
        total=round(duration_seconds, 1),
        desc=label,
        unit="s",
        bar_format=bar_format,
        ncols=70,
    ) as pbar:
        prev = 0.0
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            elapsed = min(pygame.mixer.music.get_pos() / 1000.0, duration_seconds)
            pbar.update(round(elapsed - prev, 1))
            prev = elapsed

# play a sequence of MIDI files with progress bars
def _play_sequence(tracks: list[tuple[str, Path, float]]) -> None:
    import pygame

    pygame.mixer.init()
    try:
        for label, path, duration in tracks:
            play_midi_with_progress(path, label, duration)
    finally:
        pygame.mixer.quit()


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
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play generated MIDI file(s) in sequence after generation (requires pygame).",
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
def save_model_bundle(
    model: HierarchicalMarkovModel,
    directory: Path,
    chord_to_index: dict[ChordToken, int] | None,
) -> None:
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

# print the comparison analysis of two models
def print_comparison_analysis(comparison: ComparisonResult, *, style: str) -> None:
    model1, model2 = comparison.models
    matrix1 = model1.harmony.transition_matrix
    matrix2 = model2.harmony.transition_matrix
    if matrix1 is None or matrix2 is None:
        raise RuntimeError("Harmony transition matrix is missing after training.")

    summary1 = summarise(matrix1, comparison.index_to_chord)
    summary2 = summarise(matrix2, comparison.index_to_chord)

    print(f"\n{'=' * 60}")
    print(f"Side-by-side harmony analysis - {style}")
    print(f"{'=' * 60}")
    print(f"{'':24} {'Order 1':>12} {'Order 2':>12}")
    print(f"{'Chain entropy (bits)':24} {summary1['entropy_bits']:>12.3f} {summary2['entropy_bits']:>12.3f}")
    print(
        f"{'Mixing time (steps)':24} "
        f"{summary1['mixing_time_steps']:>12} "
        f"{summary2['mixing_time_steps']:>12}"
    )
    print(
        f"{'Dominant chord':24} "
        f"{str(summary1['dominant_chord']):>12} "
        f"{str(summary2['dominant_chord']):>12}"
    )
    print(
        f"{'Dominant chord (%)':24} "
        f"{summary1['dominant_chord_pct']:>11.1f}% "
        f"{summary2['dominant_chord_pct']:>11.1f}%"
    )

# render a composition to a MIDI and WAV file
def render_composition(result: CompositionResult, output_stem: str) -> Path:
    midi_path = render_midi(result, f"{output_stem}.mid")
    print(f"MIDI written: {midi_path}")

    try:
        wav_path = render_wav(midi_path, f"{output_stem}.wav")
        print(f"WAV written:  {wav_path}")
    except RuntimeError as exc:
        print(f"WAV skipped:  {exc}", file=sys.stderr)
    return midi_path

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
) -> tuple[Path, CompositionResult]:
    composer = Composer(model)
    result = composer.compose(
        style=style,
        n_chords=n_chords,
        order=order,
        notes_per_chord=notes_per_chord,
        tempo_bpm=tempo_bpm,
    )
    midi_path = render_composition(result, output_stem)

    matrix = model.harmony.transition_matrix
    if matrix is None:
        raise RuntimeError("Harmony transition matrix is missing after training.")
    print_analysis(
        summarise(matrix, index_to_chord),
        title=f"Harmony analysis - {style}, order {order}",
    )
    return midi_path, result

# run the comparison pipeline
def run_compare(
    *,
    style: str,
    n_chords: int,
    notes_per_chord: int,
    tempo_bpm: int,
    corpus_paths: Sequence[Path],
    load_model: Path | None,
    save_model: Path | None,
    play: bool = False,
) -> int:
    models: tuple[HierarchicalMarkovModel, HierarchicalMarkovModel] | None = None
    index_to_chord: list[ChordToken] | None = None
    if load_model is not None:
        dir1 = load_model / "order1"
        dir2 = load_model / "order2"
        if not dir1.is_dir() or not dir2.is_dir():
            print(
                f"error: compare mode expects {dir1} and {dir2}",
                file=sys.stderr,
            )
            return 1
        logger.info("Loading models from %s and %s", dir1, dir2)
        model1, index_to_chord = load_model_bundle(dir1)
        model2, _ = load_model_bundle(dir2)
        models = (model1, model2)
    else:
        logger.info("Training order-1 and order-2 models on %s file(s)", len(corpus_paths))

    comparison = Composer.compare(
        style,
        n_chords,
        notes_per_chord=notes_per_chord,
        tempo_bpm=tempo_bpm,
        corpus_paths=corpus_paths if models is None else None,
        models=models,
        index_to_chord=index_to_chord,
    )

    if save_model is not None and models is None:
        chord_sequences = _collect_chord_sequences(corpus_paths)
        chord_to_index = build_chord_vocabulary(chord_sequences)
        for order, model in zip((1, 2), comparison.models, strict=True):
            save_dir = save_model / f"order{order}"
            logger.info("Saving model to %s", save_dir)
            save_model_bundle(model, save_dir, chord_to_index)

    midi1 = render_composition(comparison.order1, f"{style}_order1")
    midi2 = render_composition(comparison.order2, f"{style}_order2")
    print_comparison_analysis(comparison, style=style)

    if play:
        _play_sequence(
            [
                (f"{style} order 1", midi1, _composition_duration_seconds(comparison.order1)),
                (f"{style} order 2", midi2, _composition_duration_seconds(comparison.order2)),
            ]
        )
    return 0

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
    corpus_paths = load_corpus(args.style)

    if args.compare:
        return run_compare(
            style=args.style,
            n_chords=args.n_chords,
            notes_per_chord=args.notes_per_chord,
            tempo_bpm=args.tempo,
            corpus_paths=corpus_paths,
            load_model=args.load_model,
            save_model=args.save_model,
            play=args.play,
        )

    if args.load_model is not None:
        if not args.load_model.is_dir():
            print(f"error: model directory not found: {args.load_model}", file=sys.stderr)
            return 1
        logger.info("Loading model from %s", args.load_model)
        model, index_to_chord = load_model_bundle(args.load_model)
        chord_to_index = None
    else:
        logger.info("Training model (order=%s) on %s file(s)", args.order, len(corpus_paths))
        model, index_to_chord, chord_to_index = train_model(corpus_paths, args.order)
        if args.save_model is not None:
            logger.info("Saving model to %s", args.save_model)
            save_model_bundle(model, args.save_model, chord_to_index)

    if model.melody.order != args.order:
        print(
            f"error: loaded melody order is {model.melody.order}, but run requested order {args.order}",
            file=sys.stderr,
        )
        return 1

    midi_path, result = run_generation(
        model,
        index_to_chord,
        style=args.style,
        order=args.order,
        n_chords=args.n_chords,
        notes_per_chord=args.notes_per_chord,
        tempo_bpm=args.tempo,
        output_stem=f"{args.style}_order{args.order}",
    )
    if args.play:
        _play_sequence(
            [
                (
                    f"{args.style} order {args.order}",
                    midi_path,
                    _composition_duration_seconds(result),
                )
            ]
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())