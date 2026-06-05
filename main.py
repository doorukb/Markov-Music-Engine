"""
Markov Music Engine - command-line entry point

Flags:
  --style {classical|jazz|pop}  Corpus/style to train on (required).

  --order {1|2|3}               Melody Markov order for single-order runs (default 1).

  --orders N [N ...]            Generate one or more orders (subset of {1,2,3}), e.g. --orders 1 3.
  
  --yes / -y                    Bypass the order-3 resource warning (for scripts/CI).
  
  --n-chords N                  Number of chords (each = one 4/4 measure) to generate.
  
  --duration SECONDS            Make generated pieces last ~SECONDS (overrides --n-chords).
  
  --notes-per-chord N           Melody notes per chord/measure (default 4).
  
  --tempo BPM                   Playback/score tempo (default 120).
  
  --compare                     Shorthand for --orders 1 2 (order-1 and order-2 side by side).
  
  
  --single-source               Train the model(s) on ONE piece so output matches the original.
  
  --source PATH|STYLE           Pick the original: exact file path, a style keyword
                                (random piece of that style), or omitted (random piece of --style).
  
  --play                        Play Original -> generated piece(s) with a progress bar ('s' to skip).
  
  --save-model DIR / --load-model DIR   Persist or reuse trained model(s) under per-order subdirs.

Order 3 uses sparse per-chord melody storage (not a dense 128^3 matrix). Before training or
loading order 3, the CLI prints a resource warning and asks [y/N] unless --yes is passed.

# Hear the original, then order 1, then order 2 (most common) - try each genre:
    python main.py --style classical --compare --play --single-source
    python main.py --style jazz      --compare --play --single-source
    python main.py --style pop       --compare --play --single-source

# Compare orders 1 and 3 only:
    python main.py --style jazz --orders 1 3 --yes --single-source

# Single order-3 run (non-interactive):
    python main.py --style classical --order 3 --yes --single-source --n-chords 8

# All three orders with analysis:
    python main.py --style pop --orders 1 2 3 --yes --duration 10

# Same, but pick the exact source piece, fixed ~30s generated length:
    python main.py --style jazz --compare --play --single-source --source "outputs/jazz_original.mid" --duration 30

# Analysis only (no audio):
    python main.py --style jazz --compare
    python main.py --style classical --order 1 --n-chords 16

# Save once per order subdir, then reuse without retraining:
    python main.py --style pop --compare --save-model models/pop
    python main.py --style pop --compare --play --load-model models/pop
    python main.py --style pop --order 3 --yes --save-model models/pop3
    python main.py --style pop --order 3 --yes --load-model models/pop3
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Sequence
from config import (
    DEFAULT_N_CHORDS,
    DEFAULT_ORDER,
    DEFAULT_TEMPO_BPM,
    ORDER3_RESOURCE_MESSAGE,
    ORDER3_WARNING_THRESHOLD,
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
from markov.generator import Composer, CompositionResult, MultiOrderResult
from markov.harmony import ChordChain
from markov.matrix import HierarchicalMarkovModel
from markov.melody import MelodyChain
from markov.parser import ParseError, parse_midi
from markov.renderer import render_midi, render_wav

logger = logging.getLogger(__name__)

_CHORD_VOCAB_FILE = "chord_vocab.json"
_TOP_STATIONARY_ROWS = 10
_MEASURE_QUARTER_LENGTH = 4.0

# one chord = one 4/4 measure = 240/tempo seconds
def _seconds_to_n_chords(seconds: float, tempo_bpm: int) -> int:
    return max(1, round(seconds * tempo_bpm / (_MEASURE_QUARTER_LENGTH * 60.0)))

# calculate the duration of a composition in seconds
def _composition_duration_seconds(result: CompositionResult) -> float:
    return len(result.composition) * _MEASURE_QUARTER_LENGTH * 60.0 / result.tempo_bpm

# measure duration from the written MIDI file pygame actually plays
def _midi_file_duration_seconds(path: Path, *, tempo_bpm: int = DEFAULT_TEMPO_BPM) -> float:
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

def _resolve_source(source_arg: str | None, style: str, corpus_paths: Sequence[Path]) -> Path:
    if not source_arg:
        return random.choice(list(corpus_paths))
    p = Path(source_arg)
    if p.is_file():
        return p
    if source_arg in SUPPORTED_STYLES:
        return random.choice(load_corpus(source_arg))
    logger.warning(
        "--source %r is not a file or known style; using a random %s piece.",
        source_arg,
        style,
    )
    return random.choice(list(corpus_paths))

# resolve the effective orders to generate
def _resolve_effective_orders(args: argparse.Namespace) -> list[int]:
    if args.orders is not None:
        return sorted(set(args.orders))
    if args.compare:
        return [1, 2]
    return [args.order]

# confirm the generation of order-3
def _confirm_order3(orders: Sequence[int], yes: bool) -> bool:
    if ORDER3_WARNING_THRESHOLD not in orders or yes:
        return True
    print(ORDER3_RESOURCE_MESSAGE, end="")
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = ""
    if answer == "y":
        return True
    print("Aborted: order-3 generation cancelled.")
    return False

def _resolve_playback_score(source_path: Path):
    from music21 import converter, stream

    parsed = converter.parse(str(source_path))
    if isinstance(parsed, stream.Opus):
        if not parsed.scores:
            raise RuntimeError(f"No scores found in opus file: {source_path}")
        return parsed.scores[0]
    return parsed

# export the original MIDI source file to a MIDI file
def _export_original_midi(source_path: Path, output_stem: str) -> tuple[Path, float]:
    import shutil
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
    print(f"Original MIDI written: {midi_path}")
    return midi_path, _midi_file_duration_seconds(midi_path)

# play a MIDI file with progress bars
def play_midi_with_progress(path: Path, label: str, duration_seconds: float) -> None:
    import time
    import pygame
    from tqdm import tqdm

    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    print(f"\nNow playing: {label}   (press 's' to skip)")
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
            if msvcrt and msvcrt.kbhit() and msvcrt.getch() in (b"s", b"S", b"\r"):
                pygame.mixer.music.stop()
                print(f"Skipped {label}.")
                break
            time.sleep(0.1)
            elapsed = min(pygame.mixer.music.get_pos() / 1000.0, duration_seconds)
            pbar.update(round(elapsed - prev, 1))
            prev = elapsed
        pbar.n = pbar.total
        pbar.refresh()

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
        help=f"Melody Markov order for single-order runs (default: {DEFAULT_ORDER}). Ignored when --compare or --orders is set.",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        choices=SUPPORTED_ORDERS,
        metavar="N",
        help="Generate one or more melody orders (subset of {1,2,3}). Overrides --order.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Bypass the order-3 resource warning prompt.",
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
        "--duration",
        type=float,
        metavar="SECONDS",
        help="Target length for generated pieces in seconds (overrides --n-chords).",
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
        help="Shorthand for --orders 1 2: generate order-1 and order-2 outputs side by side.",
    )
    parser.add_argument(
        "--single-source",
        action="store_true",
        help="Train on one source piece so generated output matches the original.",
    )
    parser.add_argument(
        "--source",
        metavar="PATH|STYLE",
        help="Source piece: file path, style keyword (random piece), or omit for random --style piece.",
    )
    parser.add_argument(
        "--save-model",
        metavar="DIR",
        dest="save_model",
        type=Path,
        help="Save trained model(s) under DIR/order{N} (multi-order) or DIR (single-order).",
    )
    parser.add_argument(
        "--load-model",
        metavar="DIR",
        dest="load_model",
        type=Path,
        help="Load previously saved model(s) from DIR/order{N} (multi-order) or DIR (single-order).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play original source, then generated MIDI file(s) in sequence (requires pygame).",
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

# print the analysis of a multi-order result
def print_orders_analysis(multi: MultiOrderResult, *, style: str) -> None:
    orders = multi.orders
    summaries = []
    for order, _, model in multi.results:
        matrix = model.harmony.transition_matrix
        if matrix is None:
            raise RuntimeError(f"Harmony transition matrix is missing for order {order}.")
        summaries.append((order, summarise(matrix, multi.index_to_chord)))

    col_w = 12
    header = f"{'':24}" + "".join(f"{f'Order {o}':>{col_w}}" for o in orders)
    print(f"\n{'=' * 60}")
    print(f"Side-by-side harmony analysis - {style}")
    print(f"{'=' * 60}")
    print(header)

    ent_parts = "".join(f"{s['entropy_bits']:>{col_w}.3f}" for _, s in summaries)
    print(f"{'Chain entropy (bits)':24}{ent_parts}")
    mix_parts = "".join(f"{s['mixing_time_steps']:>{col_w}}" for _, s in summaries)
    print(f"{'Mixing time (steps)':24}{mix_parts}")
    dom_parts = "".join(f"{str(s['dominant_chord']):>{col_w}}" for _, s in summaries)
    print(f"{'Dominant chord':24}{dom_parts}")
    pct_parts = "".join(f"{s['dominant_chord_pct']:>{col_w - 1}.1f}%" for _, s in summaries)
    print(f"{'Dominant chord (%)':24}{pct_parts}")

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

# load the models for a given set of orders
def _load_models_for_orders(load_model: Path, orders: Sequence[int]) -> tuple[dict[int, HierarchicalMarkovModel], list[ChordToken]]:
    models: dict[int, HierarchicalMarkovModel] = {}
    index_to_chord: list[ChordToken] | None = None
    for order in orders:
        order_dir = load_model / f"order{order}"
        if not order_dir.is_dir():
            print(
                f"error: multi-order load expects {order_dir}",
                file=sys.stderr,
            )
            raise FileNotFoundError(str(order_dir))
        logger.info("Loading model from %s", order_dir)
        model, loaded_index = load_model_bundle(order_dir)
        if model.melody.order != order:
            raise ValueError(
                f"loaded model at {order_dir} has melody order {model.melody.order}, expected {order}"
            )
        models[order] = model
        if index_to_chord is None:
            index_to_chord = loaded_index
    assert index_to_chord is not None
    return models, index_to_chord

# run multi-order generation (generalizes former run_compare)
def run_orders(
    orders: Sequence[int],
    *,
    style: str,
    n_chords: int,
    notes_per_chord: int,
    tempo_bpm: int,
    training_paths: Sequence[Path],
    source_piece: Path,
    load_model: Path | None,
    save_model: Path | None,
    play: bool = False,
) -> int:
    models: dict[int, HierarchicalMarkovModel] | None = None
    index_to_chord: list[ChordToken] | None = None
    trained_fresh = False

    if load_model is not None:
        try:
            models, index_to_chord = _load_models_for_orders(load_model, orders)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        logger.info("Training order(s) %s on %s file(s)", orders, len(training_paths))
        trained_fresh = True

    multi = Composer.compose_orders(
        style,
        n_chords,
        orders,
        notes_per_chord=notes_per_chord,
        tempo_bpm=tempo_bpm,
        corpus_paths=training_paths if models is None else None,
        models=models,
        index_to_chord=index_to_chord,
    )

    if save_model is not None and trained_fresh:
        chord_sequences = _collect_chord_sequences(training_paths)
        chord_to_index = build_chord_vocabulary(chord_sequences)
        for order, _, model in multi.results:
            save_dir = save_model / f"order{order}"
            logger.info("Saving model to %s", save_dir)
            save_model_bundle(model, save_dir, chord_to_index)

    midi_paths: dict[int, Path] = {}
    for order, result, _ in multi.results:
        midi_paths[order] = render_composition(result, f"{style}_order{order}")

    print_orders_analysis(multi, style=style)

    if play:
        original_midi, original_duration = _export_original_midi(source_piece, f"{style}_original")
        tracks: list[tuple[str, Path, float]] = [
            ("Original", original_midi, original_duration),
        ]
        for order, result, _ in multi.results:
            tracks.append(
                (
                    f"{style} order {order}",
                    midi_paths[order],
                    _composition_duration_seconds(result),
                )
            )
        _play_sequence(tracks)
    return 0

# run a comparison of order-1 and order-2
def run_compare(
    *,
    style: str,
    n_chords: int,
    notes_per_chord: int,
    tempo_bpm: int,
    training_paths: Sequence[Path],
    source_piece: Path,
    load_model: Path | None,
    save_model: Path | None,
    play: bool = False,
) -> int:
    return run_orders(
        [1, 2],
        style=style,
        n_chords=n_chords,
        notes_per_chord=notes_per_chord,
        tempo_bpm=tempo_bpm,
        training_paths=training_paths,
        source_piece=source_piece,
        load_model=load_model,
        save_model=save_model,
        play=play,
    )

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
    if args.duration is not None and args.duration <= 0:
        print("error: --duration must be positive", file=sys.stderr)
        return 2

    effective_orders = _resolve_effective_orders(args)
    if not _confirm_order3(effective_orders, args.yes):
        return 0

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    corpus_paths = load_corpus(args.style)
    source_piece = _resolve_source(args.source, args.style, corpus_paths)
    training_paths = [source_piece] if args.single_source else corpus_paths

    n_chords = args.n_chords
    if args.duration is not None:
        if args.n_chords != DEFAULT_N_CHORDS:
            logger.info("--duration overrides --n-chords for generated pieces.")
        n_chords = _seconds_to_n_chords(args.duration, args.tempo)

    multi_order_run = len(effective_orders) > 1 or args.compare or args.orders is not None
    if multi_order_run:
        return run_orders(
            effective_orders,
            style=args.style,
            n_chords=n_chords,
            notes_per_chord=args.notes_per_chord,
            tempo_bpm=args.tempo,
            training_paths=training_paths,
            source_piece=source_piece,
            load_model=args.load_model,
            save_model=args.save_model,
            play=args.play,
        )

    order = effective_orders[0]

    if args.load_model is not None:
        if not args.load_model.is_dir():
            print(f"error: model directory not found: {args.load_model}", file=sys.stderr)
            return 1
        logger.info("Loading model from %s", args.load_model)
        model, index_to_chord = load_model_bundle(args.load_model)
        chord_to_index = None
    else:
        logger.info("Training model (order=%s) on %s file(s)", order, len(training_paths))
        model, index_to_chord, chord_to_index = train_model(training_paths, order)
        if args.save_model is not None:
            logger.info("Saving model to %s", args.save_model)
            save_model_bundle(model, args.save_model, chord_to_index)

    if model.melody.order != order:
        print(
            f"error: loaded melody order is {model.melody.order}, but run requested order {order}",
            file=sys.stderr,
        )
        return 1

    midi_path, result = run_generation(
        model,
        index_to_chord,
        style=args.style,
        order=order,
        n_chords=n_chords,
        notes_per_chord=args.notes_per_chord,
        tempo_bpm=args.tempo,
        output_stem=f"{args.style}_order{order}",
    )
    if args.play:
        original_midi, original_duration = _export_original_midi(source_piece, f"{args.style}_original")
        _play_sequence(
            [
                ("Original", original_midi, original_duration),
                (
                    f"{args.style} order {order}",
                    midi_path,
                    _composition_duration_seconds(result),
                ),
            ]
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())