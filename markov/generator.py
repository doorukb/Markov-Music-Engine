from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from config import (
    DEFAULT_N_CHORDS,
    DEFAULT_ORDER,
    DEFAULT_TEMPO_BPM,
    SUPPORTED_ORDERS,
    SUPPORTED_STYLES,
)
from markov.data import collect_chord_sequences, load_corpus
from markov.encoder import (
    ChordIndex,
    ChordToken,
    NoteIndex,
    NoteToken,
    build_chord_vocabulary,
    build_note_vocabulary,
    chord_vocabulary_inverse,
    encode_chords,
)
from markov.harmony import ChordChain, UNK_CHORD_INDEX
from markov.matrix import Composition, HierarchicalMarkovModel
from markov.melody import MelodyChain
from markov.parser import parse_midi

__all__ = ["Composer", "CompositionResult", "MultiOrderResult"]

# generated composition and run metadata (no MIDI/audio)
@dataclass(frozen=True)
class CompositionResult:
    composition: Composition
    style: str
    order: int
    tempo_bpm: int
    metadata: dict[str, Any]

# multi-order result
@dataclass(frozen=True)
class MultiOrderResult:
    results: tuple[tuple[int, CompositionResult, HierarchicalMarkovModel], ...]
    index_to_chord: tuple[ChordToken, ...]

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(order for order, _, _ in self.results)

# train the model on a list of MIDI files
def _train_model(
    paths: Sequence[Path],
    order: int,
    chord_to_index: dict[ChordToken, ChordIndex],
    note_to_index: dict[NoteToken, NoteIndex],
) -> HierarchicalMarkovModel:
    harmony = ChordChain(vocab_size=len(chord_to_index))
    melody = MelodyChain(order=order)
    model = HierarchicalMarkovModel(harmony=harmony, melody=melody)
    model.train(paths, parse_midi, encode_chords, chord_to_index, note_to_index)
    return model

# composer class for sampling compositions from a trained hierarchical model
class Composer:
    def __init__(self, model: HierarchicalMarkovModel) -> None:
        self.model = model

    # sample a chord progression and per-chord melody from the model
    # parameters are validated against config before generation
    # MIDI and audio export are handled by markov.renderer
    def compose(
        self,
        style: str,
        n_chords: int,
        order: int = DEFAULT_ORDER,
        notes_per_chord: int = DEFAULT_N_CHORDS,
        tempo_bpm: int = DEFAULT_TEMPO_BPM,
    ) -> CompositionResult:
        _validate_compose_params(style, n_chords, order, notes_per_chord, tempo_bpm)
        _validate_model_ready(self.model, order)
        return _compose_from_model(
            self.model,
            style=style,
            n_chords=n_chords,
            order=order,
            notes_per_chord=notes_per_chord,
            tempo_bpm=tempo_bpm,
        )

    # compose a composition from a model for a given set of orders
    @classmethod
    def compose_orders(
        cls,
        style: str,
        n_chords: int,
        orders: Sequence[int],
        notes_per_chord: int = DEFAULT_N_CHORDS,
        tempo_bpm: int = DEFAULT_TEMPO_BPM,
        *,
        corpus_paths: Sequence[Path] | None = None,
        models: dict[int, HierarchicalMarkovModel] | None = None,
        index_to_chord: Sequence[ChordToken] | None = None,
    ) -> MultiOrderResult:
        if not orders:
            raise ValueError("compose_orders() requires at least one order.")
        unique_orders = sorted(set(orders))
        for order in unique_orders:
            _validate_compose_params(style, n_chords, order, notes_per_chord, tempo_bpm)

        trained_models: dict[int, HierarchicalMarkovModel]
        if models is None:
            paths = list(corpus_paths) if corpus_paths is not None else load_corpus(style)
            chord_sequences = collect_chord_sequences(paths)
            if not chord_sequences:
                raise RuntimeError("No chord sequences could be parsed from the corpus.")

            chord_to_index = build_chord_vocabulary(chord_sequences)
            resolved_index_to_chord = tuple(chord_vocabulary_inverse(chord_to_index))
            note_to_index = build_note_vocabulary()

            trained_models = {
                order: _train_model(paths, order=order, chord_to_index=chord_to_index, note_to_index=note_to_index)
                for order in unique_orders
            }
        else:
            trained_models = models
            missing = [order for order in unique_orders if order not in trained_models]
            if missing:
                raise ValueError(f"compose_orders() missing pre-trained model(s) for order(s): {missing}")
            for order in unique_orders:
                if trained_models[order].melody.order != order:
                    raise ValueError(f"compose_orders() model for order {order} has melody order {trained_models[order].melody.order}.")
            if index_to_chord is None:
                first_model = trained_models[unique_orders[0]]
                if first_model.harmony.vocab_size is None:
                    raise RuntimeError("Cannot resolve chord labels: harmony layer has no vocabulary size.")
                resolved_index_to_chord = tuple(f"chord_{i}" for i in range(first_model.harmony.vocab_size))
            else:
                resolved_index_to_chord = tuple(index_to_chord)

        results: list[tuple[int, CompositionResult, HierarchicalMarkovModel]] = []
        for order in unique_orders:
            model = trained_models[order]
            _validate_model_ready(model, order=order)
            result = _compose_from_model(
                model,
                style=style,
                n_chords=n_chords,
                order=order,
                notes_per_chord=notes_per_chord,
                tempo_bpm=tempo_bpm,
            )
            results.append((order, result, model))

        return MultiOrderResult(
            results=tuple(results),
            index_to_chord=resolved_index_to_chord,
        )

# validate the composition parameters
def _validate_compose_params(
    style: str,
    n_chords: int,
    order: int,
    notes_per_chord: int,
    tempo_bpm: int,
) -> None:
    if style not in SUPPORTED_STYLES:
        raise ValueError(f"style must be one of {SUPPORTED_STYLES}; got {style!r}")
    if n_chords < 1:
        raise ValueError(f"n_chords must be at least 1; got {n_chords}")
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"order must be one of {SUPPORTED_ORDERS}; got {order}")
    if notes_per_chord < 1:
        raise ValueError(f"notes_per_chord must be at least 1; got {notes_per_chord}")
    if tempo_bpm <= 0:
        raise ValueError(f"tempo_bpm must be positive; got {tempo_bpm}")

# validate the model is ready for composition
def _validate_model_ready(model: HierarchicalMarkovModel, order: int) -> None:
    if model.melody.order != order:
        raise ValueError(f"model melody chain was trained with order={model.melody.order}, but generation requested order={order}")
    if model.harmony.transition_matrix is None:
        raise RuntimeError("Cannot compose: harmony layer is not trained and normalized.")
    if model.melody.transition_matrices is None:
        raise RuntimeError("Cannot compose: melody layer is not trained and normalized.")

# sample a start chord from the model
def _sample_start_chord(model: HierarchicalMarkovModel) -> ChordIndex:
    matrix = model.harmony.transition_matrix
    if matrix is None:
        raise RuntimeError("Cannot sample start chord: harmony layer is not trained and normalized.")

    active = np.flatnonzero(matrix.sum(axis=1) > 0)
    active = active[active != UNK_CHORD_INDEX]
    if len(active) == 0:
        raise RuntimeError("Cannot compose: no valid harmony states with outgoing transitions.")
    # sample uniformly among active harmony states (not weighted by stationary distribution).
    return int(np.random.choice(active))

# compose a composition from the model
def _compose_from_model(
    model: HierarchicalMarkovModel,
    *,
    style: str,
    n_chords: int,
    order: int,
    notes_per_chord: int,
    tempo_bpm: int,
) -> CompositionResult:
    start_chord = _sample_start_chord(model)
    composition = model.generate(
        n_chords=n_chords,
        start_chord=start_chord,
        order=order,
        notes_per_chord=notes_per_chord,
    )

    harmony = model.harmony
    melody = model.melody
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chord_vocab_size": harmony.vocab_size,
        "note_vocab_size": melody.note_vocab_size,
        "n_chords": n_chords,
        "notes_per_chord": notes_per_chord,
        "start_chord_index": start_chord,
    }

    return CompositionResult(
        composition=composition,
        style=style,
        order=order,
        tempo_bpm=tempo_bpm,
        metadata=metadata,
    )