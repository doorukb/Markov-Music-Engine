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

__all__ = ["Composer", "CompositionResult", "ComparisonResult"]

# generated composition and run metadata (no MIDI/audio)
@dataclass(frozen=True)
class CompositionResult:
    composition: Composition
    style: str
    order: int
    tempo_bpm: int
    metadata: dict[str, Any]

# comparison result
@dataclass(frozen=True)
class ComparisonResult:
    order1: CompositionResult
    order2: CompositionResult
    # both trained models are retained in memory simultaneously.
    # callers (e.g. a Streamlit dashboard) should avoid holding multiple
    # comparisonResult objects in session state at the same time.
    models: tuple[HierarchicalMarkovModel, HierarchicalMarkovModel]
    index_to_chord: tuple[ChordToken, ...]

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

    # compare two models and sample compositions from them
    # unless models are given, train order-1 and order-2 models on the corpus
    # sample both compositions and return them with the trained models
    @classmethod
    def compare(
        cls,
        style: str,
        n_chords: int,
        notes_per_chord: int = DEFAULT_N_CHORDS,
        tempo_bpm: int = DEFAULT_TEMPO_BPM,
        *,
        corpus_paths: Sequence[Path] | None = None,
        models: tuple[HierarchicalMarkovModel, HierarchicalMarkovModel] | None = None,
        index_to_chord: Sequence[ChordToken] | None = None,
    ) -> ComparisonResult:
        _validate_compose_params(style, n_chords, 1, notes_per_chord, tempo_bpm)

        if models is None:
            paths = list(corpus_paths) if corpus_paths is not None else load_corpus(style)
            chord_sequences = collect_chord_sequences(paths)
            if not chord_sequences:
                raise RuntimeError("No chord sequences could be parsed from the corpus.")

            chord_to_index = build_chord_vocabulary(chord_sequences)
            resolved_index_to_chord = tuple(chord_vocabulary_inverse(chord_to_index))
            note_to_index = build_note_vocabulary()

            model_order1 = _train_model(paths, order=1, chord_to_index=chord_to_index, note_to_index=note_to_index)
            model_order2 = _train_model(paths, order=2, chord_to_index=chord_to_index, note_to_index=note_to_index)
            models = (model_order1, model_order2)
        else:
            model_order1, model_order2 = models
            if model_order1.melody.order != 1 or model_order2.melody.order != 2:
                raise ValueError("compare() expects models with melody orders 1 and 2 respectively.")
            if index_to_chord is None:
                assert model_order1.harmony.vocab_size is not None
                resolved_index_to_chord = tuple(
                    f"chord_{i}" for i in range(model_order1.harmony.vocab_size)
                )
            else:
                resolved_index_to_chord = tuple(index_to_chord)

        _validate_model_ready(model_order1, order=1)
        _validate_model_ready(model_order2, order=2)

        order1_result = _compose_from_model(
            model_order1,
            style=style,
            n_chords=n_chords,
            order=1,
            notes_per_chord=notes_per_chord,
            tempo_bpm=tempo_bpm,
        )
        order2_result = _compose_from_model(
            model_order2,
            style=style,
            n_chords=n_chords,
            order=2,
            notes_per_chord=notes_per_chord,
            tempo_bpm=tempo_bpm,
        )
        return ComparisonResult(
            order1=order1_result,
            order2=order2_result,
            models=models,
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
    assert matrix is not None

    active = np.flatnonzero(matrix.sum(axis=1) > 0)
    active = active[active != UNK_CHORD_INDEX]
    if len(active) == 0:
        raise RuntimeError("Cannot compose: no valid harmony states with outgoing transitions.")
    # weight by stationary distribution (stationary_power_iteration) so the
    # starting chord is drawn from the model's long-run equilibrium rather than
    # uniformly, so that improves musical coherence for short compositions.
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