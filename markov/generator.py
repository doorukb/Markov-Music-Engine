from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import numpy as np
from config import (
    DEFAULT_N_CHORDS,
    DEFAULT_ORDER,
    DEFAULT_TEMPO_BPM,
    SUPPORTED_ORDERS,
    SUPPORTED_STYLES,
)
from markov.encoder import ChordIndex
from markov.harmony import UNK_CHORD_INDEX
from markov.matrix import Composition, HierarchicalMarkovModel

__all__ = ["Composer", "CompositionResult"]

# generated composition and run metadata (no MIDI/audio)
@dataclass(frozen=True)
class CompositionResult:
    composition: Composition
    style: str
    order: int
    tempo_bpm: int
    metadata: dict[str, Any]

# high-level API for sampling compositions from a trained hierarchical model
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
        self._validate_params(style, n_chords, order, notes_per_chord, tempo_bpm)

        start_chord = self._sample_start_chord()
        composition = self.model.generate(
            n_chords=n_chords,
            start_chord=start_chord,
            order=order,
            notes_per_chord=notes_per_chord,
        )

        harmony = self.model.harmony
        melody = self.model.melody
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

    # validate generation parameters against config before generation
    def _validate_params(
        self,
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
        if self.model.melody.order != order:
            raise ValueError(f"model melody chain was trained with order={self.model.melody.order}, but compose() requested order={order}")
        if notes_per_chord < 1:
            raise ValueError(f"notes_per_chord must be at least 1; got {notes_per_chord}")
        if tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be positive; got {tempo_bpm}")

        if self.model.harmony.transition_matrix is None:
            raise RuntimeError("Cannot compose: harmony layer is not trained and normalized.")
        if self.model.melody.transition_matrices is None:
            raise RuntimeError("Cannot compose: melody layer is not trained and normalized.")

    def _sample_start_chord(self) -> ChordIndex:
        matrix = self.model.harmony.transition_matrix
        assert matrix is not None

        active = np.flatnonzero(matrix.sum(axis=1) > 0)
        active = active[active != UNK_CHORD_INDEX]
        if len(active) == 0:
            raise RuntimeError("Cannot compose: no valid harmony states with outgoing transitions.")
        return int(np.random.choice(active))