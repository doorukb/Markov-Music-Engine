"""
Level 2 : note-level Markov chains conditioned on chord state
- Maintain one transition matrix per chord state
- Support order-1 (current note to next note) chains
- Support order-2 (prev note, current note to next note) chains
- MLE training from aligned (chord, note) sequence pairs
- Sample next note given (current note, current chord) context
- Serialize and deserialize all per-chord matrices
"""
from __future__ import annotations
import logging
from typing import Dict, Sequence, Tuple, Union
import numpy as np
from config import SUPPORTED_ORDERS
from markov.encoder import ChordIndex, NOTE_VOCAB_SIZE, NoteIndex
from markov.harmony import UNK_CHORD_INDEX

logger = logging.getLogger(__name__)
NoteState = Union[NoteIndex, Tuple[NoteIndex, NoteIndex]]
__all__ = ["MelodyChain", "NoteState"]

# note Markov chains conditioned on chord, with one matrix per chord context
class MelodyChain:
    def __init__(self, order: int = 1, note_vocab_size: int = NOTE_VOCAB_SIZE) -> None:
        if order not in SUPPORTED_ORDERS:
            raise ValueError(f"order must be one of {SUPPORTED_ORDERS}; got order={order}")
        self.order = order
        self.note_vocab_size = note_vocab_size
        self._state_rows = (
            if order == 1:
                note_vocab_size
            else:
                note_vocab_size * note_vocab_size
        )
        self.counts: Dict[ChordIndex, np.ndarray] = {}
        self.transition_matrices: Dict[ChordIndex, np.ndarray] | None = None

    def _matrix_for_chord(self, chord_index: ChordIndex) -> np.ndarray:
        if chord_index not in self.counts:
            self.counts[chord_index] = np.zeros((self._state_rows, self.note_vocab_size), dtype=np.int64)
        return self.counts[chord_index]

    def _validate_note(self, note: NoteIndex, label: str) -> None:
        if not 0 <= note < self.note_vocab_size:
            raise ValueError(f"{label} {note} out of range [0, {self.note_vocab_size})")

    def _state_row_index(self, state: NoteState) -> int:
        if self.order == 1:
            if not isinstance(state, int):
                raise TypeError("order-1 sample state must be an int (current_note_index)")
            self._validate_note(state, "current_note_index")
            return state

        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("order-2 sample state must be a (prev_note_index, current_note_index) tuple")
        prev_note, current_note = state
        self._validate_note(prev_note, "prev_note_index")
        self._validate_note(current_note, "current_note_index")
        return prev_note * self.note_vocab_size + current_note

    # accumulate note transition counts per chord context (raw counts only)
    def train(self, chord_sequence: Sequence[ChordIndex], note_sequence: Sequence[NoteIndex]) -> None:
        if len(chord_sequence) != len(note_sequence):
            raise ValueError("chord_sequence and note_sequence must have the same length " f"({len(chord_sequence)} != {len(note_sequence)})")

        min_notes = 2 if self.order == 1 else 3
        if len(note_sequence) < min_notes:
            logger.warning("train() called with fewer than %d notes; nothing to accumulate", min_notes)
            return

        if self.order == 1:
            self._train_order1(chord_sequence, note_sequence)
        else:
            self._train_order2(chord_sequence, note_sequence)

    def _train_order1(self, chord_sequence: Sequence[ChordIndex], note_sequence: Sequence[NoteIndex]) -> None:
        for i in range(len(note_sequence) - 1):
            chord = chord_sequence[i]
            if chord == UNK_CHORD_INDEX:
                continue

            prev_note = note_sequence[i]
            next_note = note_sequence[i + 1]
            self._validate_note(prev_note, "note index")
            self._validate_note(next_note, "note index")

            matrix = self._matrix_for_chord(chord)
            matrix[prev_note, next_note] += 1

    def _train_order2(self, chord_sequence: Sequence[ChordIndex], note_sequence: Sequence[NoteIndex]) -> None:
        for i in range(1, len(note_sequence) - 1):
            chord = chord_sequence[i]
            if chord == UNK_CHORD_INDEX:
                continue

            prev_note = note_sequence[i - 1]
            current_note = note_sequence[i]
            next_note = note_sequence[i + 1]
            self._validate_note(prev_note, "note index")
            self._validate_note(current_note, "note index")
            self._validate_note(next_note, "note index")

            state_row = prev_note * self.note_vocab_size + current_note
            matrix = self._matrix_for_chord(chord)
            matrix[state_row, next_note] += 1

    def normalize(self) -> None:
        if not self.counts:
            raise RuntimeError("Cannot normalize: train MelodyChain before normalizing.")
        if sum(matrix.sum() for matrix in self.counts.values()) == 0:
            raise RuntimeError("Cannot normalize: no note transitions were accumulated during training.")

        self.transition_matrices = {}
        for chord_index, counts in self.counts.items():
            row_sums = counts.sum(axis=1, keepdims=True, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.transition_matrices[chord_index] = np.divide(
                    counts,
                    row_sums,
                    where=row_sums > 0,
                    out=np.zeros(counts.shape, dtype=np.float64),
                )

    # sample the next note index given chord context and Markov state
    # order 1: state is current note index
    # order 2: state is (prev note index, current note index)
    def sample(self, chord_index: ChordIndex, state: NoteState) -> NoteIndex:
        if self.transition_matrices is None:
            raise RuntimeError("Cannot sample : transition matrices are not normalized. Call normalize() on MelodyChain after training.")
        if chord_index == UNK_CHORD_INDEX:
            raise ValueError("Cannot sample: chord context is UNK_CHORD_INDEX")

        matrix = self.transition_matrices.get(chord_index)
        if matrix is None:
            raise RuntimeError(f"Cannot sample: no transition matrix for chord index {chord_index}")

        row_index = self._state_row_index(state)
        row = matrix[row_index]
        if row.sum() <= 0:
            raise RuntimeError(
                f"Cannot sample: state {state!r} under chord {chord_index} has no "
                "outgoing transitions (row sum is zero). Apply smoothing before "
                "sampling if this state should be reachable during generation."
            )

        indices = np.arange(self.note_vocab_size)
        return int(np.random.choice(indices, p=row))