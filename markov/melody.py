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
from typing import Dict, Sequence
import numpy as np
from markov.encoder import ChordIndex, NOTE_VOCAB_SIZE, NoteIndex
from markov.harmony import UNK_CHORD_INDEX

logger = logging.getLogger(__name__)

__all__ = ["MelodyChain"]

# order-1 note Markov chains, one raw count matrix per chord context
class MelodyChain:
    def __init__(self, order: int = 1, note_vocab_size: int = NOTE_VOCAB_SIZE) -> None:
        if order != 1:
            raise ValueError(f"only order=1 is supported; got order={order}")
        self.order = order
        self.note_vocab_size = note_vocab_size
        self.counts: Dict[ChordIndex, np.ndarray] = {}

    def _matrix_for_chord(self, chord_index: ChordIndex) -> np.ndarray:
        if chord_index not in self.counts:
            self.counts[chord_index] = np.zeros(
                (self.note_vocab_size, self.note_vocab_size),
                dtype=np.int64,
            )
        return self.counts[chord_index]

    # accumulate note bigram counts for each chord context (raw counts only)
    def train(self, chord_sequence: Sequence[ChordIndex], note_sequence: Sequence[NoteIndex]) -> None:
        if len(chord_sequence) != len(note_sequence):
            raise ValueError("chord_sequence and note_sequence must have the same length "f"({len(chord_sequence)} != {len(note_sequence)})")
        if len(note_sequence) < 2:
            logger.warning("train() called with fewer than 2 notes; no note bigrams to accumulate.")
            return
        for i in range(len(note_sequence) - 1):
            chord = chord_sequence[i]
            if chord == UNK_CHORD_INDEX:
                continue

            prev_note = note_sequence[i]
            next_note = note_sequence[i + 1]
            if not (0 <= prev_note < self.note_vocab_size):
                raise ValueError(
                    f"note index {prev_note} out of range [0, {self.note_vocab_size})"
                )
            if not (0 <= next_note < self.note_vocab_size):
                raise ValueError(
                    f"note index {next_note} out of range [0, {self.note_vocab_size})"
                )

            matrix = self._matrix_for_chord(chord)
            matrix[prev_note, next_note] += 1