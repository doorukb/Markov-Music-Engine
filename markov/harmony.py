"""
Level 1 : chord-level Markov chain.
- Accumulate bigram counts over chord sequences (MLE)
- Aggregate counts across multiple MIDI files
- Normalize counts to row-stochastic transition matrix
- Sample next chord given current chord state
- Serialize and deserialize the transition matrix
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np
from markov.encoder import ChordIndex

UNK_CHORD_INDEX: ChordIndex = 0

__all__ = ["ChordChain", "UNK_CHORD_INDEX"]

# order-1 chord Markov chain trained from raw bigram count matrices
class ChordChain:
    def __init__(self, vocab_size: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.counts: np.ndarray | None = None

    # accumulate bigram counts from encoded chord sequences
    def train(self, chord_sequences: Sequence[Sequence[ChordIndex]]) -> None:
        if not chord_sequences:
            return

        inferred_size = max(
            (max(seq) for seq in chord_sequences if seq),
            default=UNK_CHORD_INDEX,
        ) + 1
        size = self.vocab_size if self.vocab_size is not None else inferred_size
        if self.vocab_size is not None and inferred_size > self.vocab_size:
            raise ValueError(
                f"sequence index {inferred_size - 1} exceeds vocab_size {self.vocab_size}"
            )

        if self.counts is None:
            self.counts = np.zeros((size, size), dtype=np.int64)
            self.vocab_size = size
        elif self.counts.shape != (size, size):
            raise ValueError(
                f"counts shape {self.counts.shape} does not match vocab_size {size}"
            )

        for seq in chord_sequences:
            for prev, curr in zip(seq, seq[1:]):
                if prev == UNK_CHORD_INDEX or curr == UNK_CHORD_INDEX:
                    continue
                if not (0 <= prev < size and 0 <= curr < size):
                    raise ValueError(
                        f"chord index out of range [0, {size}): ({prev}, {curr})"
                    )
                self.counts[prev, curr] += 1