"""
Level 1 : chord-level Markov chain.
- Accumulate bigram counts over chord sequences (MLE)
- Aggregate counts across multiple MIDI files
- Normalize counts to row-stochastic transition matrix
- Sample next chord given current chord state
- Serialize and deserialize the transition matrix
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence, Union
import numpy as np
from tqdm import tqdm
from markov.encoder import ChordIndex, ChordToken

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

UNK_CHORD_INDEX: ChordIndex = 0

__all__ = ["ChordChain", "UNK_CHORD_INDEX"]

# order-1 chord Markov chain trained from raw bigram count matrices
class ChordChain:
    def __init__(self, vocab_size: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.counts: np.ndarray | None = None
        self.transition_matrix: np.ndarray | None = None

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

    # parse and encode each MIDI file, accumulating bigram counts (no normalization)
    def train_corpus(self,
        paths: Sequence[PathLike],
        parser: ModuleType,
        encoder: ModuleType,
    ) -> None:
        from markov.parser import ParseError

        if not paths:
            raise ValueError(
                "Cannot train on an empty corpus: no MIDI file paths provided."
            )

        chord_to_index: Mapping[ChordToken, ChordIndex] | None = getattr(
            encoder, "chord_to_index", None
        )
        if chord_to_index is None:
            raise ValueError(
                "encoder.chord_to_index must be set before train_corpus; "
                "build it with encoder.build_chord_vocabulary()."
            )

        if self.vocab_size is None:
            self.vocab_size = len(chord_to_index)
        elif self.vocab_size != len(chord_to_index):
            raise ValueError(
                f"ChordChain vocab_size {self.vocab_size} does not match "
                f"encoder.chord_to_index size {len(chord_to_index)}"
            )

        for path in tqdm(paths, desc="Training chord corpus"):
            try:
                chord_sequence, _ = parser.parse_midi(Path(path))
                encoded = encoder.encode_chords(chord_sequence, chord_to_index)
                self.train([encoded])
            except ParseError as exc:
                logger.warning("Skipping %s: %s", path, exc)

        if self.counts is None or self.counts.sum() == 0:
            raise ValueError(
                "Cannot train on an empty corpus: no chord transitions were accumulated."
            )

    # convert raw counts to a row-stochastic transition matrix
    def normalize(self) -> None:
        if self.counts is None:
            raise RuntimeError("Cannot normalize: train ChordChain before normalizing.")
        if self.counts.sum() == 0:
            raise RuntimeError(
                "Cannot normalize: no chord transitions were accumulated during training."
            )
        row_sums = self.counts.sum(axis=1, keepdims=True, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.transition_matrix = np.divide(
                self.counts,
                row_sums,
                where=row_sums > 0,
                out=np.zeros(self.counts.shape, dtype=np.float64),
            )

    # sample the next chord index from the row for current_chord_index
    def sample(self, current_chord_index: ChordIndex) -> ChordIndex:
        if self.transition_matrix is None:
            raise RuntimeError(
                "Cannot sample: transition matrix is not normalized. "
                "Call normalize() on ChordChain after training."
            )
        if self.vocab_size is None:
            raise RuntimeError("Cannot sample: ChordChain has no vocabulary size.")

        if not 0 <= current_chord_index < self.vocab_size:
            raise ValueError(
                f"current_chord_index {current_chord_index} out of range "
                f"[0, {self.vocab_size})"
            )

        row = self.transition_matrix[current_chord_index]
        if row.sum() <= 0:
            raise RuntimeError(
                f"Cannot sample: chord index {current_chord_index} has no "
                "outgoing transitions (row sum is zero)."
            )

        indices = np.arange(self.vocab_size)
        return int(np.random.choice(indices, p=row))