from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Mapping, Sequence, Tuple, Union
import numpy as np
from tqdm import tqdm
from markov.encoder import ChordIndex
from markov.parser import ChordToken, NoteToken

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

ParseFn = Callable[[Path], Tuple[List[ChordToken], List[NoteToken]]]
EncodeFn = Callable[
    [Sequence[ChordToken], Mapping[ChordToken, ChordIndex]],
    List[ChordIndex],
]

UNK_CHORD_INDEX: ChordIndex = 0

__all__ = [
    "ChordChain",
    "UNK_CHORD_INDEX",
    "ParseFn",
    "EncodeFn",
]

# order-1 chord Markov chain trained from raw bigram count matrices
class ChordChain:
    def __init__(self, vocab_size: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.counts: np.ndarray | None = None
        self.transition_matrix: np.ndarray | None = None

    def train(self, chord_sequences: Sequence[Sequence[ChordIndex]]) -> None:
        if not chord_sequences:
            logger.warning("train() called with no chord sequences; nothing to accumulate.")
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
    def train_corpus(
        self,
        paths: Sequence[PathLike],
        parse_fn: ParseFn,
        encode_fn: EncodeFn,
        chord_to_index: Mapping[ChordToken, ChordIndex],
    ) -> None:
        from markov.parser import ParseError

        if not paths:
            raise ValueError(
                "cannot train on an empty corpus: no MIDI file paths provided."
            )

        vocab_len = len(chord_to_index)
        if self.vocab_size is None:
            self.vocab_size = vocab_len
        elif self.vocab_size != vocab_len:
            raise ValueError(
                f"chordChain vocab_size {self.vocab_size} does not match "
                f"chord_to_index size {vocab_len}"
            )

        for path in tqdm(paths, desc="training chord corpus"):
            try:
                chord_sequence, _ = parse_fn(Path(path))
                encoded = encode_fn(chord_sequence, chord_to_index)
                self.train([encoded])
            except ParseError as exc:
                logger.warning("skipping %s: %s", path, exc)

        if self.counts is None or self.counts.sum() == 0:
            raise ValueError(
                "cannot train on an empty corpus: no chord transitions were accumulated."
            )

    def normalize(self) -> None:
        if self.counts is None:
            raise RuntimeError("cannot normalize: train ChordChain before normalizing.")
        if self.counts.sum() == 0:
            raise RuntimeError(
                "cannot normalize: no chord transitions were accumulated during training."
            )
        row_sums = self.counts.sum(axis=1, keepdims=True, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.transition_matrix = np.divide(
                self.counts,
                row_sums,
                where=row_sums > 0,
                out=np.zeros(self.counts.shape, dtype=np.float64),
            )

    def sample(self, current_chord_index: ChordIndex) -> ChordIndex:
        if self.transition_matrix is None:
            raise RuntimeError(
                "cannot sample: transition matrix is not normalized. "
                "call normalize() on ChordChain after training."
            )
        if self.vocab_size is None:
            raise RuntimeError("cannot sample: ChordChain has no vocabulary size.")

        if not 0 <= current_chord_index < self.vocab_size:
            raise ValueError(
                f"current_chord_index {current_chord_index} out of range "
                f"[0, {self.vocab_size})"
            )

        row = self.transition_matrix[current_chord_index]
        if row.sum() <= 0:
            # Rows with no training mass stay zero after normalize(); generation
            # must not reach them once smoothing (markov.smoothing) has been applied.
            raise RuntimeError(
                f"cannot sample: chord index {current_chord_index} has no "
                "outgoing transitions (row sum is zero). Apply smoothing before "
                "sampling if this state should be reachable during generation."
            )

        indices = np.arange(self.vocab_size)
        return int(np.random.choice(indices, p=row))

    # persist counts, optional transition matrix, and vocab size to a ``.npz`` file.
    def save(self, path: PathLike) -> None:
        if self.counts is None or self.vocab_size is None:
            raise RuntimeError("cannot save: chordChain has not been trained.")

        path = Path(path)
        arrays: dict[str, np.ndarray] = {
            "counts": self.counts,
            "vocab_size": np.array(self.vocab_size, dtype=np.int64),
        }
        if self.transition_matrix is not None:
            arrays["transition_matrix"] = self.transition_matrix
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: PathLike) -> ChordChain:
        """Load a ChordChain saved with :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ChordChain file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            vocab_size = int(data["vocab_size"])
            counts = np.asarray(data["counts"], dtype=np.int64)
            transition_matrix = (
                np.asarray(data["transition_matrix"], dtype=np.float64)
                if "transition_matrix" in data.files
                else None
            )

        chain = cls(vocab_size=vocab_size)
        chain.counts = counts
        chain.transition_matrix = transition_matrix
        return chain
