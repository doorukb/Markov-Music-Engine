from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple, Union
import numpy as np
from tqdm import tqdm
from config import SUPPORTED_ORDERS
from markov.encoder import ChordIndex, NOTE_VOCAB_SIZE, NoteIndex, encode_notes
from markov.harmony import UNK_CHORD_INDEX
from markov.parser import ChordToken, NoteToken

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
NoteState = Union[NoteIndex, Tuple[NoteIndex, NoteIndex]]

ParseFn = Callable[[Path], Tuple[list[ChordToken], list[NoteToken]]]
EncodeChordsFn = Callable[
    [Sequence[ChordToken], Mapping[ChordToken, ChordIndex]],
    list[ChordIndex],
]

_COUNTS_KEY = re.compile(r"^counts_(\d+)$")
_TRANSITION_KEY = re.compile(r"^transition_(\d+)$")
__all__ = ["MelodyChain", "NoteState", "ParseFn", "EncodeChordsFn"]

# note Markov chains conditioned on chord (one matrix per chord context)
class MelodyChain:
    # use MelodyChain(order=1) or MelodyChain(order=2) only
    def __init__(self, order: int, note_vocab_size: int = NOTE_VOCAB_SIZE) -> None:
        if order not in (1, 2):
            raise ValueError(
                f"MelodyChain only supports order=1 or order=2; got order={order}. "
                "Use MelodyChain(order=1) or MelodyChain(order=2)."
            )
        self.order = order
        self.note_vocab_size = note_vocab_size
        self._state_rows = note_vocab_size if order == 1 else note_vocab_size * note_vocab_size
        self.counts: Dict[ChordIndex, np.ndarray] = {}
        self.transition_matrices: Dict[ChordIndex, np.ndarray] | None = None

    # get the count matrix for a given chord context
    def _matrix_for_chord(self, chord_index: ChordIndex) -> np.ndarray:
        if chord_index not in self.counts:
            self.counts[chord_index] = np.zeros((self._state_rows, self.note_vocab_size), dtype=np.int64)
        return self.counts[chord_index]

    # validate a note index
    def _validate_note(self, note: NoteIndex, label: str) -> None:
        if not 0 <= note < self.note_vocab_size:
            raise ValueError(f"{label} {note} out of range [0, {self.note_vocab_size})")

    def _state_row_index(self, state: NoteState) -> int:
        if self.order == 1:
            if isinstance(state, tuple):
                raise ValueError(
                    "Cannot use order-2 state (prev_note, current_note) on an "
                    "order-1 MelodyChain; use MelodyChain(order=2) or pass a single "
                    "current_note_index."
                )
            if not isinstance(state, int):
                raise TypeError("order-1 sample state must be an int (current_note_index)")
            self._validate_note(state, "current_note_index")
            return state

        if isinstance(state, int):
            raise ValueError(
                "Cannot use order-1 state (current_note_index only) on an "
                "order-2 MelodyChain; pass (prev_note_index, current_note_index)."
            )
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError(
                "order-2 sample state must be a "
                "(prev_note_index, current_note_index) tuple"
            )
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
            logger.warning("train() called with fewer than %d notes; nothing to accumulate.", min_notes)
            return

        if self.order == 1:
            self._train_order1(chord_sequence, note_sequence)
        else:
            self._train_order2(chord_sequence, note_sequence)

    # train the order-1 chain
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

    # train the order-2 chain
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

    # train the courpus
    # parse and encode each MIDI file; accumulate counts (no normalization)
    def train_corpus(
        self,
        paths: Sequence[PathLike],
        parse_fn: ParseFn,
        encode_fn: EncodeChordsFn,
        chord_to_index: Mapping[ChordToken, ChordIndex],
        note_to_index: Mapping[NoteToken, NoteIndex],
    ) -> None:
        from markov.parser import ParseError
        if not paths:
            raise ValueError("Cannot train on an empty corpus: no MIDI file paths provided.")

        for path in tqdm(paths, desc="Training melody corpus"):
            try:
                chord_sequence, note_sequence = parse_fn(Path(path))
                chord_ids = encode_fn(chord_sequence, chord_to_index)
                note_ids = encode_notes(note_sequence, note_to_index)
                self.train(chord_ids, note_ids)
            except ParseError as exc:
                logger.warning("Skipping %s: %s", path, exc)

        if not self.counts or sum(m.sum() for m in self.counts.values()) == 0:
            raise ValueError("Cannot train on an empty corpus: no note transitions were accumulated.")

    # row-normalize every per-chord count matrix into transition_matrices
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
            raise RuntimeError("Cannot sample: transition matrices are not normalized. Call normalize() on MelodyChain after training.")
        if chord_index == UNK_CHORD_INDEX:
            raise ValueError("Cannot sample: chord context is UNK_CHORD_INDEX.")

        matrix = self.transition_matrices.get(chord_index)
        if matrix is None:
            raise RuntimeError(f"Cannot sample: no transition matrix for chord index {chord_index}.")

        row_index = self._state_row_index(state)
        row = matrix[row_index]
        if row.sum() <= 0:
            raise RuntimeError(
                f"Cannot sample : state {state!r} under chord {chord_index} has no "
                "outgoing transitions (row sum is zero). Apply smoothing before "
                "sampling if this state should be reachable during generation."
            )

        indices = np.arange(self.note_vocab_size)
        return int(np.random.choice(indices, p=row))

    # persist order, vocab size, and per-chord count/transition matrices
    def save(self, path: PathLike) -> None:
        if not self.counts:
            raise RuntimeError("Cannot save: MelodyChain has not been trained.")

        path = Path(path)
        arrays: dict[str, np.ndarray] = {
            "order": np.array(self.order, dtype=np.int64),
            "note_vocab_size": np.array(self.note_vocab_size, dtype=np.int64),
        }
        for chord_index, matrix in self.counts.items():
            arrays[f"counts_{chord_index}"] = matrix
        if self.transition_matrices is not None:
            for chord_index, matrix in self.transition_matrices.items():
                arrays[f"transition_{chord_index}"] = matrix

        np.savez_compressed(path, **arrays)

    # load a MelodyChain saved with save()
    @classmethod
    def load(cls, path: PathLike) -> MelodyChain:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MelodyChain file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            order = int(data["order"])
            note_vocab_size = int(data["note_vocab_size"])
            chain = cls(order=order, note_vocab_size=note_vocab_size)

            for key in data.files:
                counts_match = _COUNTS_KEY.match(key)
                if counts_match:
                    chord_index = int(counts_match.group(1))
                    chain.counts[chord_index] = np.asarray(
                        data[key], dtype=np.int64
                    )
                    continue
                transition_match = _TRANSITION_KEY.match(key)
                if transition_match:
                    if chain.transition_matrices is None:
                        chain.transition_matrices = {}
                    chord_index = int(transition_match.group(1))
                    chain.transition_matrices[chord_index] = np.asarray(
                        data[key], dtype=np.float64
                    )

        if not chain.counts:
            raise ValueError(f"No per-chord count matrices found in {path}")
        return chain