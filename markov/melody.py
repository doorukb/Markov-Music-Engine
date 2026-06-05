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
from markov.smoothing import laplace_smooth

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
NoteState = Union[NoteIndex, Tuple[NoteIndex, NoteIndex], Tuple[NoteIndex, NoteIndex, NoteIndex]]

ParseFn = Callable[[Path], Tuple[list[ChordToken], list[NoteToken]]]
EncodeChordsFn = Callable[
    [Sequence[ChordToken], Mapping[ChordToken, ChordIndex]],
    list[ChordIndex],
]

_COUNTS_KEY = re.compile(r"^counts_(\d+)$")
_TRANSITION_KEY = re.compile(r"^transition_(\d+)$")
_STATES_KEY = re.compile(r"^states_(\d+)$")
__all__ = ["MelodyChain", "NoteState", "ParseFn", "EncodeChordsFn", "StateUnseen"]

# exception raised when an order-3 state has no observed transitions
class StateUnseen(Exception):
    pass

# note Markov chains conditioned on chord (one matrix per chord context)
class MelodyChain:
    def __init__(self, order: int, note_vocab_size: int = NOTE_VOCAB_SIZE) -> None:
        if order not in SUPPORTED_ORDERS:
            raise ValueError(f"MelodyChain only supports order in {SUPPORTED_ORDERS}; got order={order}.")

        self.order = order
        self.note_vocab_size = note_vocab_size
        self._state_rows = (
            note_vocab_size
            if order == 1
            else note_vocab_size * note_vocab_size
            if order == 2
            else 0
        )
        if order == 3:
            self.counts: Dict[ChordIndex, Dict[int, np.ndarray]] = {}
            self.transition_matrices: Dict[ChordIndex, Dict[int, np.ndarray]] | None = None
        else:
            self.counts: Dict[ChordIndex, np.ndarray] = {}
            self.transition_matrices: Dict[ChordIndex, np.ndarray] | None = None

    # sum the total number of counts in the chain
    def _total_count_sum(self) -> int:
        if self.order == 3:
            return sum(row.sum() for chord_counts in self.counts.values() for row in chord_counts.values())
        return sum(m.sum() for m in self.counts.values())

    # get the count matrix for a given chord context (orders 1 & 2 only)
    def _matrix_for_chord(self, chord_index: ChordIndex) -> np.ndarray:
        if chord_index not in self.counts:
            self.counts[chord_index] = np.zeros((self._state_rows, self.note_vocab_size), dtype=np.int64)
        return self.counts[chord_index]

    # get the count row for a given chord context and state row
    def _counts_row_for_state(self, chord_index: ChordIndex, state_row: int) -> np.ndarray:
        chord_counts = self.counts.get(chord_index)
        if chord_counts is None:
            chord_counts = {}
            self.counts[chord_index] = chord_counts
        if state_row not in chord_counts:
            chord_counts[state_row] = np.zeros(self.note_vocab_size, dtype=np.int64)
        return chord_counts[state_row]

    # validate a note index
    def _validate_note(self, note: NoteIndex, label: str) -> None:
        if not 0 <= note < self.note_vocab_size:
            raise ValueError(f"{label} {note} out of range [0, {self.note_vocab_size})")

    # get the index of a given state row
    def _state_row_index(self, state: NoteState) -> int:
        if self.order == 1:
            if isinstance(state, tuple):
                raise ValueError("cannot use order-2/3 state on an order-1 MelodyChain; pass a single current_note_index.")
            if not isinstance(state, int):
                raise TypeError("order-1 sample state must be an int (current_note_index)")
            self._validate_note(state, "current_note_index")
            return state

        if self.order == 2:
            if isinstance(state, int):
                raise ValueError("cannot use order-1 state on an order-2 MelodyChain; pass (prev_note_index, current_note_index).")
            if not isinstance(state, tuple) or len(state) != 2:
                raise TypeError("order-2 sample state must be a (prev_note_index, current_note_index) tuple")

            prev_note, current_note = state
            self._validate_note(prev_note, "prev_note_index")
            self._validate_note(current_note, "current_note_index")
            return prev_note * self.note_vocab_size + current_note

        if not isinstance(state, tuple) or len(state) != 3:
            raise TypeError("order-3 sample state must be a (n2, n1, n0) tuple")
        n2, n1, n0 = state
        self._validate_note(n2, "n2")
        self._validate_note(n1, "n1")
        self._validate_note(n0, "n0")
        v = self.note_vocab_size
        return n2 * v * v + n1 * v + n0

    # accumulate note transition counts per chord context (raw counts only)
    def train(self, chord_sequence: Sequence[ChordIndex], note_sequence: Sequence[NoteIndex]) -> None:
        if len(chord_sequence) != len(note_sequence):
            raise ValueError(f"chord_sequence and note_sequence must have the same length ({len(chord_sequence)} != {len(note_sequence)})")

        min_notes = {1: 2, 2: 3, 3: 4}[self.order]
        if len(note_sequence) < min_notes:
            logger.warning("train() called with fewer than %d notes; nothing to accumulate.", min_notes)
            return

        if self.order == 1:
            self._train_order1(chord_sequence, note_sequence)
        elif self.order == 2:
            self._train_order2(chord_sequence, note_sequence)
        else:
            self._train_order3(chord_sequence, note_sequence)

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

    # train the order-3 chain (sparse per-chord storage)
    def _train_order3(self, chord_sequence: Sequence[ChordIndex], note_sequence: Sequence[NoteIndex]) -> None:
        v = self.note_vocab_size
        for i in range(2, len(note_sequence) - 1):
            chord = chord_sequence[i]
            if chord == UNK_CHORD_INDEX:
                continue

            n2 = note_sequence[i - 2]
            n1 = note_sequence[i - 1]
            n0 = note_sequence[i]
            next_note = note_sequence[i + 1]
            self._validate_note(n2, "note index")
            self._validate_note(n1, "note index")
            self._validate_note(n0, "note index")
            self._validate_note(next_note, "note index")

            state_row = n2 * v * v + n1 * v + n0
            row = self._counts_row_for_state(chord, state_row)
            row[next_note] += 1

    # train the corpus
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

        if not self.counts or self._total_count_sum() == 0:
            raise ValueError("Cannot train on an empty corpus: no note transitions were accumulated.")

    def _normalize_row(self, counts_row: np.ndarray, alpha: float) -> np.ndarray:
        if alpha > 0:
            return laplace_smooth(counts_row.reshape(1, -1), alpha).reshape(-1)
        row_sum = counts_row.sum(dtype=np.float64)
        if row_sum <= 0:
            return np.zeros(self.note_vocab_size, dtype=np.float64)
        return counts_row.astype(np.float64) / row_sum

    # row-normalize every per-chord count matrix into transition_matrices
    def normalize(self, alpha: float = 0.0) -> None:
        if not self.counts:
            raise RuntimeError("Cannot normalize: train MelodyChain before normalizing.")
        if self._total_count_sum() == 0:
            raise RuntimeError("Cannot normalize: no note transitions were accumulated during training.")

        self.transition_matrices = {}
        if self.order == 3:
            for chord_index, state_counts in self.counts.items():
                self.transition_matrices[chord_index] = {
                    state_row: self._normalize_row(row, alpha)
                    for state_row, row in state_counts.items()
                }
            return

        for chord_index, counts in self.counts.items():
            if alpha > 0:
                self.transition_matrices[chord_index] = laplace_smooth(counts, alpha)
            else:
                row_sums = counts.sum(axis=1, keepdims=True, dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    self.transition_matrices[chord_index] = np.divide(
                        counts,
                        row_sums,
                        where=row_sums > 0,
                        out=np.zeros(counts.shape, dtype=np.float64),
                    )

    # sample the next note index given chord context and Markov state
    def sample(self, chord_index: ChordIndex, state: NoteState) -> NoteIndex:
        if self.transition_matrices is None:
            raise RuntimeError("Cannot sample: transition matrices are not normalized. Call normalize() on MelodyChain after training.")
        if chord_index == UNK_CHORD_INDEX:
            raise ValueError("Cannot sample: chord context is UNK_CHORD_INDEX.")

        row_index = self._state_row_index(state)

        if self.order == 3:
            chord_matrix = self.transition_matrices.get(chord_index)
            if chord_matrix is None:
                raise RuntimeError(f"Cannot sample: no transition matrix for chord index {chord_index}.")
            row = chord_matrix.get(row_index)
            if row is None or row.sum() <= 0:
                raise StateUnseen(f"unseen order-3 state {state!r} under chord {chord_index}")
            indices = np.arange(self.note_vocab_size)
            return int(np.random.choice(indices, p=row))

        matrix = self.transition_matrices.get(chord_index)
        if matrix is None:
            raise RuntimeError(f"Cannot sample: no transition matrix for chord index {chord_index}.")

        row = matrix[row_index]
        if row.sum() <= 0:
            raise RuntimeError(f"Cannot sample: state {state!r} under chord {chord_index} has no outgoing transitions (row sum is zero). Apply smoothing before sampling if this state should be reachable during generation.")

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

        if self.order == 3:
            for chord_index, state_counts in self.counts.items():
                states = np.array(sorted(state_counts.keys()), dtype=np.int64)
                counts_stack = np.stack([state_counts[s] for s in states], axis=0)
                arrays[f"states_{chord_index}"] = states
                arrays[f"counts_{chord_index}"] = counts_stack
            if self.transition_matrices is not None:
                for chord_index, state_probs in self.transition_matrices.items():
                    states = np.array(sorted(state_probs.keys()), dtype=np.int64)
                    trans_stack = np.stack([state_probs[s] for s in states], axis=0)
                    arrays[f"transition_{chord_index}"] = trans_stack
                    arrays[f"states_{chord_index}"] = states
        else:
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
            raise FileNotFoundError(f"MelodyChain file not found : {path}")

        with np.load(path, allow_pickle=False) as data:
            order = int(data["order"])
            note_vocab_size = int(data["note_vocab_size"])
            chain = cls(order=order, note_vocab_size=note_vocab_size)

            if order == 3:
                chord_indices: set[int] = set()
                for key in data.files:
                    states_match = _STATES_KEY.match(key)
                    if states_match:
                        chord_indices.add(int(states_match.group(1)))

                for chord_index in chord_indices:
                    states_key = f"states_{chord_index}"
                    counts_key = f"counts_{chord_index}"
                    if states_key not in data.files or counts_key not in data.files:
                        continue
                    states = np.asarray(data[states_key], dtype=np.int64)
                    counts_stack = np.asarray(data[counts_key], dtype=np.int64)
                    chain.counts[chord_index] = {
                        int(s): counts_stack[i] for i, s in enumerate(states)
                    }

                transition_key_pattern = _TRANSITION_KEY
                for key in data.files:
                    transition_match = transition_key_pattern.match(key)
                    if not transition_match:
                        continue
                    chord_index = int(transition_match.group(1))
                    states_key = f"states_{chord_index}"
                    if states_key not in data.files:
                        continue
                    if chain.transition_matrices is None:
                        chain.transition_matrices = {}
                    states = np.asarray(data[states_key], dtype=np.int64)
                    trans_stack = np.asarray(data[key], dtype=np.float64)
                    chain.transition_matrices[chord_index] = {
                        int(s): trans_stack[i] for i, s in enumerate(states)
                    }
            else:
                for key in data.files:
                    counts_match = _COUNTS_KEY.match(key)
                    if counts_match:
                        chord_index = int(counts_match.group(1))
                        chain.counts[chord_index] = np.asarray(data[key], dtype=np.int64)
                        continue
                    transition_match = _TRANSITION_KEY.match(key)
                    if transition_match:
                        if chain.transition_matrices is None:
                            chain.transition_matrices = {}
                        chord_index = int(transition_match.group(1))
                        chain.transition_matrices[chord_index] = np.asarray(data[key], dtype=np.float64)

        if not chain.counts:
            raise ValueError(f"No per-chord count matrices found in {path}")
        return chain