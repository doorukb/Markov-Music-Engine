from __future__ import annotations
import json
import logging
from tqdm import tqdm
from markov.encoder import encode_notes
from markov.parser import ParseError
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple
import numpy as np
from config import DEFAULT_N_CHORDS, SMOOTHING_ALPHA, SUPPORTED_ORDERS
from markov.encoder import ChordIndex, NoteIndex
from markov.harmony import ChordChain, ParseFn, PathLike, UNK_CHORD_INDEX
from markov.melody import EncodeChordsFn, MelodyChain, StateUnseen
from markov.parser import ChordToken, NoteToken

Composition = List[Tuple[ChordIndex, List[NoteIndex]]]

_CHORD_CHAIN_FILE = "chord_chain.npz"
_MELODY_CHAIN_FILE = "melody_chain.npz"
_MODEL_META_FILE = "model_meta.json"

__all__ = ["HierarchicalMarkovModel", "Composition"]
logger = logging.getLogger(__name__)

# hierarchical Markov model to compose the harmony and melody layers
class HierarchicalMarkovModel:
    def __init__(self, harmony: ChordChain, melody: MelodyChain) -> None:
        self.harmony = harmony
        self.melody = melody

    # train both layers in a single pass, then normalize both
    def train(self,
        paths: Sequence[PathLike],
        parse_fn: ParseFn,
        encode_fn: EncodeChordsFn,
        chord_to_index: Mapping[ChordToken, ChordIndex],
        note_to_index: Mapping[NoteToken, NoteIndex],
    ) -> None:
        for path in tqdm(paths, desc="training the model"):
            try:
                chord_sequence, note_sequence = parse_fn(Path(path))
                chord_ids = encode_fn(chord_sequence, chord_to_index)
                note_ids = encode_notes(note_sequence, note_to_index)
                self.harmony.train([chord_ids])
                self.melody.train(chord_ids, note_ids)
            except ParseError as exc:
                logger.warning("skipping %s: %s", path, exc)

        if self.harmony.counts is None or self.harmony.counts.sum() == 0:
            raise ValueError("no chord transitions accumulated")
        if not self.melody.counts:
            raise ValueError("no note transitions accumulated")
        self.harmony.normalize(alpha=SMOOTHING_ALPHA)
        self.melody.normalize(alpha=SMOOTHING_ALPHA)

    # sample a chord progression and melody notes per chord
    # return [(chord_index, [note_index, ...]), ...] of length n_chords
    def generate(self, n_chords: int, start_chord: ChordIndex, order: int, notes_per_chord: int = DEFAULT_N_CHORDS) -> Composition:
        if self.harmony.transition_matrix is None or self.melody.transition_matrices is None:
            raise RuntimeError("Cannot generate: model has not been trained and normalized.")
        if n_chords < 1:
            raise ValueError(f"n_chords must be at least 1; got {n_chords}")
        if notes_per_chord < 1:
            raise ValueError(f"notes_per_chord must be at least 1; got {notes_per_chord}")
        if order not in SUPPORTED_ORDERS:
            raise ValueError(f"order must be one of {SUPPORTED_ORDERS}; got {order}")
        if self.melody.order != order:
            raise ValueError(f"MelodyChain was trained with order={self.melody.order}, but generate() requested order={order}.")
        if start_chord == UNK_CHORD_INDEX:
            raise ValueError("start_chord cannot be UNK_CHORD_INDEX.")

        progression: Composition = []
        current_chord = start_chord

        for step in range(n_chords):
            notes = self._sample_notes_for_chord_with_fallback(current_chord, order, notes_per_chord)
            progression.append((current_chord, notes))
            if step < n_chords - 1:
                current_chord = self.harmony.sample(current_chord)
        return progression

    # get the chords with melody data
    def _chords_with_melody_data(self) -> List[ChordIndex]:
        available: List[ChordIndex] = []
        for chord_index in self.melody.counts:
            if chord_index == UNK_CHORD_INDEX:
                continue
            counts = self.melody.counts[chord_index]
            if self.melody.order == 3:
                if any(row.sum() > 0 for row in counts.values()):
                    available.append(chord_index)
            elif counts.sum() > 0:
                available.append(chord_index)
        return available

    def _sample_notes_for_chord_with_fallback(
        self, chord_index: ChordIndex, order: int, notes_per_chord: int
    ) -> List[NoteIndex]:
        try:
            return self._sample_notes_for_chord(chord_index, order, notes_per_chord)
        except RuntimeError:
            available = self._chords_with_melody_data()
            if not available:
                raise
            fallback = int(np.random.choice(available))
            return self._sample_notes_for_chord(fallback, order, notes_per_chord)

    # sample notes for a given chord
    # order 1: sample current note only
    # order 2: sample prev and current notes
    # order 3: sample rolling 3-tuple with backoff on unseen states
    def _sample_notes_for_chord(self, chord_index: ChordIndex, order: int, notes_per_chord: int) -> List[NoteIndex]:
        if order == 1:
            return self._sample_notes_order1(chord_index, notes_per_chord)
        if order == 2:
            return self._sample_notes_order2(chord_index, notes_per_chord)
        return self._sample_notes_order3(chord_index, notes_per_chord)

    # returns an active row index of the chord's count matrix
    # order-1: the row is a note index; order-2: the row is an encoded (prev, curr) state
    def _seed_active_row(self, chord_index: ChordIndex) -> int:
        counts = self.melody.counts.get(chord_index)
        if counts is None:
            raise RuntimeError(f"Cannot generate notes: chord index {chord_index} was not seen during melody training.")
        if self.melody.order == 3:
            active_rows = [s for s, row in counts.items() if row.sum() > 0]
            if not active_rows:
                raise RuntimeError(f"Cannot generate notes: no melody transitions for chord index {chord_index}.")
            return int(np.random.choice(active_rows))
        active_rows = np.flatnonzero(counts.sum(axis=1) > 0)
        if len(active_rows) == 0:
            raise RuntimeError(f"Cannot generate notes: no melody transitions for chord index {chord_index}.")
        return int(np.random.choice(active_rows))

    # get the active state for a given chord and order
    def _seed_active_state(self, chord_index: ChordIndex, order: int) -> int:
        if order in (1, 2):
            return self._seed_active_row(chord_index)
        return self._seed_active_row(chord_index)

    # decode a state row into a tuple of note indices
    # order 1: the row is a note index
    # order 2: the row is an encoded (prev, current) state
    # order 3: the row is an encoded (n2, n1, n0) state
    def _decode_state_row(self, state_row: int, order: int) -> tuple[int, ...]:
        vocab = self.melody.note_vocab_size
        if order == 1:
            return (state_row,)
        if order == 2:
            prev_note, current = divmod(state_row, vocab)
            return (prev_note, current)
        n2, rem = divmod(state_row, vocab * vocab)
        n1, n0 = divmod(rem, vocab)
        return (n2, n1, n0)

    # sample notes for a given chord and order 1
    # return a list of note indices of length notes_per_chord
    def _sample_notes_order1(self, chord_index: ChordIndex, notes_per_chord: int) -> List[NoteIndex]:
        notes: List[NoteIndex] = [self._seed_active_row(chord_index)]
        current = notes[0]
        for _ in range(notes_per_chord - 1):
            current = self.melody.sample(chord_index, current)
            notes.append(current)
        return notes

    # sample notes for a given chord and order 2
    # return a list of note indices of length notes_per_chord
    def _sample_notes_order2(self, chord_index: ChordIndex, notes_per_chord: int) -> List[NoteIndex]:
        vocab = self.melody.note_vocab_size
        # order-2 rows are encoded states: prev_note * vocab + current_note
        state_row = self._seed_active_row(chord_index)
        prev_note, current = divmod(state_row, vocab)

        notes: List[NoteIndex] = [prev_note, current][:notes_per_chord]
        for _ in range(notes_per_chord - 2):
            next_note = self.melody.sample(chord_index, (prev_note, current))
            notes.append(next_note)
            prev_note, current = current, next_note
        return notes

    # sample notes for a given chord and order 3 (sparse, with backoff on unseen states)
    def _sample_notes_order3(self, chord_index: ChordIndex, notes_per_chord: int) -> List[NoteIndex]:
        state_row = self._seed_active_state(chord_index, order=3)
        n2, n1, n0 = self._decode_state_row(state_row, order=3)
        notes: List[NoteIndex] = [n2, n1, n0][:notes_per_chord]

        for _ in range(notes_per_chord - 3):
            try:
                next_note = self.melody.sample(chord_index, (n2, n1, n0))
            except StateUnseen:
                state_row = self._seed_active_state(chord_index, order=3)
                n2, n1, n0 = self._decode_state_row(state_row, order=3)
                next_note = self.melody.sample(chord_index, (n2, n1, n0))
            notes.append(next_note)
            n2, n1, n0 = n1, n0, next_note
        return notes

    # save the harmony, melody, and metadata under the given directory
    def save(self, directory: PathLike) -> None:
        if self.harmony.transition_matrix is None or self.melody.transition_matrices is None:
            raise RuntimeError("Cannot save: model has not been trained and normalized.")

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.harmony.save(out_dir / _CHORD_CHAIN_FILE)
        self.melody.save(out_dir / _MELODY_CHAIN_FILE)

        meta = {
            "order": self.melody.order,
            "chord_vocab_size": self.harmony.vocab_size,
            "note_vocab_size": self.melody.note_vocab_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / _MODEL_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # load a model saved with save()
    @classmethod
    def load(cls, directory: PathLike) -> HierarchicalMarkovModel:
        in_dir = Path(directory)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {in_dir}")

        chord_path = in_dir / _CHORD_CHAIN_FILE
        melody_path = in_dir / _MELODY_CHAIN_FILE
        meta_path = in_dir / _MODEL_META_FILE

        missing = [
            path.name
            for path in (chord_path, melody_path, meta_path)
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(f"Cannot load model from {in_dir}: missing file(s): {', '.join(missing)}")

        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        harmony = ChordChain.load(chord_path)
        melody = MelodyChain.load(melody_path)

        if harmony.vocab_size != meta.get("chord_vocab_size"):
            raise ValueError(f"chord_vocab_size mismatch: metadata={meta.get('chord_vocab_size')}, loaded ChordChain={harmony.vocab_size}")
        if melody.order != meta.get("order"):
            raise ValueError(f"order mismatch: metadata={meta.get('order')}, loaded MelodyChain={melody.order}")
        if melody.note_vocab_size != meta.get("note_vocab_size"):
            raise ValueError(f"note_vocab_size mismatch: metadata={meta.get('note_vocab_size')}, loaded MelodyChain={melody.note_vocab_size}")
        # return a new HierarchicalMarkovModel with the loaded harmony and melody chains
        return cls(harmony=harmony, melody=melody)