"""
HierarchicalMarkovModel : composes the harmony and melody layers
- Provide a single .train(corpus) entry point across both layers
- Wire ChordChain and MelodyChain together into one model object
- Apply smoothing post-training across all transition matrices
- Expose .save() and .load() for full model persistence
- Act as the single object passed to the generator and analysis modules
"""
from __future__ import annotations
from typing import List, Mapping, Sequence, Tuple
import numpy as np
from config import DEFAULT_N_CHORDS, SUPPORTED_ORDERS
from markov.encoder import ChordIndex, NoteIndex
from markov.harmony import ChordChain, EncodeFn, ParseFn, PathLike, UNK_CHORD_INDEX
from markov.melody import EncodeChordsFn, MelodyChain, NoteState
from markov.parser import ChordToken, NoteToken

Composition = List[Tuple[ChordIndex, List[NoteIndex]]]
__all__ = ["HierarchicalMarkovModel", "Composition"]

# hierarchical Markov model to compose the harmony and melody layers
class HierarchicalMarkovModel:
    def __init__(self, harmony: ChordChain, melody: MelodyChain) -> None:
        self.harmony = harmony
        self.melody = melody

    # train both layers on the corpus, then normalize both
    def train(
        self,
        paths: Sequence[PathLike],
        parse_fn: ParseFn,
        encode_fn: EncodeChordsFn,
        chord_to_index: Mapping[ChordToken, ChordIndex],
        note_to_index: Mapping[NoteToken, NoteIndex],
    ) -> None:
        self.harmony.train_corpus(paths, parse_fn, encode_fn, chord_to_index)
        self.melody.train_corpus(paths, parse_fn, encode_fn, chord_to_index, note_to_index)
        self.harmony.normalize()
        self.melody.normalize()

    # sample a chord progression and melody notes per chord
    # return [(chord_index, [note_index, ...]), ...] of length n_chords
    def generate(self, n_chords: int, start_chord: ChordIndex, order: int, notes_per_chord: int = DEFAULT_N_CHORDS) -> Composition:
        if n_chords < 1:
            raise ValueError(f"n_chords must be at least 1; got {n_chords}")
        if notes_per_chord < 1:
            raise ValueError(f"notes_per_chord must be at least 1; got {notes_per_chord}")
        if order not in SUPPORTED_ORDERS:
            raise ValueError(f"order must be one of {SUPPORTED_ORDERS}; got {order}")
        if self.melody.order != order:
            raise ValueError(
                f"MelodyChain was trained with order={self.melody.order}, "
                f"but generate() requested order={order}."
            )
        if start_chord == UNK_CHORD_INDEX:
            raise ValueError("start_chord cannot be UNK_CHORD_INDEX.")

        progression: Composition = []
        current_chord = start_chord

        for step in range(n_chords):
            notes = self._sample_notes_for_chord(current_chord, order, notes_per_chord)
            progression.append((current_chord, notes))
            if step < n_chords - 1:
                current_chord = self.harmony.sample(current_chord)
        return progression

    # sample notes for a given chord
    # order 1: sample current note only
    # order 2: sample prev and current notes
    def _sample_notes_for_chord(self, chord_index: ChordIndex, order: int, notes_per_chord: int) -> List[NoteIndex]:
        if order == 1:
            return self._sample_notes_order1(chord_index, notes_per_chord)
        return self._sample_notes_order2(chord_index, notes_per_chord)

    def _seed_note_for_chord(self, chord_index: ChordIndex) -> NoteIndex:
        counts = self.melody.counts.get(chord_index)
        if counts is None:
            raise RuntimeError(f"Cannot generate notes: chord index {chord_index} was not seen during melody training.")
        active_rows = np.flatnonzero(counts.sum(axis=1) > 0)
        if len(active_rows) == 0:
            raise RuntimeError(f"Cannot generate notes: no melody transitions for chord index {chord_index}.")
        return int(np.random.choice(active_rows))

    # sample notes for a given chord and order 1
    # return a list of note indices of length notes_per_chord
    def _sample_notes_order1(self, chord_index: ChordIndex, notes_per_chord: int) -> List[NoteIndex]:
        notes: List[NoteIndex] = [self._seed_note_for_chord(chord_index)]
        current = notes[0]
        for _ in range(notes_per_chord - 1):
            current = self.melody.sample(chord_index, current)
            notes.append(current)
        return notes

    # sample notes for a given chord and order 2
    # return a list of note indices of length notes_per_chord
    def _sample_notes_order2(self, chord_index: ChordIndex, notes_per_chord: int) -> List[NoteIndex]:
        if notes_per_chord == 1:
            return [self._seed_note_for_chord(chord_index)]

        prev_note = self._seed_note_for_chord(chord_index)
        current = self._seed_note_for_chord(chord_index)
        notes = [prev_note, current]

        for _ in range(notes_per_chord - 2):
            state: NoteState = (prev_note, current)
            next_note = self.melody.sample(chord_index, state)
            notes.append(next_note)
            prev_note, current = current, next_note
        return notes