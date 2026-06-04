from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Sequence
from markov.parser import ChordToken, NoteToken

ChordIndex = int
NoteIndex = int

UNK_CHORD = "<unk>"
MIN_MIDI = 0
MAX_MIDI = 127
NOTE_VOCAB_SIZE = MAX_MIDI - MIN_MIDI + 1

__all__ = [
    "UNK_CHORD",
    "NOTE_VOCAB_SIZE",
    "ChordIndex",
    "NoteIndex",
    "build_chord_vocabulary",
    "build_note_vocabulary",
    "chord_vocabulary_inverse",
    "encode_chords",
    "decode_chords",
    "encode_notes",
    "decode_notes",
]

# build a chord label to index map from one or more chord sequences
def build_chord_vocabulary(chord_sequences: Iterable[Sequence[ChordToken]]) -> Dict[ChordToken, ChordIndex]:
    labels = {
        label
        for sequence in chord_sequences
        for label in sequence
        if label != UNK_CHORD
    }
    stoi: Dict[ChordToken, ChordIndex] = {UNK_CHORD: 0}
    for idx, label in enumerate(sorted(labels), start=1):
        stoi[label] = idx
    return stoi

# index to chord label lookup table matching chord_to_index
def chord_vocabulary_inverse(chord_to_index: Mapping[ChordToken, ChordIndex]) -> List[ChordToken]:
    size = len(chord_to_index)
    itos: List[ChordToken | None] = [None] * size
    for label, index in chord_to_index.items():
        if index < 0 or index >= size:
            raise ValueError(f"chord index {index} out of range for vocabulary size {size}")
        if itos[index] is not None and itos[index] != label:
            raise ValueError(f"duplicate chord index {index}: {itos[index]!r} and {label!r}")
        itos[index] = label
    if any(token is None for token in itos):
        missing = [i for i, token in enumerate(itos) if token is None]
        raise ValueError(f"chord vocabulary has gaps at indices: {missing}")
    return [token for token in itos if token is not None]


def build_note_vocabulary() -> Dict[NoteToken, NoteIndex]:
    return {midi: midi for midi in range(MIN_MIDI, MAX_MIDI + 1)}

def encode_chords(tokens: Sequence[ChordToken], chord_to_index: Mapping[ChordToken, ChordIndex]) -> List[ChordIndex]:
    unk = chord_to_index.get(UNK_CHORD, 0)
    return [chord_to_index.get(token, unk) for token in tokens]

def decode_chords(indices: Sequence[ChordIndex], index_to_chord: Sequence[ChordToken]) -> List[ChordToken]:
    try:
        return [index_to_chord[i] for i in indices]
    except IndexError as exc:
        raise IndexError(f"chord index out of vocabulary range: {exc}") from exc

def encode_notes(tokens: Sequence[NoteToken], note_to_index: Mapping[NoteToken, NoteIndex] | None = None) -> List[NoteIndex]:
    vocab = note_to_index if note_to_index is not None else build_note_vocabulary()
    encoded: List[NoteIndex] = []
    for midi in tokens:
        if midi not in vocab:
            if MIN_MIDI <= midi <= MAX_MIDI:
                raise KeyError(f"MIDI pitch {midi} missing from note vocabulary")
            raise ValueError(f"MIDI pitch {midi} outside {MIN_MIDI}–{MAX_MIDI}")
        encoded.append(vocab[midi])
    return encoded

def decode_notes(indices: Sequence[NoteIndex], index_to_note: Mapping[NoteIndex, NoteToken] | None = None) -> List[NoteToken]:
    if index_to_note is None:
        index_to_note = build_note_vocabulary()
    try:
        return [index_to_note[i] for i in indices]
    except KeyError as exc:
        raise KeyError(f"note index out of vocabulary range: {exc}") from exc