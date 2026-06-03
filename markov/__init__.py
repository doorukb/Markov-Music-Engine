"""
markov/

Everything about music happens here

data        : data loading and MIDI dataset management
encoder     : chord/note tokenization and vocabulary mapping
parser      : music21-based MIDI to sequence extraction
harmony     : Level 1- chord-level Markov chain (MLE training + sampling)
melody      : Level 2- note-level Markov chains conditioned on chord state
matrix      : HierarchicalMarkovModel composing harmony + melody layers
smoothing   : Laplace and other smoothing strategies for sparse matrices
analysis    : stationary distribution, entropy, mixing time, matrix powers
generator   : full composition pipeline (chord sequence → note sequence)
renderer    : MIDI file writer and FluidSynth audio synthesis
"""
from markov.encoder import (
    build_chord_vocabulary,
    build_note_vocabulary,
    chord_vocabulary_inverse,
    decode_chords,
    decode_notes,
    encode_chords,
    encode_notes,
)
from markov.parser import ChordToken, NoteToken, parse_midi

__all__ = [
    "parse_midi",
    "ChordToken",
    "NoteToken",
    "build_chord_vocabulary",
    "build_note_vocabulary",
    "chord_vocabulary_inverse",
    "encode_chords",
    "decode_chords",
    "encode_notes",
    "decode_notes",
]