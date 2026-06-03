from pathlib import Path
from markov import parse_midi

def test_parse_midi_returns_aligned_nonempty_sequences(corpus_midi_path: Path) -> None:
    chord_sequence, note_sequence = parse_midi(corpus_midi_path)
    assert len(chord_sequence) > 0
    assert len(note_sequence) > 0
    assert len(chord_sequence) == len(note_sequence)
    assert all(isinstance(label, str) and label for label in chord_sequence)
    assert all(isinstance(pitch, int) and 0 <= pitch <= 127 for pitch in note_sequence)
