from pathlib import Path

from markov import (
    build_chord_vocabulary,
    build_note_vocabulary,
    chord_vocabulary_inverse,
    decode_chords,
    decode_notes,
    encode_chords,
    encode_notes,
    parse_midi,
)

def test_encode_decode_round_trip(corpus_midi_path: Path) -> None:
    chord_sequence, note_sequence = parse_midi(corpus_midi_path)

    chord_to_index = build_chord_vocabulary([chord_sequence])
    index_to_chord = chord_vocabulary_inverse(chord_to_index)
    note_to_index = build_note_vocabulary()

    chord_ids = encode_chords(chord_sequence, chord_to_index)
    note_ids = encode_notes(note_sequence, note_to_index)

    assert decode_chords(chord_ids, index_to_chord) == chord_sequence
    assert decode_notes(note_ids, note_to_index) == note_sequence