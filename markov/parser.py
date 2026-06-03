"""
music21-based MIDI to structured sequence extraction
- Parse a MIDI file path into a music21 Score object
- Extract the chord progression as a sequence of chord tokens
- Extract the melody as a sequence of note tokens
- Return aligned (chord_sequence, note_sequence) pairs per file
- Handle polyphonic MIDI gracefully (chordify + soprano extraction)
"""
from pathlib import Path
from typing import List
from music21 import chord, converter, note
NoteToken = int

# parse a MIDI file into a monophonic melody as a flat list of note tokens
def parse_midi_notes(midi_path: Path) -> List[NoteToken]:
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    score = converter.parse(str(midi_path))
    chordified = score.chordify()

    tokens: List[NoteToken] = []
    for element in chordified.flatten().notesAndRests:
        if isinstance(element, note.Note):
            tokens.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            soprano = max(element.pitches, key=lambda p: p.midi)
            tokens.append(soprano.midi)
    return tokens