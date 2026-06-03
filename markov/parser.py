"""
music21-based MIDI to structured sequence extraction
- Parse a MIDI file path into a music21 Score object
- Extract the chord progression as a sequence of chord tokens
- Extract the melody as a sequence of note tokens
- Return aligned (chord_sequence, note_sequence) pairs per file
- Handle polyphonic MIDI gracefully (chordify + soprano extraction)
"""