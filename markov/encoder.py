"""
Chord and note tokenization to map raw music21 objects to discrete tokens
- Define the chord vocabulary (e.g., C-major to index 0)
- Define the note vocabulary (MIDI pitch 0-127 to index)
- Encode sequences of music21 objects to integer token sequences
- Decode integer token sequences to music21 objects for rendering
- Handle unknown tokens gracefully (OOV strategy)
"""