"""
music21-based MIDI to structured sequence extraction
- Parse a MIDI file path into a music21 Score object
- Extract the chord progression as a sequence of chord tokens
- Extract the melody as a sequence of note tokens
- Return aligned (chord_sequence, note_sequence) pairs per file
- Handle polyphonic MIDI gracefully (chordify + soprano extraction)
"""
from pathlib import Path
from typing import List, Tuple
from music21 import chord, converter, note, pitch
NoteToken = int
ChordToken = str

_COMMON_NAME_TO_QUALITY: dict[str, str] = {
    "major triad": "major",
    "minor triad": "minor",
    "diminished triad": "diminished",
    "augmented triad": "augmented",
    "dominant seventh chord": "dominant-seventh",
    "major seventh chord": "major-seventh",
    "minor seventh chord": "minor-seventh",
    "half-diminished seventh chord": "half-diminished-seventh",
    "diminished seventh chord": "diminished-seventh",
}

# pitch spelling for labels
def _spell_pitch(p: pitch.Pitch) -> str:
    return p.name.replace("-", "b")

def _quality_label(c: chord.Chord) -> str:
    common = (c.commonName or "").strip()
    if common in _COMMON_NAME_TO_QUALITY:
        return _COMMON_NAME_TO_QUALITY[common]
    if c.quality:
        return c.quality.replace(" ", "-")
    return "major"

def _chord_to_label(c: chord.Chord) -> ChordToken:
    root_pitch = c.root()
    if root_pitch is None:
        root_pitch = c.bass()
    if root_pitch is None:
        return "unknown"
    return f"{_spell_pitch(root_pitch)}-{_quality_label(c)}"

def _note_to_label(n: note.Note) -> ChordToken:
    return f"{_spell_pitch(n.pitch)}-major"

def _soprano_midi(element: note.Note | chord.Chord) -> int:
    if isinstance(element, note.Note):
        return element.pitch.midi
    return max(element.pitches, key=lambda p: p.midi).midi

def _load_chordified(midi_path: Path):
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    score = converter.parse(str(midi_path))
    return score.chordify()

# parse a MIDI file into aligned chord and monophonic note token sequences
def parse_midi(midi_path: Path) -> Tuple[List[ChordToken], List[NoteToken]]:
    chordified = _load_chordified(midi_path)
    chord_tokens: List[ChordToken] = []
    note_tokens: List[NoteToken] = []

    for element in chordified.flatten().notesAndRests:
        if isinstance(element, note.Rest):
            continue
        if isinstance(element, note.Note):
            chord_tokens.append(_note_to_label(element))
            note_tokens.append(_soprano_midi(element))
        elif isinstance(element, chord.Chord):
            chord_tokens.append(_chord_to_label(element))
            note_tokens.append(_soprano_midi(element))

    return chord_tokens, note_tokens

# chord progression as named labels, such as C major or A minor
def parse_midi_chords(midi_path: Path) -> List[ChordToken]:
    chords, _ = parse_midi(midi_path)
    return chords

# monomorphic melody as MIDI pitch tokens
def parse_midi_notes(midi_path: Path) -> List[NoteToken]:
    _, notes = parse_midi(midi_path)
    return notes