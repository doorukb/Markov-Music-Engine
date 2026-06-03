from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from music21 import chord, converter, note, pitch

NoteToken = int
ChordToken = str

__all__ = ["parse_midi", "ChordToken", "NoteToken"]
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

# extract an aligned chord progression and monophonic melody from one MIDI file
def parse_midi(midi_path: Path) -> Tuple[List[ChordToken], List[NoteToken]]:
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    score = converter.parse(str(midi_path))
    chordified = score.chordify()

    chord_sequence: List[ChordToken] = []
    note_sequence: List[NoteToken] = []

    for element in chordified.flatten().notesAndRests:
        if isinstance(element, note.Rest):
            continue
        if isinstance(element, note.Note):
            chord_sequence.append(_note_to_label(element))
            note_sequence.append(_soprano_midi(element))
        elif isinstance(element, chord.Chord):
            chord_sequence.append(_chord_to_label(element))
            note_sequence.append(_soprano_midi(element))

    return chord_sequence, note_sequence