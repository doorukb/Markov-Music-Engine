"""
data loading and MIDI dataset management
- Download and organize the Nottingham MIDI dataset
- Expose music21's built-in corpus (Bach chorales, etc.)
- Map style names (classical, jazz, pop) to lists of MIDI file paths
- Provide a single load_corpus(style) → List[Path] interface
"""