from pathlib import Path
import pytest
from music21 import corpus

@pytest.fixture(scope="module")
def corpus_midi_path() -> Path:
    paths = corpus.getComposer("bach")
    assert paths, "music21 Bach corpus is unavailable"
    return Path(paths[0])
