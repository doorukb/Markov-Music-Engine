# download and unzip the nottingham MIDI dataset into data/raw/nottingham/
import urllib.request
import zipfile
import logging
from pathlib import Path
from typing import List
from music21 import corpus
from config import DATA_RAW_DIR, SUPPORTED_STYLES

__all__ = ["load_corpus", "collect_chord_sequences", "download_nottingham"]

logger = logging.getLogger(__name__)

# special thanks to jukedeck for the nottingham MIDI dataset
NOTTINGHAM_URL = ("https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip")
# path to the nottingham MIDI dataset
NOTTINGHAM_DIR = DATA_RAW_DIR / "nottingham"

# mapping each style to the Nottingham MIDI filename prefix.
# Files live flat under MIDI/ as <prefix><number>.mid (e.g. ashover1.mid, jigs12.mid,
# reelsa-c1.mid), so styles are selected by filename prefix rather than by subfolder.
NOTTINGHAM_STYLE_MAP = {
    "classical": "ashover",
    "pop":       "reels",
    "jazz":      "jigs",
}
MUSIC21_STYLE_MAP = {
    "classical": "bach",
    "jazz":      "trecento",
    "pop":       "essenFolksong",
}

# download and unzip the nottingham MIDI dataset into data/raw/nottingham/
def download_nottingham() -> None:
    if NOTTINGHAM_DIR.exists():
        logger.info("Nottingham dataset already present — skipping download.")
        return

    logger.info("Downloading Nottingham dataset...")
    zip_path = DATA_RAW_DIR / "nottingham.zip"
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(NOTTINGHAM_URL, zip_path)
    logger.info("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW_DIR)

    # rename extracted folder to a clean name
    extracted = DATA_RAW_DIR / "nottingham-dataset-master"
    if extracted.exists():
        extracted.rename(NOTTINGHAM_DIR)

    zip_path.unlink(missing_ok=True)
    logger.info(f"Nottingham dataset ready at {NOTTINGHAM_DIR}")

# download the dataset and return all .mid file paths for a given style from the nottingham
def _load_nottingham_paths(style: str) -> List[Path]:
    download_nottingham()

    prefix = NOTTINGHAM_STYLE_MAP.get(style)
    if prefix is None:
        raise ValueError(f"unknown style : '{style}'. Choose from {SUPPORTED_STYLES}.")

    midi_dir = NOTTINGHAM_DIR / "MIDI"
    if not midi_dir.is_dir():
        raise FileNotFoundError(f"expected Nottingham MIDI directory not found: {midi_dir}\ntry deleting {NOTTINGHAM_DIR} and re-running to re-download.")

    paths = sorted(midi_dir.glob(f"{prefix}*.mid"))
    if not paths:
        raise FileNotFoundError(f"no Nottingham MIDI files with prefix {prefix!r} under {midi_dir}\ntry deleting {NOTTINGHAM_DIR} and re-running to re-download.")
    logger.info(f"Nottingham [{style}]: {len(paths)} files found.")
    return paths

# return file paths for a given style from music21's built-in dataset
def _load_music21_paths(style: str) -> List[Path]:
    composer = MUSIC21_STYLE_MAP.get(style)
    if composer is None:
        raise ValueError(f"Unknown style '{style}'. Choose from {SUPPORTED_STYLES}.")

    try:
        paths = [Path(p) for p in corpus.getComposer(composer)]
        logger.info(f"music21 corpus [{style} / {composer}]: {len(paths)} files found.")
        return paths
    except Exception as e:
        logger.warning(f"music21 corpus unavailable for '{composer}': {e}")
        return []

# load MIDI file paths for a given style
def load_corpus(style: str, source: str = "both") -> List[Path]:
    if style not in SUPPORTED_STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from {SUPPORTED_STYLES}.")

    paths: List[Path] = []

    if source in ("nottingham", "both"):
        try:
            paths += _load_nottingham_paths(style)
        except Exception as e:
            logger.warning(f"Nottingham load failed for '{style}': {e}")
    if source in ("music21", "both"):
        paths += _load_music21_paths(style)
    if not paths:
        raise RuntimeError(f"No MIDI files found for style='{style}', source='{source}'. Check your data directory: {DATA_RAW_DIR}")

    # deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    logger.info(f"load_corpus(style='{style}'): {len(unique)} total files.")
    return unique

# collect chord sequences from a list of MIDI files
def collect_chord_sequences(paths: List[Path]) -> List[List[str]]:
    from markov.parser import ParseError, parse_midi
    
    sequences: List[List[str]] = []
    for path in paths:
        try:
            chord_sequence, _ = parse_midi(path)
            sequences.append(chord_sequence)
        except ParseError as exc:
            logger.warning("skipping %s: %s", path, exc)
    return sequences