"""Soundfont provisioning and WAV synthesis capability checks."""
from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

from config import SOUNDFONT_PATH

__all__ = ["check_audio_setup", "ensure_soundfont"]

logger = logging.getLogger(__name__)

_SOUNDFONT_URLS = (
    "https://github.com/pianobooster/fluid-soundfont/releases/download/v3.1/FluidR3_GM.sf2",
    "https://github.com/fhunleth/midi_synth/releases/download/v0.1.0/FluidR3_GM.sf2",
)
_MIN_SOUNDFONT_BYTES = 500_000


def check_audio_setup() -> dict[str, bool | str]:
    cli = shutil.which("fluidsynth")
    soundfont_present = SOUNDFONT_PATH.is_file()
    pyfluidsynth_ok = False
    if soundfont_present:
        try:
            import fluidsynth  # noqa: F401  # pyfluidsynth package

            _ = fluidsynth.Synth
            pyfluidsynth_ok = True
        except (ImportError, AttributeError, OSError):
            pass
    can_synthesize = soundfont_present and (cli is not None or pyfluidsynth_ok)
    return {
        "fluidsynth_cli": cli is not None,
        "fluidsynth_cli_path": cli or "",
        "soundfont_present": soundfont_present,
        "pyfluidsynth": pyfluidsynth_ok,
        "can_synthesize_wav": can_synthesize,
    }


def ensure_soundfont() -> Path:
    if SOUNDFONT_PATH.is_file():
        size = SOUNDFONT_PATH.stat().st_size
        if size >= _MIN_SOUNDFONT_BYTES:
            return SOUNDFONT_PATH
        logger.warning("Existing soundfont at %s is too small (%s bytes); re-downloading.", SOUNDFONT_PATH, size)
        SOUNDFONT_PATH.unlink(missing_ok=True)

    SOUNDFONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = SOUNDFONT_PATH.with_suffix(".sf2.part")
    last_error: Exception | None = None
    for url in _SOUNDFONT_URLS:
        try:
            logger.info("Downloading soundfont from %s …", url)
            urllib.request.urlretrieve(url, tmp_path)
            size = tmp_path.stat().st_size
            if size < _MIN_SOUNDFONT_BYTES:
                raise RuntimeError(f"Downloaded soundfont is too small ({size} bytes).")
            tmp_path.replace(SOUNDFONT_PATH)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)
    if last_error is not None:
        raise RuntimeError(
            f"Could not download soundfont to {SOUNDFONT_PATH}: {last_error}"
        ) from last_error

    logger.info("Soundfont ready at %s", SOUNDFONT_PATH)
    return SOUNDFONT_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = ensure_soundfont()
    print(f"Soundfont ready: {path}")
    print(check_audio_setup())
