from __future__ import annotations
import logging
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from config import FLUIDSYNTH_BIN_DIR, SOUNDFONT_PATH

__all__ = [
    "add_fluidsynth_dll_directory",
    "check_audio_setup",
    "ensure_audio_dependencies",
    "ensure_fluidsynth_binary",
    "ensure_soundfont",
    "get_fluidsynth_executable",
]

logger = logging.getLogger(__name__)

_SOUNDFONT_URLS = (
    "https://github.com/pianobooster/fluid-soundfont/releases/download/v3.1/FluidR3_GM.sf2",
    "https://github.com/fhunleth/midi_synth/releases/download/v0.1.0/FluidR3_GM.sf2",
)
_MIN_SOUNDFONT_BYTES = 500_000

_FLUIDSYNTH_VERSION = "2.5.4"
_FLUIDSYNTH_WIN_ZIP = f"fluidsynth-v{_FLUIDSYNTH_VERSION}-win10-x64-cpp11.zip"
_FLUIDSYNTH_RELEASE_URL = (
    f"https://github.com/FluidSynth/fluidsynth/releases/download/"
    f"v{_FLUIDSYNTH_VERSION}/{_FLUIDSYNTH_WIN_ZIP}"
)
_BUNDLED_FLUIDSYNTH_EXE = "fluidsynth.exe"

# get the bundled fluidsynth executable
def _bundled_fluidsynth_exe() -> Path:
    return FLUIDSYNTH_BIN_DIR / _BUNDLED_FLUIDSYNTH_EXE

# verify the fluidsynth executable
def _verify_fluidsynth_exe(exe: Path) -> bool:
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(exe.parent) if exe.parent.is_dir() else None,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("FluidSynth probe failed for %s: %s", exe, exc)
        return False

# prefer the system PATH, then a bundled Windows binary under data/bin/
def get_fluidsynth_executable() -> Path | None:
    cli = shutil.which("fluidsynth")
    if cli:
        return Path(cli)
    bundled = _bundled_fluidsynth_exe()
    if bundled.is_file():
        return bundled
    return None

# add the fluidsynth DLL directory to the PATH
def add_fluidsynth_dll_directory() -> None:
    import os
    if sys.platform == "win32" and hasattr(os, "add_dll_directory") and FLUIDSYNTH_BIN_DIR.is_dir():
        os.add_dll_directory(str(FLUIDSYNTH_BIN_DIR.resolve()))

# probe the pyfluidsynth library
def _probe_pyfluidsynth() -> bool:
    if not SOUNDFONT_PATH.is_file():
        return False
    add_fluidsynth_dll_directory()
    try:
        import fluidsynth
        synth = fluidsynth.Synth()
        synth.delete()
        return True
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("pyfluidsynth probe failed: %s", exc)
        return False

# check the audio setup
def check_audio_setup() -> dict[str, bool | str]:
    exe = get_fluidsynth_executable()
    fluidsynth_cli = exe is not None and _verify_fluidsynth_exe(exe)
    soundfont_present = SOUNDFONT_PATH.is_file()
    pyfluidsynth_ok = soundfont_present and _probe_pyfluidsynth()
    can_synthesize = soundfont_present and (fluidsynth_cli or pyfluidsynth_ok)
    return {
        "fluidsynth_cli": fluidsynth_cli,
        "fluidsynth_cli_path": str(exe) if exe and fluidsynth_cli else "",
        "soundfont_present": soundfont_present,
        "pyfluidsynth": pyfluidsynth_ok,
        "can_synthesize_wav": can_synthesize,
    }

# extract the windows fluidsynth zip
def _extract_windows_fluidsynth_zip(zip_path: Path) -> None:
    FLUIDSYNTH_BIN_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            normalized = member.replace("\\", "/")
            if normalized.endswith("/"):
                continue
            parts = Path(normalized).parts
            if "bin" not in parts:
                continue
            bin_index = parts.index("bin")
            relative = Path(*parts[bin_index + 1 :])
            if not relative.parts:
                continue
            target = FLUIDSYNTH_BIN_DIR / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())

# download and extract portable FluidSynth on Windows when not on PATH
def ensure_fluidsynth_binary() -> Path | None:
    existing = get_fluidsynth_executable()
    if existing is not None and _verify_fluidsynth_exe(existing):
        return existing
    if platform.system() != "Windows":
        return existing
    bundled = _bundled_fluidsynth_exe()
    if bundled.is_file() and _verify_fluidsynth_exe(bundled):
        return bundled
    if bundled.is_file():
        logger.warning("Bundled FluidSynth at %s failed verification; re-downloading.", bundled)
        shutil.rmtree(FLUIDSYNTH_BIN_DIR, ignore_errors=True)

    FLUIDSYNTH_BIN_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = FLUIDSYNTH_BIN_DIR / f".{_FLUIDSYNTH_WIN_ZIP}.part"
    try:
        logger.info("Downloading FluidSynth from %s …", _FLUIDSYNTH_RELEASE_URL)
        urllib.request.urlretrieve(_FLUIDSYNTH_RELEASE_URL, tmp_zip)
        _extract_windows_fluidsynth_zip(tmp_zip)
    except Exception as exc:
        raise RuntimeError(f"Could not download FluidSynth binary to {FLUIDSYNTH_BIN_DIR}: {exc}") from exc
    finally:
        tmp_zip.unlink(missing_ok=True)

    if not bundled.is_file():
        raise RuntimeError(f"FluidSynth executable missing after extract: {bundled}")
    if not _verify_fluidsynth_exe(bundled):
        raise RuntimeError(f"FluidSynth binary failed verification: {bundled}")

    logger.info("FluidSynth ready at %s", bundled)
    return bundled

# provision soundfront and fluidsynth binary (dashboard / make setup-audio)
def ensure_audio_dependencies() -> None:
    ensure_soundfont()
    ensure_fluidsynth_binary()

# ensure the soundfont is present
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
        raise RuntimeError(f"Could not download soundfont to {SOUNDFONT_PATH}: {last_error}") from last_error

    logger.info("Soundfont ready at %s", SOUNDFONT_PATH)
    return SOUNDFONT_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sf = ensure_soundfont()
    print(f"Soundfont ready: {sf}")
    fs = ensure_fluidsynth_binary()
    if fs is not None:
        print(f"FluidSynth ready: {fs}")
    print(check_audio_setup())