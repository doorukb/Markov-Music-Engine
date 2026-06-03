# global configuration to add all path constants, model hyperparameters, 
# and style definitions. All modules import from here
import os
from pathlib import Path

# paths
ROOT_DIR        = Path(__file__).parent
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
SOUNDFONT_PATH  = ROOT_DIR / "data" / "soundfont.sf2"

# model hyperparameters

SMOOTHING_ALPHA = 0.01
SUPPORTED_ORDERS = [1, 2]
DEFAULT_ORDER        = 1
DEFAULT_N_CHORDS     = 16
DEFAULT_TEMPO_BPM    = 120
CONVERGENCE_THRESHOLD = 1e-8
MAX_ITERATIONS        = 10_000

# styles
SUPPORTED_STYLES = ["classical", "jazz", "pop"]

# audio
SAMPLE_RATE     = 44100
AUDIO_FORMAT    = "wav"