# Markov Music Engine

- Python 3.10+
- NumPy
- music21
- pygame

 

Markov Music Engine is a hierarchical Markov chain system that learns the statistical structure of music from a MIDI corpus and generates new, stylistically coherent compositions. Unlike toy Markov text generators, this engine operates on two nested levels simultaneously: a chord-level chain that captures harmonic progressions, and a per-chord melody chain where transition probabilities depend on the current harmonic context. The result is a Hierarchical Markov Model that produces compositions that genuinely reflect the style of the source material rather than random note sequences.

What this project asks is not "can this produce music indistinguishable from Bach" but "what does statistical sequence modeling actually capture in music, and how far can we push it before it structurally fails." The engine is built to answer that question concretely, with a full analysis layer that exposes the Markov structure directly through stationary distributions, Shannon entropy, spectral gaps, and mixing time estimates.



## How it works

### Layer 1: Harmony (Chord Chain)

A first-order Markov chain over chord states. The parser extracts chord labels from each MIDI file (e.g. D-major, A-minor, G-dominant-seventh) using music21's chordify method and soprano voice extraction. These are accumulated into a square transition matrix where entry `T[i, j]` is the probability of chord `j` following chord `i`, estimated by maximum likelihood counting across the entire training corpus. Laplace smoothing with `alpha = 0.01` ensures every row sums to 1, which prevents zero-probability transitions from halting generation.

The harmony layer is always order-1 regardless of the melody order selected.

### Layer 2: Melody (Melody Chain)

For each chord context encountered during training, a separate Markov chain is learned over MIDI note indices (0-127). The melody chain's order controls how many previous notes are used as context when predicting the next note:

- **Order 1**: next note depends on the current note only. State space per chord is 128 rows x 128 columns. Memory-safe for any corpus size.
- **Order 2**: next note depends on the previous two notes. State space per chord is 16,384 rows (128 x 128 encoded states) x 128 columns. Memory-safe for any corpus size, though training takes longer on large corpora.
- **Order 3**: next note depends on the previous three notes. State space per chord grows to 2,097,152 encoded states. Order-3 uses sparse per-chord storage internally (not a dense matrix), but memory and CPU usage still grow significantly with corpus size. The CLI will print a resource warning and ask for confirmation before proceeding. See the section on order-3 below.

Higher order means more local coherence (melodic phrases sound more intentional) but also a higher chance of reproducing training material verbatim, especially on small corpora. On a single-piece training run (`--single-source`), order-2 and order-3 melodies can drift into near-random behavior once they reach note combinations that were never observed in training, because smoothing kicks in and the chain is effectively guessing. This is a fundamental property of the model, not a bug.

### Training

Parsing, encoding, and counting happen in a single pass per file. The corpus loader supports two sources for each style:

- **Nottingham Dataset** (downloaded automatically on first run): a collection of traditional folk tunes organized by style. The classical corpus maps to ashover tunes, jazz to jigs, and pop to reels.
- **music21 built-in corpus**: Bach chorales for classical, trecento music for jazz, and Essen folk songs for pop. Available offline, used as fallback when Nottingham subfolders are missing.

Both sources are combined and deduplicated by default.

### Generation

1. A starting chord is sampled uniformly from all active harmony states (states with at least one outgoing transition).
2. The chord chain samples the next chord, producing a progression of `n_chords` steps.
3. For each chord step, the melody chain samples `notes_per_chord` notes conditioned on that chord context.
4. The result is rendered to a music21 Score and written to a MIDI file under `outputs/`.

Each chord step occupies exactly one 4/4 measure at the specified tempo. The duration of the generated piece is therefore `n_chords * (4 * 60 / tempo_bpm)` seconds.

 

## Analysis layer

After every generation run the engine prints a harmony analysis dashboard for each order:

- **Dominant chord**: the chord with the highest stationary probability, i.e. the chord the progression spends the most time on in the long run.
- **Chain entropy (bits)**: the stationary-weighted average Shannon entropy of each row in the transition matrix. Higher entropy means the chain is less predictable. Maximum entropy is `log2(vocab_size)`.
- **Mixing time (est.)**: the number of steps until the chain is approximately at its stationary distribution, estimated from the spectral gap (`1 / (1 - lambda_2)`). A low mixing time means the style converges quickly to its characteristic chord.
- **Stationary distribution**: the top-10 chords ranked by long-run probability.

When multiple orders are generated together, these metrics are shown in a side-by-side table. The harmony metrics are identical across orders because the harmony layer is always order-1; the structural difference between orders lies entirely in the melody layer.

 

## Source coherence

By default the engine trains on the entire corpus for the selected style (hundreds of files). In that mode, the generated pieces statistically reflect the full style but have no particular relationship to any single source piece.

When `--single-source` is set, both harmony and melody models are trained on exactly one piece (the resolved source). The generated output shares the same chord vocabulary and note range as that piece, making the comparison between the original and the generated outputs meaningful. Use this whenever you want to hear what the engine learns from a specific piece.

 

## Installation

```bash
git clone https://github.com/doorukb/Markov-Music-Engine.git
cd Markov-Music-Engine
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

The Nottingham MIDI dataset is downloaded automatically the first time you run the engine for any style. No manual dataset setup is required.

 

## CLI

All generation is done through `main.py`. Run from the project root.

```bash
python main.py --help
```

### Flag reference

- `--style {classical|jazz|pop}` - Style corpus to train on. Required.
- `--order {1|2|3}` - Melody Markov order for single-order runs. Default: 1.
- `--orders N [N ...]` - Generate multiple orders in one run, e.g. `--orders 1 2 3`. Overrides `--order`.
- `--compare` - Shorthand for `--orders 1 2`. Generates order-1 and order-2 side by side.
- `--yes` / `-y` - Skip the order-3 resource confirmation prompt. Useful for scripts and CI.
- `--n-chords N` - Number of 4/4 measures to generate. Default: 16.
- `--duration SECONDS` - Target length for generated pieces in seconds. Overrides `--n-chords`.
- `--notes-per-chord N` - Melody notes sampled per measure. Default: 4.
- `--tempo BPM` - Tempo of the generated score and playback. Default: 120.
- `--single-source` - Train on one piece only so generated output reflects that specific piece.
- `--source PATH|STYLE` - Which piece to use as the original. See source resolution below.
- `--play` - After generation, play Original then each generated order in sequence. Requires pygame.
- `--save-model DIR` - Save trained model(s) under `DIR/order{N}` for multi-order or `DIR` for single-order.
- `--load-model DIR` - Load previously saved model(s) instead of retraining.

### Source resolution (`--source`)

The `--source` flag controls which piece is treated as the "original" for export and playback, and also which piece is used for training when `--single-source` is set.

- **Omitted**: a random piece from the `--style` corpus is chosen each run.
- **A style keyword** (e.g. `--source jazz`): a random piece from that style's corpus is used, regardless of `--style`.
- **A file path** (e.g. `--source outputs/jazz_original.mid`): that exact file is used.

If the value is not a valid file path or a recognized style keyword, the engine logs a warning and falls back to a random `--style` piece.

### Order 3 resource warning

Order-3 melody chains grow the state space to 128^3 possible note triplets per chord context. Even with sparse storage the memory and training time scale significantly with corpus size. Before any order-3 run the CLI prints:

```
Order-3 melody chains use sparse per-chord storage but can still consume
significant memory and CPU on large corpora. Continue? [y/N]
```

Type `y` to proceed or press Enter to cancel. Pass `--yes` / `-y` to skip this prompt entirely.

On a single-piece training run (`--single-source`) order-3 is much lighter than corpus-wide training and typically completes in a few seconds.

### Playback (`--play`)

When `--play` is set, after generation the engine plays the following sequence with a tqdm progress bar for each track:

1. **Original** - the resolved source piece, played at its full natural length.
2. **Order 1** (or whichever orders were generated), played in order.

During playback, press `s` to skip the current track and advance to the next one. The bar fills to 100% when a track finishes or is skipped.

Playback requires `pygame` (included in `requirements.txt`) and uses the system's built-in MIDI synthesizer, so no soundfont or FluidSynth installation is needed.

WAV export (via FluidSynth) is attempted automatically after each MIDI write but is skipped gracefully if FluidSynth is not installed.

 

## Example commands

### Hear original, order 1, then order 2 (most common workflow)

The most useful way to experience the engine. Train on one piece so all three outputs are comparable, then listen to them in sequence:

```bash
# Classical - random Bach chorale
python main.py --style classical --compare --play --single-source

# Jazz - random jig from the Nottingham corpus
python main.py --style jazz --compare --play --single-source

# Pop - random reel from the Nottingham corpus
python main.py --style pop --compare --play --single-source
```

Each of these trains two models (one per melody order) on the same randomly chosen piece, writes `outputs/{style}_order1.mid` and `outputs/{style}_order2.mid`, prints the side-by-side harmony analysis, then plays original -> order 1 -> order 2.

### Pin the source piece and control the length

```bash
# Use a specific source file, generate ~30s of output for each order
python main.py --style jazz --compare --play --single-source \
    --source "outputs/jazz_original.mid" --duration 30

# Pick a random jazz piece as the source even though --style is classical
python main.py --style classical --compare --play --single-source --source jazz
```

### All three orders together

```bash
# Pop, all orders 1/2/3, 10s per generated piece, skip confirmation prompt
python main.py --style pop --orders 1 2 3 --yes --duration 10 --play --single-source

# Same but without playback - analysis table only
python main.py --style pop --orders 1 2 3 --yes --duration 10 --single-source
```

### Single order runs

```bash
# Generate one order-2 piece from a random classical piece, ~20s long
python main.py --style classical --order 2 --play --single-source --duration 20

# Order-3 from a single jazz piece (confirmation prompt will appear)
python main.py --style jazz --order 3 --single-source --n-chords 8

# Order-3, skip confirmation, non-interactive
python main.py --style classical --order 3 --yes --single-source --n-chords 8
```

### Analysis only (no audio)

```bash
# Side-by-side order 1 vs order 2, corpus-wide training, no playback
python main.py --style jazz --compare

# Single order, 16 chords, print stationary distribution
python main.py --style classical --order 1 --n-chords 16

# All three orders, analysis table
python main.py --style pop --orders 1 2 3 --yes
```

### Save and reload trained models

Training on the full corpus can take several minutes. Save the model once and reload it on subsequent runs:

```bash
# Train and save order 1 and 2
python main.py --style pop --compare --save-model models/pop

# Reload and play without retraining
python main.py --style pop --compare --play --load-model models/pop

# Train and save order 3
python main.py --style pop --order 3 --yes --save-model models/pop3

# Reload order 3
python main.py --style pop --order 3 --yes --load-model models/pop3
```

Multi-order saves write each order to `DIR/order{N}` (e.g. `models/pop/order1`, `models/pop/order2`). Single-order saves write directly to `DIR`.

 

## Outputs

All generated files are written to `outputs/`:

- `outputs/{style}_original.mid` - The source piece converted to MIDI for playback. Only created when `--play` is set.
- `outputs/{style}_order{N}.mid` - Generated composition for order N.
- `outputs/{style}_order{N}.wav` - WAV render of the above, if FluidSynth is installed.
 

## WAV export (optional)

If FluidSynth is installed and a soundfont is placed at `data/soundfont.sf2`, the engine will render each generated MIDI to WAV automatically. If FluidSynth is not available, WAV export is silently skipped and MIDI files are still written. The `--play` flag uses pygame directly and does not require FluidSynth.

FluidSynth installation: https://www.fluidsynth.org