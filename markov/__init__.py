"""
markov/

Everything about music happens here

data        : data loading and MIDI dataset management
encoder     : chord/note tokenization and vocabulary mapping
parser      : music21-based MIDI to sequence extraction
harmony     : Level 1- chord-level Markov chain (MLE training + sampling)
melody      : Level 2- note-level Markov chains conditioned on chord state
matrix      : HierarchicalMarkovModel composing harmony + melody layers
smoothing   : Laplace and other smoothing strategies for sparse matrices
analysis    : stationary distribution, entropy, mixing time, matrix powers
generator   : full composition pipeline (chord sequence → note sequence)
renderer    : MIDI file writer and FluidSynth audio synthesis
"""