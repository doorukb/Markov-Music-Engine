"""
main entry point for the Markov Music Engine

Usage : 
    python main.py --style classical --order 1 --n_chords 16 --tempo 120
    python main.py --style jazz --order 2 --compare

What it does : 
- Parse command-line arguments
- Load the appropriate corpus and train the hierarchical model
- Run generation and write outputs
- Print analysis results (stationary distribution, entropy, mixing time)
"""