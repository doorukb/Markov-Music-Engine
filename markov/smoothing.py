"""
Smoothing strategies for sparse Markov transition matrices
- Implement Laplace (add-alpha) smoothing
- Apply smoothing to a raw count matrix before normalization
- Ensure no row is all-zero after smoothing (dead-end prevention)
- Accept alpha as a parameter (default from config.SMOOTHING_ALPHA)
"""