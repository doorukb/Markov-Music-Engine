"""
HierarchicalMarkovModel : composes the harmony and melody layers
- Provide a single .train(corpus) entry point across both layers
- Wire ChordChain and MelodyChain together into one model object
- Apply smoothing post-training across all transition matrices
- Expose .save() and .load() for full model persistence
- Act as the single object passed to the generator and analysis modules
"""