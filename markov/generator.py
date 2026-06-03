"""
full composition pipeline
- Accept a trained HierarchicalMarkovModel and generation parameters
- Sample a chord progression from the harmony layer
- For each chord, sample notes from the corresponding melody chain
- Return a structured composition as a list of (chord, [notes]) pairs
- Support order-1 and order-2 melody generation via a single parameter
- Support a comparison mode : generate order-1 and order-2 in one call
"""