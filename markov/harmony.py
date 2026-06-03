"""
Level 1 : chord-level Markov chain.
- Accumulate bigram counts over chord sequences (MLE)
- Aggregate counts across multiple MIDI files
- Normalize counts to row-stochastic transition matrix
- Sample next chord given current chord state
- Serialize and deserialize the transition matrix
"""