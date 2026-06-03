"""
Level 2 : note-level Markov chains conditioned on chord state
- Maintain one transition matrix per chord state
- Support order-1 (current note to next note) chains
- Support order-2 (prev note, current note to next note) chains
- MLE training from aligned (chord, note) sequence pairs
- Sample next note given (current note, current chord) context
- Serialize and deserialize all per-chord matrices
"""