from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

import markov.encoder as encoder
import markov.parser as parser
from markov import build_chord_vocabulary, encode_chords, parse_midi
from markov.harmony import UNK_CHORD_INDEX, ChordChain

ROW_SUM_RTOL = 1e-9

@pytest.fixture
def trained_chain(corpus_midi_path: Path) -> ChordChain:
    chord_sequence, _ = parse_midi(corpus_midi_path)
    chord_to_index = build_chord_vocabulary([chord_sequence])
    encoded = encode_chords(chord_sequence, chord_to_index)

    chain = ChordChain(vocab_size=len(chord_to_index))
    chain.train([encoded])
    chain.normalize()
    return chain

def test_normalized_rows_with_transitions_sum_to_one(trained_chain: ChordChain) -> None:
    assert trained_chain.transition_matrix is not None
    assert trained_chain.counts is not None
    assert np.isfinite(trained_chain.transition_matrix).all()

    active_rows = trained_chain.counts.sum(axis=1) > 0
    row_sums = trained_chain.transition_matrix[active_rows].sum(axis=1)
    assert np.allclose(row_sums, 1.0, rtol=ROW_SUM_RTOL)

def test_sample_returns_valid_chord_index(trained_chain: ChordChain) -> None:
    assert trained_chain.vocab_size is not None
    assert trained_chain.counts is not None

    active_states = np.flatnonzero(trained_chain.counts.sum(axis=1) > 0)
    assert len(active_states) > 0

    np.random.seed(0)
    for _ in range(100):
        state = int(np.random.choice(active_states))
        sampled = trained_chain.sample(state)
        assert isinstance(sampled, int)
        assert 0 <= sampled < trained_chain.vocab_size

def test_empty_corpus_raises_descriptive_error() -> None:
    chain = ChordChain()
    encoder.chord_to_index = {"<unk>": UNK_CHORD_INDEX, "C-major": 1}

    with pytest.raises(ValueError, match="empty corpus.*no MIDI file paths"):
        chain.train_corpus([], parser, encoder)

    chain.train([])
    with pytest.raises(RuntimeError, match="train ChordChain before normalizing"):
        chain.normalize()

    chain2 = ChordChain(vocab_size=3)
    chain2.train([[1, 1]])  # only self-loop at 1; valid counts
    chain2.counts[:] = 0
    with pytest.raises(RuntimeError, match="no chord transitions were accumulated"):
        chain2.normalize()

def test_counts_accumulate_across_sequences() -> None:
    chain = ChordChain(vocab_size=4)
    seq_a = [1, 2, 3]
    seq_b = [2, 3, 1]

    chain.train([seq_a])
    assert chain.counts is not None
    assert chain.counts[1, 2] == 1
    assert chain.counts[2, 3] == 1

    chain.train([seq_b])
    assert chain.counts[1, 2] == 1
    assert chain.counts[2, 3] == 2
    assert chain.counts[3, 1] == 1
    assert int(chain.counts.sum()) == 4