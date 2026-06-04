from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from markov import build_chord_vocabulary, build_note_vocabulary, encode_chords, encode_notes, parse_midi
from markov.melody import MelodyChain

ROW_SUM_RTOL = 1e-9

def _trained_melody(corpus_midi_path: Path, order: int) -> MelodyChain:
    chord_sequence, note_sequence = parse_midi(corpus_midi_path)
    chord_to_index = build_chord_vocabulary([chord_sequence])
    note_to_index = build_note_vocabulary()
    chord_ids = encode_chords(chord_sequence, chord_to_index)
    note_ids = encode_notes(note_sequence, note_to_index)

    chain = MelodyChain(order=order)
    chain.train(chord_ids, note_ids)
    chain.normalize()
    return chain

def _assert_active_rows_sum_to_one(chain: MelodyChain) -> None:
    assert chain.transition_matrices is not None
    assert chain.counts is not None
    for chord_index, counts in chain.counts.items():
        transition = chain.transition_matrices[chord_index]
        active_rows = counts.sum(axis=1) > 0
        assert active_rows.any(), f"chord {chord_index} has no transitions"
        row_sums = transition[active_rows].sum(axis=1)
        assert np.allclose(row_sums, 1.0, rtol=ROW_SUM_RTOL), (
            f"chord {chord_index} active rows do not sum to 1"
        )

@pytest.fixture
def melody_order1(corpus_midi_path: Path) -> MelodyChain:
    return _trained_melody(corpus_midi_path, order=1)

@pytest.fixture
def melody_order2(corpus_midi_path: Path) -> MelodyChain:
    return _trained_melody(corpus_midi_path, order=2)

def test_order1_active_rows_sum_to_one(melody_order1: MelodyChain) -> None:
    assert melody_order1.order == 1
    _assert_active_rows_sum_to_one(melody_order1)

def test_order2_active_rows_sum_to_one_flattened_state(melody_order2: MelodyChain) -> None:
    assert melody_order2.order == 2
    assert melody_order2._state_rows == melody_order2.note_vocab_size ** 2
    _assert_active_rows_sum_to_one(melody_order2)

def test_sample_returns_valid_note_index(melody_order1: MelodyChain) -> None:
    assert melody_order1.transition_matrices is not None
    assert melody_order1.counts is not None

    np.random.seed(0)
    for chord_index, counts in melody_order1.counts.items():
        active_notes = np.flatnonzero(counts.sum(axis=1) > 0)
        for _ in range(20):
            current_note = int(np.random.choice(active_notes))
            sampled = melody_order1.sample(chord_index, current_note)
            assert isinstance(sampled, int)
            assert 0 <= sampled < melody_order1.note_vocab_size

def test_order2_state_on_order1_chain_raises(melody_order1: MelodyChain) -> None:
    chord_index = next(iter(melody_order1.counts))
    with pytest.raises(ValueError, match="order-2 state.*order-1 MelodyChain"):
        melody_order1.sample(chord_index, (60, 62))

def test_order1_state_on_order2_chain_raises(melody_order2: MelodyChain) -> None:
    chord_index = next(iter(melody_order2.counts))
    with pytest.raises(ValueError, match="order-1 state.*order-2 MelodyChain"):
        melody_order2.sample(chord_index, 60)

def test_unseen_chord_context_raises_gracefully(melody_order1: MelodyChain) -> None:
    unseen_chord = max(melody_order1.counts) + 999
    with pytest.raises(RuntimeError, match="no transition matrix for chord index"):
        melody_order1.sample(unseen_chord, 60)