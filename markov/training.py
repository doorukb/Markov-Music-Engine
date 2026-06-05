from __future__ import annotations
from pathlib import Path
from typing import Sequence
from markov.data import collect_chord_sequences
from markov.encoder import (
    ChordToken,
    build_chord_vocabulary,
    build_note_vocabulary,
    chord_vocabulary_inverse,
    encode_chords,
)
from markov.harmony import ChordChain
from markov.matrix import HierarchicalMarkovModel
from markov.melody import MelodyChain
from markov.parser import parse_midi

__all__ = ["train_model", "train_models"]

# train one model per melody order on the same chord vocabulary
# returns (models_by_order, index_to_chord, chord_to_index)
def train_models(paths: Sequence[Path], orders: Sequence[int]) -> tuple[dict[int, HierarchicalMarkovModel], list[ChordToken], dict[ChordToken, int]]:
    unique_orders = sorted(set(orders))
    if not unique_orders:
        raise ValueError("orders must contain at least one melody order")

    chord_sequences = collect_chord_sequences(list(paths))
    if not chord_sequences:
        raise RuntimeError("No chord sequences could be parsed from the corpus.")

    chord_to_index = build_chord_vocabulary(chord_sequences)
    index_to_chord = list(chord_vocabulary_inverse(chord_to_index))
    note_to_index = build_note_vocabulary()

    models: dict[int, HierarchicalMarkovModel] = {}
    for order in unique_orders:
        harmony = ChordChain(vocab_size=len(chord_to_index))
        melody = MelodyChain(order=order)
        model = HierarchicalMarkovModel(harmony=harmony, melody=melody)
        model.train(list(paths), parse_midi, encode_chords, chord_to_index, note_to_index)
        models[order] = model

    return models, index_to_chord, chord_to_index

# train a single-order model (convenience wrapper around train_models)
# returns (model, index_to_chord, chord_to_index)
def train_model(paths: Sequence[Path], order: int) -> tuple[HierarchicalMarkovModel, list[ChordToken], dict[ChordToken, int]]:
    models, index_to_chord, chord_to_index = train_models(paths, [order])
    return models[order], index_to_chord, chord_to_index