from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Sequence
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    DEFAULT_N_CHORDS,
    DEFAULT_TEMPO_BPM,
    OUTPUTS_DIR,
    SUPPORTED_STYLES,
)
from main import train_model  # noqa: E402
from markov.data import load_corpus  # noqa: E402
from markov.encoder import ChordToken  # noqa: E402
from markov.generator import Composer, MultiOrderResult  # noqa: E402
from markov.matrix import HierarchicalMarkovModel  # noqa: E402
from markov.renderer import render_midi  # noqa: E402

_ORDER_OPTIONS: dict[str, list[int]] = {
    "Order 1": [1],
    "Order 2": [2],
    "Both (compare)": [1, 2],
}

_SESSION_DEFAULTS: dict[str, object] = {
    "model_cache_key": None,
    "models": None,
    "index_to_chord": None,
    "source_piece": None,
    "last_result": None,
    "last_midi_paths": None,
}

# initialize the session state
def _init_session_state() -> None:
    for key, value in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


# generate a cache key for the model
def _model_cache_key(style: str, orders: Sequence[int], single_source: bool) -> tuple:
    return (style, tuple(sorted(set(orders))), single_source)

# resolve the orders to generate
def _resolve_orders(order_label: str) -> list[int]:
    return list(_ORDER_OPTIONS[order_label])

# get the training paths for the model
def _training_paths(style: str, single_source: bool) -> list[Path]:
    corpus_paths = list(load_corpus(style))
    if not corpus_paths:
        raise RuntimeError(f"No MIDI files found for style {style!r}.")
    if single_source:
        source = random.choice(corpus_paths)
        st.session_state.source_piece = source
        return [source]
    st.session_state.source_piece = None
    return corpus_paths

# ensure the models are trained
def _ensure_trained_models(style: str, orders: list[int], single_source: bool) -> None:
    """Train models only when style, order selection, or single-source mode changes."""
    cache_key = _model_cache_key(style, orders, single_source)
    if (
        st.session_state.model_cache_key == cache_key
        and st.session_state.models is not None
        and all(order in st.session_state.models for order in orders)
    ):
        return

    training_paths = _training_paths(style, single_source)
    models: dict[int, HierarchicalMarkovModel] = {}
    index_to_chord: list[ChordToken] | None = None

    for order in sorted(set(orders)):
        model, resolved_index, _ = train_model(training_paths, order)
        models[order] = model
        if index_to_chord is None:
            index_to_chord = list(resolved_index)

    st.session_state.model_cache_key = cache_key
    st.session_state.models = models
    st.session_state.index_to_chord = index_to_chord

# run the generation
def _run_generation(
    style: str,
    orders: list[int],
    *,
    n_chords: int,
    notes_per_chord: int,
    tempo_bpm: int,
) -> tuple[MultiOrderResult, dict[int, Path]]:
    models: dict[int, HierarchicalMarkovModel] = st.session_state.models
    index_to_chord = st.session_state.index_to_chord

    multi = Composer.compose_orders(
        style,
        n_chords,
        orders,
        notes_per_chord=notes_per_chord,
        tempo_bpm=tempo_bpm,
        models=models,
        index_to_chord=index_to_chord,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    midi_paths: dict[int, Path] = {}
    for order, result, _ in multi.results:
        stem = f"{style}_order{order}"
        midi_paths[order] = render_midi(result, f"{stem}.mid")

    return multi, midi_paths

# render the results placeholder
def _render_results_placeholder() -> None:
    st.subheader("Results")
    last_result: MultiOrderResult | None = st.session_state.last_result
    midi_paths: dict[int, Path] | None = st.session_state.last_midi_paths

    if last_result is None or midi_paths is None:
        st.info("Adjust the sidebar controls and click **Generate** to create music.")
        return

    for order, result, _ in last_result.results:
        path = midi_paths.get(order)
        st.markdown(f"**Order {order}** — {result.metadata.get('n_chords')} chords, "
                    f"{result.metadata.get('notes_per_chord')} notes/chord, "
                    f"{result.tempo_bpm} BPM")
        if path is not None:
            st.caption(f"MIDI: `{path}`")
        st.markdown("_Plots and audio players will appear here._")

def main() -> None:
    st.set_page_config(
        page_title="Markov Music Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    st.title("Markov Music Engine")
    st.caption(
        "Train hierarchical Markov chains on folk MIDI corpora and sample new chord progressions with melodies."
    )

    with st.sidebar:
        st.header("Controls")
        style = st.selectbox("Style", SUPPORTED_STYLES, index=0)
        tempo_bpm = st.slider(
            "Tempo (BPM)",
            min_value=40,
            max_value=220,
            value=DEFAULT_TEMPO_BPM,
            step=1,
        )
        order_label = st.radio(
            "Melody order",
            list(_ORDER_OPTIONS.keys()),
            index=2,
            help="Order 1: chord-conditioned notes. Order 2: adds previous-note context. Both runs a side-by-side comparison.",
        )
        n_chords = st.slider(
            "Number of chords",
            min_value=1,
            max_value=64,
            value=DEFAULT_N_CHORDS,
            step=1,
            help="Each chord is one 4/4 measure.",
        )
        notes_per_chord = st.slider(
            "Notes per chord",
            min_value=1,
            max_value=16,
            value=4,
            step=1,
        )
        single_source = st.checkbox(
            "Single source",
            value=False,
            help="Train on one randomly chosen piece from the style corpus (matches CLI --single-source).",
        )
        generate = st.button("Generate", type="primary", use_container_width=True)

    orders = _resolve_orders(order_label)

    if generate:
        try:
            with st.spinner("Training model(s)…"):
                _ensure_trained_models(style, orders, single_source)
            with st.spinner("Generating composition(s)…"):
                multi, midi_paths = _run_generation(
                    style,
                    orders,
                    n_chords=n_chords,
                    notes_per_chord=notes_per_chord,
                    tempo_bpm=tempo_bpm,
                )
            st.session_state.last_result = multi
            st.session_state.last_midi_paths = midi_paths
            st.success("Generation complete.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    _render_results_placeholder()


if __name__ == "__main__":
    main()