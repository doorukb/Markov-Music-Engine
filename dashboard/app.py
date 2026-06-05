from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from config import (
    DEFAULT_N_CHORDS,
    DEFAULT_TEMPO_BPM,
    OUTPUTS_DIR,
    SUPPORTED_STYLES,
)
from main import train_model
from markov.data import collect_chord_sequences, load_corpus
from markov.encoder import (
    build_chord_vocabulary,
    build_note_vocabulary,
    chord_vocabulary_inverse,
    encode_chords,
)
from markov.harmony import ChordChain
from markov.melody import MelodyChain
from markov.parser import parse_midi
from markov.encoder import ChordToken
from markov.generator import Composer, CompositionResult, MultiOrderResult
from markov.matrix import HierarchicalMarkovModel
from markov.analysis import summarise
from dashboard.player import PreparedAudio, audio_widget, prepare_audio
from visualization.plots import (
    plot_metrics_panel,
    plot_stationary_distribution,
    plot_transition_matrix,
    shared_top_chord_indices,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    "last_audio": None,
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
# train the models only when the style, order selection, or single-source mode changes
def _ensure_trained_models(style: str, orders: list[int], single_source: bool) -> None:
    cache_key = _model_cache_key(style, orders, single_source)
    if st.session_state.model_cache_key == cache_key and st.session_state.models is not None and all(order in st.session_state.models for order in orders):
        return

    training_paths = _training_paths(style, single_source)
    models: dict[int, HierarchicalMarkovModel] = {}
    unique_orders = sorted(set(orders))

    if len(unique_orders) > 1:
        chord_sequences = collect_chord_sequences(training_paths)
        if not chord_sequences:
            raise RuntimeError("No chord sequences could be parsed from the training corpus.")
        chord_to_index = build_chord_vocabulary(chord_sequences)
        index_to_chord = list(chord_vocabulary_inverse(chord_to_index))
        note_to_index = build_note_vocabulary()
        for order in unique_orders:
            harmony = ChordChain(vocab_size=len(chord_to_index))
            melody = MelodyChain(order=order)
            model = HierarchicalMarkovModel(harmony=harmony, melody=melody)
            model.train(training_paths, parse_midi, encode_chords, chord_to_index, note_to_index)
            models[order] = model
    else:
        order = unique_orders[0]
        model, resolved_index, _ = train_model(training_paths, order)
        models[order] = model
        index_to_chord = list(resolved_index)

    st.session_state.model_cache_key = cache_key
    st.session_state.models = models
    st.session_state.index_to_chord = index_to_chord

# run the generation
def _run_generation(style: str, orders: list[int], *, n_chords: int, notes_per_chord: int, tempo_bpm: int) -> tuple[MultiOrderResult, dict[int, PreparedAudio]]:
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
    audio_by_order: dict[int, PreparedAudio] = {}
    for order, result, _ in multi.results:
        stem = f"{style}_order{order}"
        audio_by_order[order] = prepare_audio(result, stem)

    return multi, audio_by_order

# generate harmony summaries for a MultiOrderResult
def _summaries_for_result(last_result: MultiOrderResult) -> dict[int, dict]:
    summaries: dict[int, dict] = {}
    index_to_chord = last_result.index_to_chord
    for order, _, model in last_result.results:
        matrix = model.harmony.transition_matrix
        if matrix is None:
            raise RuntimeError(f"Harmony transition matrix is missing for order {order}.")
        summaries[order] = summarise(matrix, index_to_chord)
    return summaries

# calculate the maximum value for the heatmap
def _heatmap_vmax(matrices: Sequence, index_to_chord: Sequence[ChordToken], chord_indices: Sequence[int]) -> float:
    values = []
    for matrix in matrices:
        sub = matrix[np.ix_(chord_indices, chord_indices)]
        values.append(float(sub.max()))
    return max(values) if values else 1e-9

# render a single order column
def _render_order_column(
    *,
    order: int, # the order of the melody
    style: str, # the style of the music
    result: CompositionResult, # the result of the composition
    model: HierarchicalMarkovModel, # the model of the composition
    summary: dict, # the summary of the composition
    audio: PreparedAudio, # the audio of the composition
    index_to_chord: Sequence[ChordToken], # the index to chord of the composition
    baseline_summary: dict | None, # the baseline summary of the composition
    chord_indices: Sequence[int] | None, # the chord indices of the composition
    heatmap_vmax: float | None, # the maximum value for the heatmap
) -> None:
    st.markdown(f"### Order {order}")
    plot_metrics_panel(summary, order, baseline=baseline_summary)

    matrix = model.harmony.transition_matrix
    if matrix is None:
        raise RuntimeError(f"Harmony transition matrix is missing for order {order}")

    fig_heatmap = plot_transition_matrix(
        matrix,
        index_to_chord,
        f"Harmony transitions — {style}",
        chord_indices=chord_indices,
        vmax=heatmap_vmax,
    )
    st.pyplot(fig_heatmap)
    plt.close(fig_heatmap)

    fig_stationary = plot_stationary_distribution(summary["stationary_distribution"], f"Stationary distribution — {style}")
    st.pyplot(fig_stationary)
    plt.close(fig_stationary)

    st.caption(f"{result.metadata.get('n_chords')} chords, {result.metadata.get('notes_per_chord')} notes/chord, {result.tempo_bpm} BPM — MIDI: `{audio.midi_path}`")
    audio_widget(audio.wav_path, f"Order {order}", midi_path=audio.midi_path)

# render the compare results
def _render_compare_results(
    last_result: MultiOrderResult, # the last result of the composition
    last_audio: dict[int, PreparedAudio], # the last audio of the composition
    summaries: dict[int, dict], # the summaries of the composition
) -> None:
    index_to_chord = last_result.index_to_chord
    style = last_result.results[0][1].style
    orders = sorted(order for order, _, _ in last_result.results)
    baseline = summaries.get(1)

    matrices = []
    for order in orders:
        model = next(m for o, _, m in last_result.results if o == order)
        matrix = model.harmony.transition_matrix
        if matrix is None:
            raise RuntimeError(f"Harmony transition matrix is missing for order {order}.")
        matrices.append(matrix)

    chord_indices = shared_top_chord_indices(matrices)
    heatmap_vmax = _heatmap_vmax(matrices, index_to_chord, chord_indices)

    col_left, col_right = st.columns(2)
    columns = [col_left, col_right]
    for col, order in zip(columns, orders):
        result = next(r for o, r, _ in last_result.results if o == order)
        model = next(m for o, _, m in last_result.results if o == order)
        audio = last_audio[order]
        with col:
            _render_order_column(
                order=order,
                style=style,
                result=result,
                model=model,
                summary=summaries[order],
                audio=audio,
                index_to_chord=index_to_chord,
                baseline_summary=baseline if order != 1 else None,
                chord_indices=chord_indices,
                heatmap_vmax=heatmap_vmax,
            )

# render the single order results
def _render_single_order_results(
    last_result: MultiOrderResult, # the last result of the composition
    last_audio: dict[int, PreparedAudio], # the last audio of the composition
    summaries: dict[int, dict], # the summaries of the composition
) -> None:
    order, result, model = last_result.results[0] # the order, result, and model of the composition
    audio = last_audio[order]
    style = result.style
    _render_order_column(
        order=order,
        style=style,
        result=result,
        model=model,
        summary=summaries[order],
        audio=audio,
        index_to_chord=last_result.index_to_chord,
        baseline_summary=None,
        chord_indices=None,
        heatmap_vmax=None,
    )

# render the results placeholder
def _render_results_placeholder() -> None:
    st.subheader("Results")
    last_result: MultiOrderResult | None = st.session_state.last_result
    last_audio: dict[int, PreparedAudio] | None = st.session_state.last_audio

    if last_result is None or last_audio is None:
        st.info("Adjust the sidebar controls and click **Generate** to create music.")
        return

    summaries = _summaries_for_result(last_result)
    compare = len(last_result.results) > 1

    if compare:
        st.markdown("#### Order comparison")
        _render_compare_results(last_result, last_audio, summaries)
    else:
        _render_single_order_results(last_result, last_audio, summaries)

def main() -> None:
    st.set_page_config(
        page_title="Markov Music Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    st.title("Markov Music Engine")
    st.caption("Train hierarchical Markov chains on folk MIDI corpora and sample new chord progressions with melodies.")

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
                multi, audio_by_order = _run_generation(
                    style,
                    orders,
                    n_chords=n_chords,
                    notes_per_chord=notes_per_chord,
                    tempo_bpm=tempo_bpm,
                )
            st.session_state.last_result = multi
            st.session_state.last_audio = audio_by_order
            st.success("Generation complete.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    _render_results_placeholder()


if __name__ == "__main__":
    main()