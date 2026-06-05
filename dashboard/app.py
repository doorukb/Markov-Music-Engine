from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    DEFAULT_N_CHORDS,
    DEFAULT_TEMPO_BPM,
    ORDER3_RESOURCE_MESSAGE,
    ORDER3_WARNING_THRESHOLD,
    OUTPUTS_DIR,
    SUPPORTED_STYLES,
)
from dashboard.playback_ui import PlaybackTrack, render_playback_studio  # noqa: E402
from dashboard.player import PreparedAudio, prepare_audio, prepare_original_audio  # noqa: E402
from markov.analysis import summarise  # noqa: E402
from markov.audio_setup import ensure_soundfont  # noqa: E402
from markov.data import collect_chord_sequences, load_corpus  # noqa: E402
from markov.encoder import ChordToken, build_chord_vocabulary  # noqa: E402
from markov.generator import Composer, CompositionResult, MultiOrderResult  # noqa: E402
from markov.matrix import HierarchicalMarkovModel  # noqa: E402
from markov.playback import (  # noqa: E402
    load_models_for_orders,
    resolve_source,
    save_model_bundle,
    seconds_to_n_chords,
)
from markov.training import train_models  # noqa: E402
from visualization.plots import (  # noqa: E402
    plot_metrics_panel,
    plot_stationary_distribution,
    plot_transition_matrix,
    shared_top_chord_indices,
)

_MELODY_MODES: dict[str, list[int]] = {
    "Order 1": [1],
    "Order 2": [2],
    "Compare orders 1 & 2": [1, 2],
    "Order 3 — Warning: high resource use": [3],
    "All orders (1 + 2 + 3)": [1, 2, 3],
}

_SESSION_DEFAULTS: dict[str, object] = {
    "model_cache_key": None,
    "models": None,
    "index_to_chord": None,
    "source_piece": None,
    "last_result": None,
    "last_audio": None,
    "last_original_audio": None,
    "playback_tracks": None,
    "soundfont_ready": False,
}

def _init_session_state() -> None:
    for key, value in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

# generate a cache key for the model based on the style, orders, single source, and load model
def _model_cache_key(style: str, orders: Sequence[int], single_source: bool, load_model: str | None) -> tuple:
    return (style, tuple(sorted(set(orders))), single_source, load_model or "")

# resolve the orders to use for generation based on the melody mode and confirm order 3
def _resolve_orders(melody_mode: str, *, confirm_order3: bool) -> list[int] | None:
    orders = list(_MELODY_MODES[melody_mode])
    if ORDER3_WARNING_THRESHOLD in orders and not confirm_order3:
        st.error("Confirm order 3 (high resource use) before generating.")
        return None
    return orders

# resolve the training paths based on the style, single source, and source text
def _training_paths(style: str, single_source: bool, source_text: str | None) -> list[Path]:
    corpus_paths = list(load_corpus(style))
    if not corpus_paths:
        raise RuntimeError(f"No MIDI files found for style {style!r}.")
    source_piece = resolve_source(source_text, style, corpus_paths)
    st.session_state.source_piece = source_piece
    if single_source:
        return [source_piece]
    return corpus_paths

# ensure the models are trained and cached
def _ensure_trained_models(style: str, orders: list[int], single_source: bool, source_text: str | None, load_model_dir: Path | None) -> None:
    load_key = str(load_model_dir) if load_model_dir else ""
    cache_key = _model_cache_key(style, orders, single_source, load_key)
    if st.session_state.model_cache_key == cache_key and st.session_state.models is not None and all(order in st.session_state.models for order in orders):
        return

    if load_model_dir is not None:
        if not load_model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {load_model_dir}")
        models, index_to_chord = load_models_for_orders(load_model_dir, orders)
    else:
        training_paths = _training_paths(style, single_source, source_text)
        models, index_to_chord, _ = train_models(training_paths, orders)

    st.session_state.model_cache_key = cache_key
    st.session_state.models = models
    st.session_state.index_to_chord = index_to_chord

# synthesize all audio for the multi-order result
def _synthesize_all_audio(style: str, multi: MultiOrderResult, *, progress: st.delta_generator.DeltaGenerator) -> tuple[dict[int, PreparedAudio], PreparedAudio | None]:
    audio_by_order: dict[int, PreparedAudio] = {}
    n_total = len(multi.results) + 1
    step = 0

    source = st.session_state.get("source_piece")
    original: PreparedAudio | None = None
    if source is not None:
        progress.progress(int(100 * step / n_total), text="Synthesizing Original…")
        original = prepare_original_audio(Path(source), f"{style}_original")
        step += 1

    for order, result, _ in multi.results:
        progress.progress(int(100 * step / n_total), text=f"Synthesizing Order {order}…")
        stem = f"{style}_order{order}"
        audio_by_order[order] = prepare_audio(result, stem)
        step += 1

    progress.progress(100, text="Audio ready")
    return audio_by_order, original

# build the playback tracks for the multi-order result
def _build_playback_tracks(original: PreparedAudio | None, audio_by_order: dict[int, PreparedAudio], style: str) -> list[PlaybackTrack]:
    tracks: list[PlaybackTrack] = []
    if original is not None:
        tracks.append(
            PlaybackTrack(
                label="Original",
                midi_path=original.midi_path,
                wav_path=original.wav_path,
                duration_seconds=original.duration_seconds,
            )
        )
    for order in sorted(audio_by_order):
        assets = audio_by_order[order]
        tracks.append(
            PlaybackTrack(
                label=f"{style} — Order {order}",
                midi_path=assets.midi_path,
                wav_path=assets.wav_path,
                duration_seconds=assets.duration_seconds,
            )
        )
    return tracks

def _run_generation(
    style: str, # the style to use for generation
    orders: list[int],
    *,
    n_chords: int, # the number of chords to generate
    notes_per_chord: int, # the number of notes to generate per chord
    tempo_bpm: int, # the tempo to use for generation
    save_model_dir: Path | None, # the path to save the model
    load_model_dir: Path | None, # the path to load the model
    single_source: bool, # whether to use a single source
    source_text: str | None, # the text of the source piece
) -> tuple[MultiOrderResult, dict[int, PreparedAudio], PreparedAudio | None]:
    if load_model_dir is None:
        _ensure_trained_models(style, orders, single_source, source_text, None)
    else:
        _ensure_trained_models(style, orders, single_source, source_text, load_model_dir)
        corpus_paths = list(load_corpus(style))
        st.session_state.source_piece = resolve_source(source_text, style, corpus_paths)

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

    if save_model_dir is not None and load_model_dir is None:
        training_paths = _training_paths(style, single_source, source_text)
        chord_sequences = collect_chord_sequences(training_paths)
        chord_to_index = build_chord_vocabulary(chord_sequences)
        for order, _, model in multi.results:
            save_model_bundle(model, save_model_dir / f"order{order}", chord_to_index)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0, text="Preparing audio…")
    audio_by_order, original = _synthesize_all_audio(style, multi, progress=progress)
    return multi, audio_by_order, original

# build the summaries for the multi-order result
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
def _heatmap_vmax(matrices: Sequence, chord_indices: Sequence[int]) -> float:
    values = []
    for matrix in matrices:
        sub = matrix[np.ix_(chord_indices, chord_indices)]
        values.append(float(sub.max()))
    return max(values) if values else 1e-9

# render the order column for the multi-order result
def _render_order_column(
    *,
    order: int, # the order to render
    style: str, # the style to use for generation
    result: CompositionResult, # the result to render
    model: HierarchicalMarkovModel, # the model to use for generation
    summary: dict, # the summary to render
    audio: PreparedAudio, # the audio to render
    index_to_chord: Sequence[ChordToken], # the mapping of chord indices to chord tokens
    baseline_summary: dict | None, # the baseline summary to render
    chord_indices: Sequence[int] | None, # the chord indices to render
    heatmap_vmax: float | None, # the maximum value for the heatmap
) -> None:
    st.markdown(f"### Order {order}")
    plot_metrics_panel(summary, order, baseline=baseline_summary)

    matrix = model.harmony.transition_matrix
    if matrix is None:
        raise RuntimeError(f"Harmony transition matrix is missing for order {order}.")

    fig_heatmap = plot_transition_matrix(
        matrix,
        index_to_chord,
        f"Harmony transitions — {style}",
        chord_indices=chord_indices,
        vmax=heatmap_vmax,
    )
    st.pyplot(fig_heatmap)
    plt.close(fig_heatmap)

    fig_stationary = plot_stationary_distribution(
        summary["stationary_distribution"],
        f"Stationary distribution — {style}",
    )
    st.pyplot(fig_stationary)
    plt.close(fig_stationary)

    st.caption(
        f"{result.metadata.get('n_chords')} chords, "
        f"{result.metadata.get('notes_per_chord')} notes/chord, "
        f"{result.tempo_bpm} BPM"
    )

# render the analysis for the multi-order result
def _render_analysis(last_result: MultiOrderResult) -> None:
    summaries = _summaries_for_result(last_result)
    compare = len(last_result.results) > 1
    index_to_chord = last_result.index_to_chord
    style = last_result.results[0][1].style
    orders = sorted(order for order, _, _ in last_result.results)

    if compare:
        st.markdown("#### Order comparison")
        baseline = summaries.get(1)
        matrices = []
        for order in orders:
            model = next(m for o, _, m in last_result.results if o == order)
            matrix = model.harmony.transition_matrix
            if matrix is None:
                raise RuntimeError(f"Harmony transition matrix is missing for order {order}.")
            matrices.append(matrix)
        chord_indices = shared_top_chord_indices(matrices)
        heatmap_vmax = _heatmap_vmax(matrices, chord_indices)
        columns = st.columns(len(orders))
        for col, order in zip(columns, orders):
            result = next(r for o, r, _ in last_result.results if o == order)
            model = next(m for o, _, m in last_result.results if o == order)
            audio = st.session_state.last_audio[order]
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
    else:
        order, result, model = last_result.results[0]
        audio = st.session_state.last_audio[order]
        _render_order_column(
            order=order,
            style=style,
            result=result,
            model=model,
            summary=summaries[order],
            audio=audio,
            index_to_chord=index_to_chord,
            baseline_summary=None,
            chord_indices=None,
            heatmap_vmax=None,
        )

# render the results for the multi-order result
def _render_results() -> None:
    last_result: MultiOrderResult | None = st.session_state.last_result
    last_audio: dict[int, PreparedAudio] | None = st.session_state.last_audio
    tracks: list[PlaybackTrack] | None = st.session_state.playback_tracks

    if last_result is None or last_audio is None or tracks is None:
        st.info("Configure the sidebar and click **Generate** to create music.")
        return

    _render_analysis(last_result)
    render_playback_studio(tracks)

    source = st.session_state.get("source_piece")
    if source is not None:
        st.caption(f"Source piece: `{source}`")


def main() -> None:
    st.set_page_config(
        page_title="Markov Music Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    if not st.session_state.soundfont_ready:
        try:
            ensure_soundfont()
            st.session_state.soundfont_ready = True
        except Exception as exc:
            st.sidebar.warning(f"Soundfont setup: {exc}")

    st.title("Markov Music Engine")
    st.caption("Interactive dashboard with the same capabilities as the CLI — train, generate, analyze, and play.")

    with st.sidebar:
        st.header("Generation")
        style = st.selectbox("Style", SUPPORTED_STYLES, index=0)

        melody_mode = st.radio(
            "Melody mode",
            list(_MELODY_MODES.keys()),
            index=2,
        )
        if ORDER3_WARNING_THRESHOLD in _MELODY_MODES[melody_mode]:
            st.caption(ORDER3_RESOURCE_MESSAGE.strip())
        confirm_order3 = st.checkbox(
            "Proceed with order 3 (high resource use)",
            value=False,
            help="Equivalent to CLI --yes for order-3 runs.",
        )

        st.subheader("Piece length")
        length_mode = st.radio("Length mode", ["Chord count", "Target duration"], index=0)
        tempo_bpm = st.slider("Tempo (BPM)", 40, 220, DEFAULT_TEMPO_BPM)
        if length_mode == "Chord count":
            n_chords = st.slider("Number of chords", 1, 64, DEFAULT_N_CHORDS)
            duration_seconds: float | None = None
        else:
            duration_seconds = st.slider("Target duration (seconds)", 5.0, 180.0, 30.0, 1.0)
            n_chords = seconds_to_n_chords(duration_seconds, tempo_bpm)
            st.caption(f"≈ {n_chords} chords at {tempo_bpm} BPM")

        notes_per_chord = st.slider("Notes per chord", 1, 16, 4)

        st.subheader("Training")
        single_source = st.checkbox("Single source (--single-source)", value=False)
        source_text = st.text_input(
            "Source (--source)",
            placeholder="Path to .mid or style keyword (jazz, pop, classical)",
            help="Leave empty for a random piece from the selected style.",
        )

        st.subheader("Model I/O")
        load_model_text = st.text_input("Load model directory (--load-model)", value="")
        save_model_text = st.text_input("Save model directory (--save-model)", value="")

        st.divider()
        generate = st.button("Generate", type="primary", use_container_width=True)

    if generate:
        orders = _resolve_orders(melody_mode, confirm_order3=confirm_order3)
        if orders is None:
            st.stop()

        load_dir = Path(load_model_text.strip()) if load_model_text.strip() else None
        save_dir = Path(save_model_text.strip()) if save_model_text.strip() else None

        try:
            with st.spinner("Training / loading model(s)…"):
                multi, audio_by_order, original = _run_generation(
                    style,
                    orders,
                    n_chords=n_chords,
                    notes_per_chord=notes_per_chord,
                    tempo_bpm=tempo_bpm,
                    save_model_dir=save_dir,
                    load_model_dir=load_dir,
                    single_source=single_source,
                    source_text=source_text.strip() or None,
                )
            st.session_state.last_result = multi
            st.session_state.last_audio = audio_by_order
            st.session_state.last_original_audio = original
            st.session_state.playback_tracks = _build_playback_tracks(original, audio_by_order, style)
            st.toast("Generation complete.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    _render_results()

if __name__ == "__main__":
    main()