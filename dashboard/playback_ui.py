from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dashboard.player import get_last_synthesis_error, wav_to_base64
from markov.audio_setup import check_audio_setup

__all__ = [
    "PlaybackTrack",
    "render_playback_studio",
]

MAX_WAV_EMBED_BYTES = 15 * 1024 * 1024


# playback track data class
@dataclass(frozen=True)
class PlaybackTrack:
    label: str  # the label of the track
    midi_path: Path | None  # the path to the MIDI file
    wav_path: Path | None  # the path to the WAV file
    duration_seconds: float  # the duration of the track in seconds


# --------------------------------------------------------------------------- #
# Shared in-iframe audio player
#
# Each components.html() call renders an isolated iframe, so a single player
# template with fixed element ids is reused for both the per-track player and
# the full-session playlist. The only difference is whether the "Skip" control
# and auto-advance behaviour are enabled (session mode). The CSS and JS live in
# brace-free constants so they can be interpolated without f-string escaping.
# --------------------------------------------------------------------------- #

_PLAYER_CSS = """
* { box-sizing: border-box; }
body { margin: 0; }
.mme {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: #161A23;
  border: 1px solid #2A2F3C;
  border-radius: 12px;
  padding: 14px 16px;
  color: #E6E6F0;
}
.mme-now { font-weight: 600; font-size: 0.98rem; margin-bottom: 10px; color: #CFCFE8; }
.mme-bar {
  width: 100%; height: 8px; border: none; display: block;
  appearance: none; -webkit-appearance: none;
  border-radius: 999px; overflow: hidden; background: #2A2F3C;
}
.mme-bar::-webkit-progress-bar { background: #2A2F3C; border-radius: 999px; }
.mme-bar::-webkit-progress-value { background: linear-gradient(90deg,#7C5CFF,#9D7CFF); border-radius: 999px; }
.mme-bar::-moz-progress-bar { background: #7C5CFF; border-radius: 999px; }
.mme-seek { width: 100%; margin: 10px 0 6px; accent-color: #7C5CFF; }
.mme-time { color: #8A8AA3; font-variant-numeric: tabular-nums; font-size: 0.85rem; margin-bottom: 12px; }
.mme-controls { display: flex; gap: 8px; flex-wrap: wrap; }
.mme button {
  font: inherit; font-weight: 600; cursor: pointer; color: #fff;
  background: #7C5CFF; border: none; padding: 8px 16px;
  border-radius: 8px; transition: background .15s ease;
}
.mme button:hover { background: #6A48F0; }
.mme button.secondary { background: #2A2F3C; color: #D6D6E8; }
.mme button.secondary:hover { background: #353B4A; }
.mme-missing { color: #8A8AA3; font-style: italic; font-family: system-ui, sans-serif; }
"""

_PLAYER_JS = """
(function () {
  const cfg = JSON.parse(document.getElementById('mme-cfg').textContent);
  const items = cfg.items || [];
  const session = !!cfg.session;
  if (!items.length) return;

  const audio = document.getElementById('mme-audio');
  const bar = document.getElementById('mme-bar');
  const seek = document.getElementById('mme-seek');
  const now = document.getElementById('mme-now');
  const timeEl = document.getElementById('mme-time');
  const playBtn = document.getElementById('mme-play');
  const pauseBtn = document.getElementById('mme-pause');
  const skipBtn = document.getElementById('mme-skip');

  let index = 0;
  let seeking = false;
  let started = false;

  function fmt(s) { return (s || 0).toFixed(1) + 's'; }
  function total() { return Math.max(audio.duration || 0, items[index].duration || 0, 0.001); }

  function setNow() {
    if (!session) { now.textContent = items[index].label; return; }
    now.textContent = (started ? 'Now playing: ' : 'Up next: ') +
      items[index].label + '  (' + (index + 1) + '/' + items.length + ')';
  }

  function syncUi() {
    const t = total();
    const elapsed = audio.currentTime || 0;
    if (!seeking) {
      bar.value = Math.min(100, (elapsed / t) * 100);
      seek.value = Math.min(1000, Math.round((elapsed / t) * 1000));
    }
    timeEl.textContent = fmt(elapsed) + ' / ' + fmt(t);
  }

  function load(i) {
    index = i;
    audio.src = items[i].src;
    bar.value = 0;
    seek.value = 0;
    setNow();
    syncUi();
  }

  audio.addEventListener('timeupdate', syncUi);
  audio.addEventListener('loadedmetadata', syncUi);
  audio.addEventListener('ended', () => {
    if (session && index + 1 < items.length) {
      load(index + 1);
      audio.play();
    } else {
      started = false;
      playBtn.textContent = session ? 'Play session' : 'Play';
      now.textContent = session ? 'Session complete' : items[index].label;
    }
  });

  seek.addEventListener('input', () => {
    seeking = true;
    const t = total();
    const ratio = Number(seek.value) / 1000;
    bar.value = ratio * 100;
    timeEl.textContent = fmt(ratio * t) + ' / ' + fmt(t);
  });
  seek.addEventListener('change', () => {
    audio.currentTime = (Number(seek.value) / 1000) * total();
    seeking = false;
    syncUi();
  });

  playBtn.onclick = () => {
    if (session && !started) { load(0); }
    started = true;
    setNow();
    audio.play();
    if (session) { playBtn.textContent = 'Restart session'; }
  };
  pauseBtn.onclick = () => { audio.pause(); };
  if (skipBtn) {
    skipBtn.onclick = () => {
      if (index + 1 < items.length) {
        started = true;
        load(index + 1);
        audio.play();
      } else {
        audio.pause();
        started = false;
        now.textContent = 'Session complete';
        playBtn.textContent = 'Play session';
      }
    };
  }

  load(0);
})();
"""


# build the full HTML for the shared audio player
def _player_html(items: list[dict], *, session: bool) -> str:
    cfg = json.dumps({"items": items, "session": session})
    play_label = "Play session" if session else "Play"
    skip_button = '<button id="mme-skip" class="secondary" type="button">Skip</button>' if session else ""
    return f"""
<style>{_PLAYER_CSS}</style>
<div class="mme">
  <div id="mme-now" class="mme-now"></div>
  <progress id="mme-bar" class="mme-bar" value="0" max="100"></progress>
  <input id="mme-seek" class="mme-seek" type="range" min="0" max="1000" value="0" aria-label="Seek">
  <div id="mme-time" class="mme-time">0.0s / 0.0s</div>
  <div class="mme-controls">
    <button id="mme-play" type="button">{play_label}</button>
    <button id="mme-pause" class="secondary" type="button">Pause</button>
    {skip_button}
  </div>
  <audio id="mme-audio" preload="metadata"></audio>
</div>
<script id="mme-cfg" type="application/json">{cfg}</script>
<script>{_PLAYER_JS}</script>
"""


# the audio-unavailable placeholder shown inside an iframe
def _unavailable_html(message: str) -> str:
    return f'<style>{_PLAYER_CSS}</style><div class="mme"><p class="mme-missing">{message}</p></div>'


# build a data: URI for a WAV file, or None if it exceeds the embed limit
def _wav_data_uri(wav_path: Path) -> str | None:
    if wav_path.stat().st_size > MAX_WAV_EMBED_BYTES:
        return None
    return f"data:audio/wav;base64,{wav_to_base64(wav_path)}"


# collect the tracks that can be embedded for in-browser playback
def _playable_items(tracks: list[PlaybackTrack]) -> list[dict]:
    items: list[dict] = []
    for track in tracks:
        if not track.wav_path or not track.wav_path.is_file():
            continue
        uri = _wav_data_uri(track.wav_path)
        if uri is None:
            continue
        items.append({"label": track.label, "src": uri, "duration": track.duration_seconds})
    return items


# render the single-track player plus its download buttons
def _render_track_player(track: PlaybackTrack) -> None:
    items = _playable_items([track])
    if items:
        components.html(_player_html(items, session=False), height=180)
    elif track.wav_path and track.wav_path.is_file():
        st.warning(
            f"WAV for {track.label} is too large for in-browser playback "
            f"(>{MAX_WAV_EMBED_BYTES // (1024 * 1024)} MB). Use the downloads below."
        )

    if track.midi_path and track.midi_path.is_file():
        st.download_button(
            "Download MIDI",
            data=track.midi_path.read_bytes(),
            file_name=track.midi_path.name,
            mime="audio/midi",
            key=f"midi_dl_{track.label}",
            use_container_width=True,
        )
    if track.wav_path and track.wav_path.is_file():
        st.download_button(
            "Download WAV",
            data=track.wav_path.read_bytes(),
            file_name=track.wav_path.name,
            mime="audio/wav",
            key=f"wav_dl_{track.label}",
            use_container_width=True,
        )


# render the per-track unavailable state (no WAV produced)
def _render_track_unavailable(track: PlaybackTrack) -> None:
    setup = check_audio_setup()
    err = get_last_synthesis_error()
    if err:
        st.warning(err)
    elif not setup["soundfont_present"]:
        st.warning("Soundfont missing — run `make setup-audio` or restart the dashboard.")
    elif not setup["can_synthesize_wav"]:
        st.warning("WAV synthesis unavailable — run `make setup-audio` or restart after first-run download.")

    if track.midi_path and track.midi_path.is_file():
        st.download_button(
            "Download MIDI",
            data=track.midi_path.read_bytes(),
            file_name=track.midi_path.name,
            mime="audio/midi",
            key=f"midi_only_{track.label}",
            use_container_width=True,
        )
    else:
        st.warning("No audio could be prepared for this track.")


# render the audio setup status strip
def _render_audio_setup_strip() -> None:
    setup = check_audio_setup()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Soundfont", "Ready" if setup["soundfont_present"] else "Missing")
    with c2:
        st.metric("FluidSynth", "Found" if setup["fluidsynth_cli"] else "Not available")
    with c3:
        st.metric("WAV engine", "OK" if setup["can_synthesize_wav"] else "Unavailable")
    if not setup["can_synthesize_wav"]:
        st.caption(
            "Dashboard playback needs a soundfont and FluidSynth. On first launch the app "
            "downloads both automatically, or run `make setup-audio`."
        )


# render the full session player (Original -> generated orders)
def _render_session_player(tracks: list[PlaybackTrack]) -> None:
    st.markdown("##### Play full session")
    st.caption("Original → generated orders, in sequence (the dashboard equivalent of CLI `--play`).")
    items = _playable_items(tracks)
    if items:
        components.html(_player_html(items, session=True), height=210)
    else:
        components.html(_unavailable_html("No WAV tracks available for session playback yet."), height=90)


# render the playback studio (setup strip + session player + per-track players)
def render_playback_studio(tracks: list[PlaybackTrack]) -> None:
    st.markdown("#### 🎧 Playback studio")
    _render_audio_setup_strip()
    _render_session_player(tracks)

    if not tracks:
        return

    st.markdown("##### Individual tracks")
    cols = st.columns(len(tracks))
    for col, track in zip(cols, tracks):
        with col:
            st.markdown(f"**{track.label}**")
            if track.duration_seconds > 0:
                st.caption(f"~{track.duration_seconds:.1f}s")
            if track.wav_path and track.wav_path.is_file():
                _render_track_player(track)
            else:
                _render_track_unavailable(track)
