from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dashboard.player import PreparedAudio, audio_widget, wav_to_base64
from markov.audio_setup import check_audio_setup

__all__ = ["PlaybackTrack", "render_audio_setup_strip", "render_playback_studio", "render_session_player"]

# playback track data class
@dataclass(frozen=True)
class PlaybackTrack:
    label: str
    midi_path: Path | None
    wav_path: Path | None
    duration_seconds: float

# render the audio setup strip
def render_audio_setup_strip() -> None:
    setup = check_audio_setup()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Soundfont", "Ready" if setup["soundfont_present"] else "Missing")
    with c2:
        st.metric("FluidSynth CLI", "Found" if setup["fluidsynth_cli"] else "Not on PATH")
    with c3:
        st.metric("WAV engine", "OK" if setup["can_synthesize_wav"] else "Unavailable")
    if not setup["can_synthesize_wav"]:
        st.caption("In-browser playback needs a soundfont and FluidSynth (CLI or pyfluidsynth). The dashboard downloads a soundfont automatically on first generate.")

# generate the HTML for the playlist. This is used to display the playlist in the browser
def _playlist_html(tracks: list[PlaybackTrack]) -> str:
  playable = [t for t in tracks if t.wav_path and t.wav_path.is_file()]
  if not playable:
    return "<p>No WAV tracks available for session playback.</p>"

  items = []
  for track in playable:
    b64 = wav_to_base64(track.wav_path)  # type: ignore[arg-type]
    items.append(
      {
        "label": track.label,
        "src": f"data:audio/wav;base64,{b64}",
        "duration": track.duration_seconds,
      }
    )
  data_json = json.dumps(items)
  return f"""
<div id="mme-player" style="font-family: system-ui, sans-serif; padding: 0.5rem 0;">
  <div id="mme-now" style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">Ready</div>
  <progress id="mme-bar" value="0" max="100" style="width: 100%; height: 1.25rem;"></progress>
  <div id="mme-time" style="margin: 0.35rem 0 0.75rem; color: #555;">0.0s / 0.0s</div>
  <button id="mme-play" style="margin-right: 0.5rem; padding: 0.4rem 1rem;">Play session</button>
  <button id="mme-skip" style="padding: 0.4rem 1rem;">Skip track</button>
  <audio id="mme-audio" style="display:none;"></audio>
</div>
<script>
(function() {{
  const tracks = {data_json};
  let index = 0;
  let playing = false;
  const audio = document.getElementById('mme-audio');
  const bar = document.getElementById('mme-bar');
  const now = document.getElementById('mme-now');
  const time = document.getElementById('mme-time');
  const playBtn = document.getElementById('mme-play');
  const skipBtn = document.getElementById('mme-skip');

  function fmt(s) {{ return s.toFixed(1) + 's'; }}

  function load(i) {{
    const t = tracks[i];
    audio.src = t.src;
    now.textContent = 'Now playing: ' + t.label + ' (' + (i+1) + '/' + tracks.length + ')';
    bar.value = 0;
    bar.max = 100;
    time.textContent = '0.0s / ' + fmt(t.duration);
  }}

  function tick() {{
    if (!playing || !audio.duration) return;
    const t = tracks[index];
    const elapsed = audio.currentTime;
    const total = Math.max(audio.duration, t.duration, 0.001);
    bar.value = Math.min(100, (elapsed / total) * 100);
    time.textContent = fmt(elapsed) + ' / ' + fmt(total);
  }}

  audio.addEventListener('timeupdate', tick);
  audio.addEventListener('ended', () => {{
    index += 1;
    if (index >= tracks.length) {{
      playing = false;
      now.textContent = 'Session complete';
      playBtn.textContent = 'Play session';
      return;
    }}
    load(index);
    audio.play();
  }});

  playBtn.onclick = () => {{
    if (!tracks.length) return;
    if (!playing) {{
      index = 0;
      playing = true;
      load(0);
      audio.play();
      playBtn.textContent = 'Restart session';
    }} else {{
      index = 0;
      load(0);
      audio.play();
    }}
  }};

  skipBtn.onclick = () => {{
    if (!playing) return;
    audio.pause();
    index += 1;
    if (index >= tracks.length) {{
      playing = false;
      now.textContent = 'Session complete';
      playBtn.textContent = 'Play session';
      return;
    }}
    load(index);
    audio.play();
  }};
}})();
</script>
"""

# render the session player for the multi-order result
def render_session_player(tracks: list[PlaybackTrack]) -> None:
    st.markdown("##### Play full session")
    st.caption("Original → generated orders (same sequence as CLI `--play`).")
    components.html(_playlist_html(tracks), height=200)

# render the playback studio for the multi-order result
def render_playback_studio(tracks: list[PlaybackTrack]) -> None:
    st.markdown("#### Playback studio")
    render_audio_setup_strip()
    render_session_player(tracks)

    if not tracks:
        return
    cols = st.columns(len(tracks))
    for col, track in zip(cols, tracks):
        with col:
            st.markdown(f"**{track.label}**")
            if track.duration_seconds > 0:
                st.caption(f"~{track.duration_seconds:.1f}s")
            audio_widget(track.wav_path, track.label, midi_path=track.midi_path)