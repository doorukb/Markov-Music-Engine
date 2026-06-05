from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dashboard.player import get_last_synthesis_error, wav_to_base64
from markov.audio_setup import check_audio_setup

__all__ = [
    "PlaybackTrack",
    "render_audio_setup_strip",
    "render_playback_studio",
    "render_session_player",
    "render_track_player",
    "track_player_html",
    "session_playlist_html",
    "session_playlist_tracks",
    "sanitize_dom_id",
]

MAX_WAV_EMBED_BYTES = 15 * 1024 * 1024

# playback track data class
@dataclass(frozen=True)
class PlaybackTrack:
    label: str # the label of the track
    midi_path: Path | None # the path to the MIDI file
    wav_path: Path | None # the path to the WAV file
    duration_seconds: float # the duration of the track in seconds

# sanitize the DOM ID for the track
def sanitize_dom_id(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", label.strip().lower()).strip("-")
    return slug or "track"

# get the playable tracks for the session playlist
def session_playlist_tracks(tracks: list[PlaybackTrack]) -> list[PlaybackTrack]:
    playable: list[PlaybackTrack] = []
    for track in tracks:
        if not track.wav_path or not track.wav_path.is_file():
            continue
        if track.wav_path.stat().st_size > MAX_WAV_EMBED_BYTES:
            continue
        playable.append(track)
    return playable

# format the duration of the track in seconds
def _fmt_duration(seconds: float) -> str:
    return f"{max(seconds, 0.0):.1f}s"

# generate the HTML for the track player
def track_player_html(*, dom_id: str, label: str, audio_src: str | None, duration_seconds: float) -> str:
    if not audio_src:
        return (
            f'<p id="{dom_id}-missing" style="font-family: system-ui, sans-serif; color: #666;">'
            "WAV not available — use Download MIDI below."
            "</p>"
        )

    total_hint = max(duration_seconds, 0.001)
    return f"""
<div id="{dom_id}" class="mme-track-player" style="font-family: system-ui, sans-serif; padding: 0.25rem 0;">
  <div style="font-weight: 600; margin-bottom: 0.35rem;">{label}</div>
  <progress id="{dom_id}-bar" value="0" max="100" style="width: 100%; height: 1rem;"></progress>
  <input id="{dom_id}-seek" type="range" min="0" max="1000" value="0"
         style="width: 100%; margin: 0.35rem 0;" aria-label="Seek">
  <div id="{dom_id}-time" style="margin: 0.25rem 0 0.5rem; color: #555;">0.0s / {_fmt_duration(total_hint)}</div>
  <button id="{dom_id}-play" type="button" style="margin-right: 0.5rem; padding: 0.35rem 0.9rem;">Play</button>
  <button id="{dom_id}-pause" type="button" style="padding: 0.35rem 0.9rem;">Pause</button>
  <audio id="{dom_id}-audio" preload="metadata" src="{audio_src}" style="display:none;"></audio>
</div>
<script>
(function() {{
  const domId = {json.dumps(dom_id)};
  const fallbackDuration = {total_hint};
  const audio = document.getElementById(domId + '-audio');
  const bar = document.getElementById(domId + '-bar');
  const seek = document.getElementById(domId + '-seek');
  const time = document.getElementById(domId + '-time');
  const playBtn = document.getElementById(domId + '-play');
  const pauseBtn = document.getElementById(domId + '-pause');
  let seeking = false;

  function total() {{
    return Math.max(audio.duration || 0, fallbackDuration, 0.001);
  }}

  function fmt(s) {{ return s.toFixed(1) + 's'; }}

  function syncUi() {{
    const t = total();
    const elapsed = audio.currentTime || 0;
    if (!seeking) {{
      bar.value = Math.min(100, (elapsed / t) * 100);
      seek.value = Math.min(1000, Math.round((elapsed / t) * 1000));
    }}
    time.textContent = fmt(elapsed) + ' / ' + fmt(t);
  }}

  audio.addEventListener('timeupdate', syncUi);
  audio.addEventListener('loadedmetadata', syncUi);
  seek.addEventListener('input', () => {{
    seeking = true;
    const t = total();
    const ratio = Number(seek.value) / 1000;
    bar.value = ratio * 100;
    time.textContent = fmt(ratio * t) + ' / ' + fmt(t);
  }});
  seek.addEventListener('change', () => {{
    audio.currentTime = (Number(seek.value) / 1000) * total();
    seeking = false;
    syncUi();
  }});
  playBtn.onclick = () => {{ audio.play(); }};
  pauseBtn.onclick = () => {{ audio.pause(); }};
}})();
</script>
"""

# generate the HTML for the session playlist
def session_playlist_html(tracks: list[PlaybackTrack]) -> str:
    playable = session_playlist_tracks(tracks)
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
  <input id="mme-seek" type="range" min="0" max="1000" value="0"
         style="width: 100%; margin: 0.35rem 0;" aria-label="Seek session">
  <div id="mme-time" style="margin: 0.35rem 0 0.75rem; color: #555;">0.0s / 0.0s</div>
  <button id="mme-play" style="margin-right: 0.5rem; padding: 0.4rem 1rem;">Play session</button>
  <button id="mme-pause" style="margin-right: 0.5rem; padding: 0.4rem 1rem;">Pause</button>
  <button id="mme-skip" style="padding: 0.4rem 1rem;">Skip track</button>
  <audio id="mme-audio" style="display:none;"></audio>
</div>
<script>
(function() {{
  const tracks = {data_json};
  let index = 0;
  let playing = false;
  let paused = false;
  let seeking = false;
  const audio = document.getElementById('mme-audio');
  const bar = document.getElementById('mme-bar');
  const seek = document.getElementById('mme-seek');
  const now = document.getElementById('mme-now');
  const time = document.getElementById('mme-time');
  const playBtn = document.getElementById('mme-play');
  const pauseBtn = document.getElementById('mme-pause');
  const skipBtn = document.getElementById('mme-skip');

  function fmt(s) {{ return s.toFixed(1) + 's'; }}

  function trackTotal(t) {{
    return Math.max(audio.duration || 0, t.duration, 0.001);
  }}

  function syncUi() {{
    const t = tracks[index];
    const total = trackTotal(t);
    const elapsed = audio.currentTime || 0;
    if (!seeking) {{
      bar.value = Math.min(100, (elapsed / total) * 100);
      seek.value = Math.min(1000, Math.round((elapsed / total) * 1000));
    }}
    time.textContent = fmt(elapsed) + ' / ' + fmt(total);
  }}

  function load(i) {{
    const t = tracks[i];
    audio.src = t.src;
    now.textContent = 'Now playing: ' + t.label + ' (' + (i+1) + '/' + tracks.length + ')';
    bar.value = 0;
    seek.value = 0;
    syncUi();
  }}

  audio.addEventListener('timeupdate', syncUi);
  audio.addEventListener('loadedmetadata', syncUi);
  seek.addEventListener('input', () => {{
    seeking = true;
    const total = trackTotal(tracks[index]);
    const ratio = Number(seek.value) / 1000;
    bar.value = ratio * 100;
    time.textContent = fmt(ratio * total) + ' / ' + fmt(total);
  }});
  seek.addEventListener('change', () => {{
    const total = trackTotal(tracks[index]);
    audio.currentTime = (Number(seek.value) / 1000) * total;
    seeking = false;
    syncUi();
  }});

  audio.addEventListener('ended', () => {{
    index += 1;
    if (index >= tracks.length) {{
      playing = false;
      paused = false;
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
      paused = false;
      load(0);
      audio.play();
      playBtn.textContent = 'Restart session';
    }} else if (paused) {{
      paused = false;
      audio.play();
    }} else {{
      index = 0;
      load(0);
      audio.play();
    }}
  }};

  pauseBtn.onclick = () => {{
    if (!playing || paused) return;
    audio.pause();
    paused = true;
  }};

  skipBtn.onclick = () => {{
    if (!playing) return;
    audio.pause();
    paused = false;
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

# generate the data URI for the WAV file. If too large return None
def _wav_data_uri(wav_path: Path) -> str | None:
    size = wav_path.stat().st_size
    if size > MAX_WAV_EMBED_BYTES:
        return None
    return f"data:audio/wav;base64,{wav_to_base64(wav_path)}"

# render the track player for the track
def render_track_player(track: PlaybackTrack) -> None:
    dom_id = f"mme-track-{sanitize_dom_id(track.label)}"
    audio_src: str | None = None
    if track.wav_path and track.wav_path.is_file():
        audio_src = _wav_data_uri(track.wav_path)
        if audio_src is None:
            st.warning(f"WAV for {track.label} is too large for in-browser playback (>{MAX_WAV_EMBED_BYTES // (1024 * 1024)} MB). Download MIDI or WAV below.")

    components.html(
        track_player_html(
            dom_id=dom_id,
            label=track.label,
            audio_src=audio_src,
            duration_seconds=track.duration_seconds,
        ),
        height=160,
    )

    if track.midi_path and track.midi_path.is_file():
        st.download_button(
            "Download MIDI",
            data=track.midi_path.read_bytes(),
            file_name=track.midi_path.name,
            mime="audio/midi",
            key=f"midi_dl_{dom_id}",
        )
    if track.wav_path and track.wav_path.is_file():
        st.download_button(
            "Download WAV",
            data=track.wav_path.read_bytes(),
            file_name=track.wav_path.name,
            mime="audio/wav",
            key=f"wav_dl_{dom_id}",
        )

# render the audio setup strip
# this is used to display the audio setup status
def render_audio_setup_strip() -> None:
    setup = check_audio_setup()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Soundfont", "Ready" if setup["soundfont_present"] else "Missing")
    with c2:
        st.metric("FluidSynth", "Found" if setup["fluidsynth_cli"] else "Not available")
    with c3:
        st.metric("WAV engine", "OK" if setup["can_synthesize_wav"] else "Unavailable")
    if not setup["can_synthesize_wav"]:
        st.caption("Dashboard playback needs a soundfont and FluidSynth. On first launch the app downloads both automatically, or run `make setup-audio`.")

# render the session player for the session
def render_session_player(tracks: list[PlaybackTrack]) -> None:
    st.markdown("##### Play full session")
    st.caption("Original → generated orders (same sequence as CLI `--play`).")
    components.html(session_playlist_html(tracks), height=220)

# render the playback studio for the tracks
# this is used to display the playback studio
def render_playback_studio(tracks: list[PlaybackTrack]) -> None:
    st.markdown("#### Playback studio")
    render_audio_setup_strip()
    render_session_player(tracks)

    if not tracks:
        return

    cols = st.columns(len(tracks))
    for col, track in zip(cols, tracks):
        with col:
            if track.duration_seconds > 0:
                st.caption(f"~{track.duration_seconds:.1f}s")
            if track.wav_path and track.wav_path.is_file():
                render_track_player(track)
            else:
                _render_track_unavailable(track)

# render the track unavailable for the track
# this is used to display the track unavailable status
def _render_track_unavailable(track: PlaybackTrack) -> None:
    st.markdown(f"**{track.label}**")
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
            key=f"midi_only_{sanitize_dom_id(track.label)}",
        )
    else:
        st.warning("No audio could be prepared for this track.")