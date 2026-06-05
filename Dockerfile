# Markov Music Engine — Streamlit dashboard image
#
# Ships a system FluidSynth so WAV rendering works inside the container.
# Note: the CLI `--play` flag (pygame + host MIDI device) is a host-only path;
# in-container audio is WAV-via-FluidSynth, served through the dashboard.
FROM python:3.11-slim

# System FluidSynth (CLI + shared library) for MIDI -> WAV synthesis.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fluidsynth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

EXPOSE 8501

# Streamlit must bind to 0.0.0.0 to be reachable from outside the container.
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
