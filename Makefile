run:
    streamlit run dashboard/app.py

test:
    pytest tests/

generate:
    python main.py --style classical --compare --single-source

docker-build:
    docker build -t markov-music-engine .

docker-run:
    docker compose up