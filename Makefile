run:
    python -m streamlit run dashboard/app.py

setup-audio:
	python -m markov.audio_setup

test:
	pytest tests/ --ignore=tests/test_order3.py -q

test-parallel:
	pytest tests/ -n auto --ignore=tests/test_order3.py -q

generate:
    python main.py --style classical --compare --single-source

docker-build:
    docker build -t markov-music-engine .

docker-run:
    docker compose up