.PHONY: install test lint run docker-up docker-down

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check app tests

run:
	python -m app.main

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
