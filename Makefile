.PHONY: install lint format test data train evaluate reproduce api docker-up

install:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff check . --fix
	python -m ruff format .

test:
	python -m pytest

data:
	python -m ecommerce_recommender.data.generate
	python -m ecommerce_recommender.features.prepare

train:
	python -m ecommerce_recommender.models.train

evaluate:
	python -m ecommerce_recommender.models.evaluate

reproduce:
	dvc repro

api:
	python -m uvicorn ecommerce_recommender.api.main:app --reload

docker-up:
	docker compose up --build
