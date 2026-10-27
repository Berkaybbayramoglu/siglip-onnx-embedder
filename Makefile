.PHONY: install install-dev export test build run-local serve benchmark

MODEL_ID ?= google/siglip2-base-patch16-256

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

export:
	python3 scripts/export_model.py --model-id $(MODEL_ID)

test:
	pytest -v

build:
	docker build -t siglip-embedder .

run-local: build
	docker run --rm -p 9000:8080 siglip-embedder

serve:
	python3 scripts/serve.py

benchmark:
	python3 scripts/benchmark.py
