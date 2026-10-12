.PHONY: install dev data index finetune-data train eval serve test lint docker

install:
	pip install -r requirements.txt && pip install -e .

dev:
	pip install -e ".[dev]"

data:
	./scripts/download_data.sh

index:
	python -m cia.build_index

finetune-data:
	python -m cia.finetune.prepare_data

train:
	python -m cia.finetune.train_lora

eval:
	python -m cia.eval.run_eval

serve:
	uvicorn cia.api.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check src tests

docker:
	docker build -t clinical-insight-assistant .
