# TensorGym - developer commands
VENV ?= .venv
PY := $(VENV)/bin/python
IMAGE := tensorgym-executor:py312-torch2.12.0-cpu

.PHONY: help setup build-executor test test-fast run clean

help:
	@echo "setup          create venv and install host deps"
	@echo "build-executor build the pinned PyTorch 2.12.0 CPU sandbox image"
	@echo "test           run the full suite (requires Docker + executor image)"
	@echo "test-fast      run only the non-Docker unit tests (feedback engine)"
	@echo "run            start the API + UI at http://127.0.0.1:8000"
	@echo "clean          remove the local SQLite db and ephemeral job dirs"

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

build-executor:
	docker build -t $(IMAGE) services/executor

test:
	$(PY) -m pytest tests/ -v

test-fast:
	$(PY) -m pytest tests/test_feedback.py -v

run:
	$(PY) -m uvicorn app.main:app --app-dir apps/api --host 127.0.0.1 --port 8000

clean:
	rm -rf var/tensorgym.db var/jobs
