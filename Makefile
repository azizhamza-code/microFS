# Makefile for microFS

.PHONY: setup run-fp run-train run-infer run-all-pipelines test clean

setup:
	poetry install

run-fp:
	python pipelines/feature_pipeline.py

run-train:
	python pipelines/training_pipeline.py

run-infer:
	python pipelines/inference_pipeline.py

run-all-pipelines: run-fp run-train run-infer

test:
	pytest tests/

clean:
	rm -rf data/fs_state/*
	rm -rf .pytest_cache
	rm -rf microfs/__pycache__
	rm -rf pipelines/__pycache__
	rm -rf tests/__pycache__ 