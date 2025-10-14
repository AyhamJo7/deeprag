SHELL := /bin/bash

.PHONY: help setup lint test train evaluate

help:
	@echo "Commands:"
	@echo "  setup      : Install dependencies and setup pre-commit hooks."
	@echo "  lint       : Run linter and code formatter."
	@echo "  test       : Run all tests."
	@echo "  train      : Run a sample training job."
	@echo "  evaluate   : Run a sample evaluation job."

setup:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -e ".[dev,viz]"
	@echo "Installing pre-commit hooks..."
	pre-commit install

lint:
	@echo "Running linter and formatter..."
	pre-commit run --all-files

test:
	@echo "Running tests..."
	pytest

train:
	@echo "Running a sample DSI pre-training job..."
	python cli.py train-dsi --config-name=train/dsi_pretrain data=synthetic model=dsi_small training.max_steps=10

evaluate:
	@echo "Running a sample evaluation..."
	python cli.py evaluate-model
