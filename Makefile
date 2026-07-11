# Project Makefile

PYTHON = python3
PIP = pip

# Default target

.DEFAULT_GOAL := help

# setup

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]

clean:
	find . -name "__pycache__" -delete
	find . -name "*.pyc" -delete

# run

train:
	train-rl

episode:
	run-episode

# tests

test:
	pytest

lint:
	ruff check .

# evaluation

eval:
	evid-eval --checkpoint $(checkpoint) --baselines random,greedy_llm,fewshot_k3,best_of_5

eval-quick:
	evid-eval --checkpoint $(checkpoint) --baselines random,greedy_llm --n-episodes 20

eval-full:
	evid-eval --checkpoint $(checkpoint) --baselines random,majority,greedy_llm,fewshot_k3,fewshot_k5,best_of_5 --n-episodes 100

eval-ci:
	evid-eval --checkpoint $(checkpoint) --baselines greedy_llm --n-episodes 50

collect:
	evid-collect

collect-n:
	evid-collect --n-episodes $(n)

eval-imitation:
	evid-eval --checkpoint $(checkpoint) --trajectories $(trajectories) --baselines random,greedy_llm,fewshot_k3,best_of_5,imitation

migrate:
	evid-migrate

# experiments

train-ppo:
	train-rl --method ppo --episodes 100

train-pg:
	train-rl --method pg --episodes 100

train-bandit:
	train-rl --method bandit --episodes 100

dashboard:
	streamlit run dashboard/app.py

plot:
	plot-exp --path=$(path)

compare:
	compare-exp --paths $(paths)

# full reset

reset: clean
	rm -rf build dist *.egg-info

# help

help:
	@echo "Available commands:"
	@echo "  make install        Install package"
	@echo "  make install-dev    Install with dev deps (pytest, ruff)"
	@echo "  make train          Run training (train-rl defaults)"
	@echo "  make episode        Run single episode"
	@echo "  make test           Run tests"
	@echo "  make lint           Run ruff"
	@echo "  make clean          Remove cache files"
	@echo "  make reset          Full cleanup"
	@echo ""
	@echo "Evaluation (requires checkpoint=<path>):"
	@echo "  make eval           Eval vs random,greedy,fewshot_k3,best_of_5"
	@echo "  make eval-quick     Fast eval (random,greedy, 20 episodes)"
	@echo "  make eval-full      Full eval all baselines, 100 episodes"
	@echo "  make eval-ci        CI gate — exits 1 if RL doesn't beat greedy"
	@echo "  make eval-imitation checkpoint=<path> trajectories=<path>"
	@echo "                      Eval including imitation baseline"
	@echo "  make collect        Collect trajectories via evid-collect"
	@echo "  make collect-n n=<int>  Collect N episodes of trajectories"
	@echo "  make migrate        Migrate trajectory files via evid-migrate"
	@echo ""
	@echo "Analysis:"
	@echo "  make plot path=<path>         Plot a single experiment"
	@echo "  make compare paths=<paths>    Compare multiple experiments"
