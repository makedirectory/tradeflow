# Preconfigured one-command workflows. Override any variable inline, e.g.
#   make backtest SYMBOLS=AAPL,MSFT START=2024-06-01 END=2024-09-01
.DEFAULT_GOAL := help

# --- tunable defaults -------------------------------------------------------
SYMBOLS ?= NVDA,RIVN,NFLX,META,BAC,MS,TSLA,GS,AMD,AAPL
START   ?= 2024-01-02
END      ?= 2024-04-01
CAPITAL ?= 100000
PY       = uv run python main.py

.PHONY: help demo demo-agent demo-agent-live init install install-optimize verdict backtest backtest-no-scan scan live \
        allocate allocate-utility alphas risk info horizon optimize optimize-bayesian cancel-orders close-positions \
        check test check-links docs docs-build docker-build docker-run up down compose-run compose-smoke \
        build release-check clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- try it now (no setup) --------------------------------------------------
demo:  ## Run the whole pipeline on synthetic data — no API keys, no network
	$(PY) demo

demo-agent:  ## Narrate an AI research session on live data (needs Alpaca keys; no LLM key)
	$(PY) demo-agent

demo-agent-live:  ## Same, but with a live Claude proposer (needs ANTHROPIC_API_KEY + 'ai' extra)
	uv run --extra ai python main.py demo-agent --provider anthropic

demo-artifact:  ## Regenerate the README demo image (equity curve + verdict)
	uv run --extra viz python main.py demo --chart website/static/img/demo.png

# --- setup ------------------------------------------------------------------
init:  ## Guided first-run setup: write a valid .env and check it
	$(PY) init

check:  ## Diagnose the current setup (no writes, no network)
	$(PY) init --check

install:  ## Create the uv environment and install dependencies
	uv sync

install-optimize:  ## Install with the optional Bayesian-optimization extra
	uv sync --extra optimize

install-portfolio:  ## Install with the optional OR-Tools portfolio extra
	uv sync --extra portfolio

# --- preconfigured trading combos ------------------------------------------
verdict:  ## The whole pipeline in one command: scan -> alphas -> portfolio -> information
	$(PY) verdict --strategy volume_spike --scanner volume --symbols $(SYMBOLS) --start $(START) --end $(END) --capital $(CAPITAL)

backtest:  ## Backtest: volume scanner -> volume_spike strategy
	$(PY) backtest --strategy volume_spike --scanner volume --symbols $(SYMBOLS) --start $(START) --end $(END) --capital $(CAPITAL)

backtest-no-scan:  ## Backtest the fixed symbol list (skip the scanner)
	$(PY) backtest --strategy volume_spike --scanner none --symbols $(SYMBOLS) --start $(START) --end $(END) --capital $(CAPITAL)

backtest-beta:  ## Backtest with beta-scaled position sizing
	$(PY) backtest --strategy volume_spike --scanner volume --symbols $(SYMBOLS) --start $(START) --end $(END) --capital $(CAPITAL) --beta-sizing

scan:  ## Run the universe scanner and print flagged symbols
	$(PY) scan --scanner volume --symbols $(SYMBOLS)

allocate: install-portfolio  ## Weight a portfolio over scanned symbols (OR-Tools)
	$(PY) allocate --scanner volume --symbols $(SYMBOLS) --capital $(CAPITAL)

alphas:  ## Rank a universe by continuous alpha (residual-return forecast) — read-only
	$(PY) alphas --strategy volume_spike --symbols $(SYMBOLS) --as-of $(END) --ic 0.03

risk:  ## Estimate the universe covariance Σ and summarize its risk structure — read-only
	$(PY) risk --symbols $(SYMBOLS) --as-of $(END) --model shrinkage

allocate-utility:  ## Mean-variance portfolio construction (alpha + Σ) — read-only proposal
	$(PY) allocate --objective utility --strategy volume_spike --symbols $(SYMBOLS) --as-of $(END) --target-te 0.04

info:  ## Information report: IC, breadth, predicted-vs-realized IR — read-only
	$(PY) info --strategy volume_spike --symbols $(SYMBOLS) --start $(START) --end $(END)

horizon:  ## Alpha decay / half-life + rebalance cadence + lagged blend — read-only
	$(PY) horizon --strategy volume_spike --symbols $(SYMBOLS) --start $(START) --end $(END)

live:  ## Paper-trade: volume scanner -> volume_spike strategy
	$(PY) live --strategy volume_spike --scanner volume --symbols $(SYMBOLS)

live-portfolio: install-portfolio  ## Paper-trade with OR-Tools portfolio-weighted sizing
	$(PY) live --strategy volume_spike --scanner volume --symbols $(SYMBOLS) --portfolio

live-beta:  ## Paper-trade with beta-scaled position sizing
	$(PY) live --strategy volume_spike --scanner volume --symbols $(SYMBOLS) --beta-sizing

# --- parameter modeling -----------------------------------------------------
optimize:  ## Grid-search strategy params (objective: sharpe_ratio)
	$(PY) optimize --strategy volume_spike --scanner none --symbols $(SYMBOLS) --start $(START) --end $(END) --method grid --max-evals 50

optimize-bayesian: install-optimize  ## Train a GP surrogate to tune params
	$(PY) optimize --strategy volume_spike --scanner none --symbols $(SYMBOLS) --start $(START) --end $(END) --method bayesian

# --- account utilities ------------------------------------------------------
cancel-orders:  ## Cancel all open orders
	uv run python cancel_all_orders.py

close-positions:  ## Close all open positions (also cancels orders)
	uv run python close_all_positions.py

# --- release ----------------------------------------------------------------
build:  ## Build the wheel and sdist into dist/
	rm -rf dist && uv build

release-check: build  ## Build, validate the metadata, and install into a clean venv
	uvx twine check dist/*
	@rm -rf /tmp/tradeflow-release-check
	@python3 -m venv /tmp/tradeflow-release-check
	@/tmp/tradeflow-release-check/bin/pip install -q dist/*.whl
	@TRADEFLOW_HOME=/tmp/tradeflow-release-state /tmp/tradeflow-release-check/bin/tradeflow --version
	@echo "OK — the wheel installs and runs outside this checkout."

# --- quality & docs ---------------------------------------------------------
check-links:  ## Verify Markdown links + heading anchors (includes local specs/)
	uv run python scripts/check_links.py

test:  ## Run the offline test suite (no API keys needed)
	uv run --extra dev pytest -q

docs:  ## Serve the documentation site at http://localhost:3000
	cd website && npm install && npm run start

docs-build:  ## Build the static documentation site
	cd website && npm install && npm run build

# --- docker -----------------------------------------------------------------
up:  ## Boot the local dev stack (MCP + persistent state). Never starts live trading.
	docker compose up -d

down:  ## Stop the stack (named volumes, and your trial history, survive)
	docker compose down

compose-run:  ## Run one verb in the stack, e.g. make compose-run CMD="verdict --symbols NVDA"
	docker compose run --rm $(CMD)

compose-smoke:  ## Verify the compose wiring end to end (needs Docker; not run in CI)
	./scripts/compose_smoke.sh

docker-build:  ## Build the Docker image
	docker build -t tradeflow .

docker-run:  ## Run the container (prints help; pass a verb to do anything)
	docker run --rm -it -v $$(pwd)/.env:/state/.env:ro tradeflow

clean:  ## Remove caches, build output, and results
	rm -rf .venv optimization_results.csv website/build website/node_modules
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
