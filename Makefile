# Preconfigured one-command workflows. Override any variable inline, e.g.
#   make backtest SYMBOLS=AAPL,MSFT START=2024-06-01 END=2024-09-01
.DEFAULT_GOAL := help

# --- tunable defaults -------------------------------------------------------
SYMBOLS ?= NVDA,RIVN,NFLX,META,BAC,MS,TSLA,GS,AMD,AAPL
START   ?= 2024-01-02
END      ?= 2024-04-01
CAPITAL ?= 100000
PY       = uv run python main.py

.PHONY: help demo install install-optimize backtest backtest-no-scan scan live \
        optimize optimize-bayesian cancel-orders close-positions \
        test docs docs-build docker-build docker-run clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- try it now (no setup) --------------------------------------------------
demo:  ## Run the whole pipeline on synthetic data — no API keys, no network
	$(PY) demo

demo-artifact:  ## Regenerate the README demo image (equity curve + verdict)
	uv run --extra viz python main.py demo --chart website/static/img/demo.png

# --- setup ------------------------------------------------------------------
install:  ## Create the uv environment and install dependencies
	uv sync

install-optimize:  ## Install with the optional Bayesian-optimization extra
	uv sync --extra optimize

install-portfolio:  ## Install with the optional OR-Tools portfolio extra
	uv sync --extra portfolio

# --- preconfigured trading combos ------------------------------------------
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

# --- quality & docs ---------------------------------------------------------
test:  ## Run the offline test suite (no API keys needed)
	uv run --extra dev pytest -q

docs:  ## Serve the documentation site at http://localhost:3000
	cd website && npm install && npm run start

docs-build:  ## Build the static documentation site
	cd website && npm install && npm run build

# --- docker -----------------------------------------------------------------
docker-build:  ## Build the Docker image
	docker build -t tradeflow .

docker-run:  ## Run the container (paper live trading; mounts your .env)
	docker run --rm -it -v $$(pwd)/.env:/app/.env tradeflow

clean:  ## Remove caches, build output, and results
	rm -rf .venv optimization_results.csv website/build website/node_modules
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
