---
sidebar_position: 12
title: HTML reports
---

# HTML reports

`--html PATH` writes one **self-contained** HTML file for a run: openable anywhere,
attachable, and readable a month later without the context that produced it. It is
available on the four commands that end in a result worth keeping:

```bash
quantred verdict     --scanner none --symbols AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,JPM,XOM,UNH,COST,WMT \
    --start 2024-01-01 --end 2024-12-31 --html verdict.html
quantred backtest    --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31 --html backtest.html
quantred walkforward --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31 --html wf.html
quantred info        --scanner none --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN,AVGO,JPM,XOM,UNH \
    --start 2024-01-01 --end 2024-12-31 --html info.html
```

Each command prints its usual report, then `HTML report written to <path>`. Open the file
in any browser. The reports need the same data as the run itself: Alpaca keys from
[`quantred init`](first-run), or `--offline` against a warmed cache.

**No file can mean the run produced no result.** The report renders the result the command
already computed, so a command that stops early writes nothing. `info` over too few
names (three is too few for a cross-sectional measurement) prints a short note about how
the IC is measured and no report; use twelve or so names. An `infeasible` verdict still
writes a report, which shows the refusal. If the file cannot be written the command
prints `HTML report skipped: <reason>` and the run itself is not lost.

`--html` is independent of the other output flags. Of these four commands only
`verdict` also has `--json [PATH]` (the structured result object, printed or written to
`PATH`), and there the two can both be given - with a bare `--json` the HTML notice goes to
stderr so stdout stays one JSON object; `backtest`, `walkforward` and `info` have `--html` only. A memoized run
renders its reuse warning in the report rather than hiding it. A reused walk-forward renders
as a reduced report - the stored out-of-sample aggregate, the original trial's id and age, and
a list of what a reused result does not retain (per-fold detail, promotion prerequisites,
the individual gate checks, and the holdout only when the original run did not record one)
- over both the CLI and `render_report`. When the original run did record a holdout, the
report shows it in a "Holdout (from the original run)" section instead of listing it as
missing; re-run with force for the full one. A reused backtest's header names the benchmark and timeframe as
requested.

Where the numbers come from: [verdict](verdict), [backtesting](backtesting),
[walk-forward](walk-forward) and [information](information). Reports of past runs are found
through [trials](trials).

## Self-contained means self-contained

Opening a report issues **zero** network requests. No CDN stylesheet, no web font,
no remote image, no analytics. CSS is inlined; charts are embedded as base64 PNGs.
That is an offline guarantee and a privacy one at once — a report you forward must
not phone home from someone else's machine. The test suite asserts it with a regex
over every `src=`, `href=`, and `url(` in the output rather than trusting the
intent.

## What it shows, and in what order

**Provenance first.** Window, universe, timeframe, benchmark, cost model, git SHA,
generation time, and the campaign's trial count — mandatory header, never a
footnote. A file that outlives its context needs all of it to not be misread.

**Warnings before numbers.** The medium flatters; the report must not. These render
as prominent banners above every section:

| Banner | When |
|---|---|
| `REUSED` | The result was served from a prior trial, with the original run's timestamp and age |
| `INCOMPLETE — no verdict` | A pipeline step failed; nothing below is safe to act on |
| Gate failures | Any verdict check or promotion gate that did not pass, named |
| `NOT PROMOTABLE` | A walk-forward whose promotion gates did not clear |
| `PROMOTABLE` is the statistical gates only | A walk-forward whose gates passed but whose promotion prerequisites are unmet or unknown; the verdict is not drawn as clear |
| Leakage probe FAILED | The result is not trustworthy at all |
| Non-default configuration | An evidence-gated, default-off feature (conditional risk, the aim trading policy, the Bayesian posterior) was enabled |

**Then the sections**, per kind: a verdict's five stages and its gate scorecard; a
backtest's metrics and equity curve; a walk-forward's per-fold table, out-of-sample
aggregate, holdout (formatted with the same units as the aggregate), and the promotion
prerequisites with evidence completeness and excess return by fold (the same block the
terminal prints); an information report's IC, breadth, and IR reconciliation.

The terminal's caveats are carried too, from the same functions rather than a second
phrasing: a verdict's alphas heading (scaling, assumed IC, name count, thin-universe
warning); a backtest's Verdicts block (statistical validation and evidence completeness
"not assessed here" - a backtest is an in-sample replay), its universe provenance
including the survivorship statement, and "Where the P&L came from" with the
exit-concentration warning (for a reused backtest, which kept no trades, the card says so);
a walk-forward's Verdicts block. A metric stored as a placeholder zero beside a flag that
it was unavailable (Treynor with a near-zero beta; alpha, beta and the information ratio
without a benchmark) reads "not measured". A journaled walk-forward names its trial id and
the campaign's trial count; timestamps are UTC.

## It renders, it does not compute

The input is the same result dict the command already printed —
`render_html(result, kind)` is a pure function. Nothing in the report is derived a
second way, so the file and the terminal cannot disagree about a number. A payload
handed to the wrong renderer fails loudly rather than half-rendering: a composite
verdict carries a schema stamp that is checked exactly.

One consequence worth knowing: **a backtest's equity chart comes from the CLI, not
from the result dict.** The dict deliberately omits the equity curve (it can run to
tens of thousands of points, and it crosses the MCP wire), so the CLI passes the
curve it is already holding. A backtest report rendered from the dict alone — over
MCP, say — shows the metrics table without the curve.

## Without the plotting extra

Charts need matplotlib (the `viz` extra). Without it the report still renders
completely in text and each chart slot reads "Chart not rendered — the viz extra is not
installed." Install the `viz` extra as described in [optional extras](extras) (name the extras you already use in the same command), then re-run.

Charts are rendered at a deliberately modest resolution — an embedded PNG costs
about 4/3 its own size once base64-encoded, and a report should stay well under a
few megabytes.

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, `uv run quantred <command> ...` runs the same thing (`uv run python main.py <command> ...` also works). Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
