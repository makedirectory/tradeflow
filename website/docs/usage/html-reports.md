---
sidebar_position: 12
title: HTML reports
---

# HTML reports

`--html PATH` writes one **self-contained** HTML file for a run: openable anywhere,
attachable, and readable a month later without the context that produced it. It is
available on the four commands that end in a result worth keeping:

```bash
python main.py verdict     --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31 --html verdict.html
python main.py backtest    --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31 --html backtest.html
python main.py walkforward --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31 --html wf.html
python main.py info        --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31 --html info.html
```

It composes with everything else: `--json` and `--html` can both be given, and a
memoized run renders its reuse warning rather than hiding it.

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
| Leakage probe FAILED | The result is not trustworthy at all |
| Non-default configuration | An evidence-gated, default-off feature (conditional risk, the aim trading policy, the Black–Litterman posterior) was enabled |

**Then the sections**, per kind: a verdict's five stages and its gate scorecard; a
backtest's metrics and equity curve; a walk-forward's per-fold table, out-of-sample
aggregate, and holdout; an information report's IC, breadth, and IR reconciliation.

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
completely in text and each chart slot says how to install it:

```bash
uv sync --extra viz
```

Charts are rendered at a deliberately modest resolution — an embedded PNG costs
about 4/3 its own size once base64-encoded, and a report should stay well under a
few megabytes.
