"""What validated a config, assembled from what was already recorded.

Promoting the winning row of a walk-forward captures the parameters and loses the
recipe. But a walk-forward is not one row — it is a *validation recipe* (the folds, the
embargo, the objective, the search method, the book it searched at) plus a chosen
parameter set plus the universe that was actually resolved. Without the recipe the
config cannot say what validated it, and "this passed a walk-forward" is not a fact
anybody can check.

None of this is re-derived and none of it is re-run. The recipe is in the journal,
where it was written as the trial's dedup identity; the universe is in the journal, in
full, because the store keeps only its hash. This reads both back.

**Three kinds of thing, labelled as such, because they age differently.**

``recipe`` is *how it was validated* — reusable, still meaningful next year, and the
thing you would repeat to check the claim.

``evidence`` is *what was measured* — scoped to an accounting era and never valid
outside it. When ``ACCOUNTING_VERSION`` moves, the recipe survives unchanged and every
number here becomes a historical curiosity. Marking the boundary in the artifact is
what stops the two being read as one.

``metadata`` is *about the record itself* — where it came from, what is still
recoverable, what was never captured.

A section that could not be assembled says so with a reason. Absent is never rendered
as empty, and an artifact that quietly omits a section it failed to build claims more
completeness than it has.
"""

from typing import Any, Dict, List, Optional

#: The label on each block. Kept as data rather than prose so a reader — or an agent —
#: can filter on it: after an accounting bump, everything marked ``evidence`` is stale
#: and everything marked ``recipe`` is not.
RECIPE = "recipe"
EVIDENCE = "evidence"
METADATA = "metadata"

#: Reserved keys inside a recorded params/recipe dict. They are the dedup identity's
#: own bookkeeping, not knobs anybody set.
_RESERVED_PREFIX = "_"


def _split_recipe(recorded: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A recorded dedup recipe split into the knobs and the folded keys.

    The cost model and the book limits are folded into the recipe as reserved keys
    because they change what a validation *means*, but they are not search settings and
    reading them as such would be misleading.
    """
    recorded = dict(recorded or {})
    folded = {k: v for k, v in recorded.items() if k.startswith(_RESERVED_PREFIX)}
    knobs = {k: v for k, v in recorded.items() if not k.startswith(_RESERVED_PREFIX)}
    return {"settings": knobs, "folded": folded}


def campaign_material(store, trial_id: str, *, journal_path: Optional[Any] = None) -> Dict[str, Any]:
    """Everything recorded about what validated one trial, as one labelled block.

    Takes an open store rather than opening one — the CLI and the MCP server both reach
    this through the handle they already have, and a fourth way of opening the trial
    store is the last thing this codebase needs.
    """
    from tradeflow.engine.backtest import ACCOUNTING_VERSION
    from tradeflow.services.audit import journal_record_for_trial

    material: Dict[str, Any] = {"available": False, "reason": "", "trial_id": trial_id}

    row = store.row_for(trial_id)
    if row is None:
        material["reason"] = (
            f"no trial with id {trial_id!r} is indexed. Its journal line may still exist — "
            "`trials rebuild` reindexes from the journal"
        )
        return material

    # `audit_log` flattens a trial's extra fields onto the record rather than nesting
    # them under a key, so `dedup_params`, `returns` and `trades` sit at the top level
    # beside `inputs`. Printed once rather than inferred from the writer's signature:
    # assuming a nested `extra` compiled, passed, and reported every recipe as absent.
    record = journal_record_for_trial(trial_id, journal_path)
    inputs = (record or {}).get("inputs") or {}

    accounting = row.get("accounting")
    material.update(
        {
            "available": True,
            "kind": row.get("kind"),
            "strategy": row.get("strategy"),
            RECIPE: _recipe_section(row, inputs, record),
            EVIDENCE: _evidence_section(store, row, inputs, trial_id, accounting, ACCOUNTING_VERSION),
            METADATA: _metadata_section(row, inputs, record, trial_id),
        }
    )
    return material


#: Context facts that describe how a run was set up, in the order a reader wants them.
_CONTEXT_FACTS = ("capital", "scanner", "scan_as_of", "cache")


def _context_section(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """The recorded run context, with every fact carrying whether it was recorded.

    Each fact reports ``recorded`` rather than being present-or-missing, so a reader -
    or an agent - can tell "this run used no scanner" from "nobody wrote down whether it
    did". Every trial predating this block reads as the second, which is what it is: the
    fact was never captured, and defaulting it would put a capital or a cache policy in
    the record that nobody chose.
    """
    context = inputs.get("context") or {}
    return {fact: {"recorded": fact in context, "value": context.get(fact)} for fact in _CONTEXT_FACTS}


def _recorded(inputs: Dict[str, Any], key: str) -> Dict[str, Any]:
    """One context fact off a trial's journal inputs, with its recorded status.

    Reads the *journal* rather than the store row: the context is not indexed, because
    nothing queries on it and duplicating it into a column would be a second copy of a
    fact the journal already holds.
    """
    context = (inputs or {}).get("context") or {}
    return {"recorded": key in context, "value": context.get(key)}


def _recipe_section(row, inputs, record) -> Dict[str, Any]:
    """How it was validated. Survives an accounting bump unchanged."""
    section: Dict[str, Any] = {
        "kind": RECIPE,
        "note": "how this was validated — reusable, and unaffected by an accounting bump",
        "window": inputs.get("window") or {"start": row.get("window_start"), "end": row.get("window_end")},
        "objective": inputs.get("objective") or None,
        # How the run was set up, for the parts that are *not* its identity: the capital
        # it ran at, the scanner and as-of clock its universe came through, the cache
        # policy its bars were read under. Labelled separately from `validation` because
        # nothing here is hashed - two runs differing only in these keys are one trial to
        # the memo, which a reader comparing them needs to know.
        "context": _context_section(inputs),
        # The universe that was actually resolved, and the list it was resolved *from*.
        # The store keeps only a hash of the first and nothing of the second, so this is
        # the only place a reader gets them back - and without it the module's own
        # claim to read the universe back was false, and a config's symbols could
        # contradict the campaign with nothing able to notice.
        "universe": {
            "symbols": list(inputs.get("symbols") or []) or None,
            "candidate_symbols": list(inputs.get("candidate_symbols") or []) or None,
        },
    }
    if record is None:
        # The store knows the trial happened and the journal line is where the recipe
        # lives. Saying which half is missing beats an empty recipe that reads as "this
        # was validated with no settings".
        section["available"] = False
        section["reason"] = (
            "the journal line for this trial could not be read, and the recipe is only "
            "recorded there — the store keeps a hash of it, not the recipe itself"
        )
        return section

    dedup = record.get("dedup_params")
    if dedup is None:
        # Most kinds omit it because `params` already serves both roles; a backtest's
        # identity *is* its parameters. Only a search has a recipe distinct from them.
        section["available"] = False
        section["reason"] = (
            f"a {row.get('kind')!r} trial records no separate validation recipe — its "
            "identity is its parameters, which are the config's own `params`"
        )
        return section

    split = _split_recipe(dedup)
    section.update(
        {
            "available": True,
            "validation": split["settings"],
            # Named rather than dropped: these are why two validations with identical
            # settings can still be different validations.
            "folded_into_identity": split["folded"],
        }
    )
    return section


def _parse_metrics(row) -> Optional[Dict[str, Any]]:
    """A row's stored metric block, parsed. ``None`` when there is none - never ``{}``,
    which would read as a trial that measured nothing."""
    import json

    raw = row.get("metrics")
    if isinstance(raw, dict):
        return raw or None
    try:
        parsed = json.loads(row.get("metrics_json") or "{}")
    except (TypeError, ValueError):
        return None
    return parsed or None


def _evidence_section(store, row, inputs, trial_id, accounting, current) -> Dict[str, Any]:
    """What was measured. Scoped to one accounting era and worthless outside it."""
    strategy, universe_hash = row.get("strategy"), row.get("universe_hash")
    family = (
        store.family_count_by_hash(strategy, universe_hash, accounting)
        if strategy and universe_hash and accounting is not None
        else None
    )
    section: Dict[str, Any] = {
        "kind": EVIDENCE,
        "note": (
            "what was measured, under accounting "
            f"v{accounting} — not comparable with a run on a different version"
        ),
        "accounting": accounting,
        "current_accounting": current,
        "comparable_with_current_engine": accounting == current,
        "trial_ids": [trial_id],
        # The raw row carries `metrics_json`, a string. `get_trial` parses it; this
        # reads the light row and has to do the same, or every materialised config
        # records `metrics: null` for a trial that measured plenty.
        "metrics": _parse_metrics(row),
        # SQLite has no boolean, so the column is 0/1/NULL. A JSON artifact that says
        # `"promotable": 1` invites a reader to treat the count-looking value as one.
        "promotable": None if row.get("promotable") is None else bool(row.get("promotable")),
        # Which probes ran and what they said. A probe absent here did not run, which is
        # not the same as running and passing - the distinction the probe suite itself
        # is built around, carried into the record.
        "probes": _recorded(inputs, "probes"),
        "family_n_trials": family,
        "quarantined": bool(row.get("contaminated_at")),
        "quarantine_reason": row.get("contamination_reason"),
    }
    if accounting != current:
        section["staleness"] = (
            f"recorded under accounting v{accounting}; this engine is v{current}. The "
            "recipe above still applies — every number in this section does not"
        )
    # Later trials served from this one are part of the same evidence: the number was
    # reused, and a reader tracing a config back deserves the whole chain.
    reused = store.reused_by(row) or []
    if reused:
        section["trial_ids"].extend(r["id"] for r in reused)
        section["reused_by"] = reused
    return section


def _seed(row, record) -> Dict[str, Any]:
    """The seed this run used, and whether anybody wrote it down.

    Read from the validation recipe first, because that is where it actually lives. The
    ``seed`` column exists but only the research agent's session records ever populate
    it — ``journal_trial`` has no seed parameter at all — so reading the column alone
    reported ``None`` for every walk-forward driven from the CLI or MCP *while the same
    block carried the real value two sections above*. A record that contradicts itself
    about a fact it holds is worse than one that admits the fact is missing.

    The column is still consulted, for the research-agent trials that do populate it and
    may carry no recipe. Absent stays absent: a run whose seed nobody recorded reads as
    unrecorded rather than as some default, which is the same rule every other fact in
    this block follows.
    """
    recipe_seed = ((record or {}).get("dedup_params") or {}).get("seed")
    if recipe_seed is not None:
        return {"recorded": True, "value": recipe_seed, "from": "validation recipe"}
    stored = row.get("seed")
    if stored is not None:
        return {"recorded": True, "value": stored, "from": "session record"}
    return {"recorded": False, "value": None, "from": None}


def _metadata_section(row, inputs, record, trial_id) -> Dict[str, Any]:
    """About the record: where it came from, and what is still recoverable.

    Artifacts are named by the command that reads them rather than by a path. A config
    is a portable thing and the reader may be running an installed copy with a state
    root somewhere else entirely, so a filesystem path is the one form of this that is
    guaranteed wrong for somebody.
    """
    from tradeflow.services.setup import invocation

    stored: List[Dict[str, Any]] = []
    for name, key, how in (
        ("return series", "returns", invocation(f"trials compare {trial_id} <other-trial-id>")),
        ("trade table", "trades", invocation(f"trials analyze {trial_id}")),
        ("proposed book", "weights", invocation(f"trials show {trial_id}")),
    ):
        present = record is not None and record.get(key) is not None
        stored.append({"artifact": name, "recorded": present, "read_with": how if present else None})

    return {
        "kind": METADATA,
        "note": "about this record — not a claim about the strategy",
        "trial_id": trial_id,
        "recorded_at": row.get("ts"),
        "git_sha": row.get("git_sha"),
        "seed": _seed(row, record),
        "journal_line_found": record is not None,
        "notes": _recorded(inputs, "notes"),
        "artifacts": stored,
    }
