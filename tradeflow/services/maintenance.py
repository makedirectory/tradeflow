"""Trial-store maintenance: quarantining a subset, and retiring an era.

Two operations, deliberately not one command. They answer different questions, and
they differ in how reversible they are:

**Quarantine** (:func:`mark_contaminated`) is for a *suspect subset*. Something was
learned about some trials after they were recorded — a bug in a scanner, a data vendor
correction, a config that turned out not to be the one you thought. History stays
exactly as written; an event is appended saying what was learned, and readers honour it.

**Archive** (:func:`archive`) is for a *whole era*. An accounting bump changes what the
engine computes, so every recorded metric becomes incommensurable with anything produced
afterwards — not suspect, but measured with a different instrument. That is not a subset
and no annotation fixes it: the journal and its index are moved aside together and a
fresh pair started.

Collapsing them would be a mistake in both directions. Quarantining an era leaves
thousands of rows carrying a caveat nobody can act on; archiving a subset throws away
the evidence around it.

**Nothing here deletes anything.** There is no `reset`, because the destructive version
of archive is exactly what the append-only rule exists to forbid: the journal has nothing
behind it to rebuild from, so a record removed is a record gone. Archive moves; the files
are still on disk under a name that says when and why.

The two files move **together**, always. The store is derived from the journal, so moving
one without the other leaves a store indexing a file that no longer exists — or worse, a
store full of an old era's rows sitting beside a fresh journal, silently disagreeing with
its own source and reporting a multiple-testing count for evidence that is no longer
there. That is precisely the hand-rolled state this exists to replace.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tradeflow.engine.backtest import ACCOUNTING_VERSION
from tradeflow.services import audit
from tradeflow.settings import state_root
from tradeflow.store.trials import CONTAMINATION_TOOL, TrialStore, db_path_for_journal

logger = logging.getLogger(__name__)

#: Where retired eras go. Under the state root, beside the journal they came from, so
#: an archive travels with the campaign rather than wherever the command was run.
ARCHIVE_DIRNAME = "archive"

#: Written into each archive directory. The files alone do not say why they were
#: retired, and "why" is the part nobody reconstructs a year later.
MANIFEST_NAME = "manifest.json"

#: Appended to the *new* journal as its first line after an archive. A fresh journal that
#: simply starts empty loses the fact that an era existed, and the emptiness then reads
#: as "nothing has ever been run here" — which is how a campaign's evidence quietly
#: appears to have never happened.
ARCHIVE_TOOL = "trials:archived"


def _resolve_journal(journal_path: Optional[Any]) -> Path:
    return Path(journal_path) if journal_path else audit.default_trial_journal()


def select_trials(
    *,
    journal_path: Optional[Any] = None,
    trial_ids: Optional[Iterable[str]] = None,
    strategy: Optional[str] = None,
    kind: Optional[str] = None,
    accounting: Optional[int] = None,
    before: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """The trials a quarantine would name, resolved to explicit rows.

    Filters are resolved to ids **now**, and the event records those ids rather than the
    filter. A stored filter would re-evaluate on every rebuild and could silently pull in
    trials recorded after the decision was made — a quarantine that grows on its own is
    not a record of what somebody decided.
    """
    journal = _resolve_journal(journal_path)
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        if trial_ids:
            rows = [store.get_trial(str(t)) for t in trial_ids]
            return [r for r in rows if r]
        # `until` rather than a second date comparison here: the store owns one
        # definition of what a filter means, and a listing that disagreed with the
        # store's own count of the same filter is the bug that definition prevents.
        return store.list_trials(
            strategy=strategy,
            kind=kind,
            accounting=accounting,
            all_accounting=accounting is None,
            until=before,
            limit=1_000_000,
        )


def mark_contaminated(
    *,
    reason: str,
    journal_path: Optional[Any] = None,
    trial_ids: Optional[Iterable[str]] = None,
    strategy: Optional[str] = None,
    kind: Optional[str] = None,
    accounting: Optional[int] = None,
    before: Optional[Any] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Quarantine a suspect subset of trials, reason on the record.

    Appends one event naming every affected trial, then applies it to the derived index.
    A rebuild replays the event and reaches the same rows, so the quarantine survives the
    store being thrown away — which it is, routinely, whenever the schema moves.

    ``reason`` is required and not defaulted. An unexplained quarantine is worse than
    none: a later reader sees rows excluded from every leaderboard with nothing saying
    what was wrong with them, and has no way to judge whether it still applies.
    """
    if not (reason or "").strip():
        raise ValueError("A quarantine needs a reason: an unexplained one cannot be judged later")

    journal = _resolve_journal(journal_path)
    selected = select_trials(
        journal_path=journal,
        trial_ids=trial_ids,
        strategy=strategy,
        kind=kind,
        accounting=accounting,
        before=before,
    )
    already = [r for r in selected if r.get("contaminated_at")]
    fresh = [r for r in selected if not r.get("contaminated_at")]
    ids = [r["id"] for r in fresh]

    payload = {
        "selected": len(selected),
        "already_contaminated": len(already),
        "to_mark": len(ids),
        "trial_ids": ids,
        "reason": reason,
        "journal_path": str(journal),
        "dry_run": dry_run,
    }
    if dry_run or not ids:
        payload["applied"] = 0
        payload["note"] = (
            "Nothing selected." if not selected else "Dry run: nothing was written."
        ) + " A quarantined trial still counts toward its family's multiple-testing total."
        return payload

    run_id = audit.audit_log(
        CONTAMINATION_TOOL,
        {"trial_ids": ids, "reason": reason, "selected": len(selected)},
        path=journal,
    )
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        applied = store.mark_contaminated(ids, reason=reason)
    payload["applied"] = applied
    payload["event_run_id"] = run_id
    payload["note"] = (
        "Quarantined trials are never served as a memo and never ranked, and they still "
        "count toward their family's multiple-testing total — the search happened, and "
        "removing it would lower the deflation bar rather than raise it."
    )
    return payload


def archive(
    *,
    reason: str,
    journal_path: Optional[Any] = None,
    label: Optional[str] = None,
    stamp: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Retire a whole era: move the journal and its index aside, together.

    Together is the requirement, not a convenience. The store is an index over the
    journal, so moving one alone leaves the other describing something that is not there
    — a store still reporting an era's multiple-testing count beside a journal that no
    longer holds it, with nothing erroring, because both files are individually valid.

    The archive keeps a manifest saying what was retired, when, why, and under which
    accounting version and commit. And the *new* journal opens with a record of the
    archive, so its emptiness reads as "an era was retired here" rather than as "nothing
    has ever been run".
    """
    if not (reason or "").strip():
        raise ValueError(
            "An archive needs a reason: a retired era with no explanation cannot be judged later"
        )

    journal = _resolve_journal(journal_path)
    db = db_path_for_journal(journal)
    when = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{when}-{label}" if label else when
    destination = state_root() / ARCHIVE_DIRNAME / name

    summary: Dict[str, Any] = {"rows": 0, "journal_lines": 0}
    if journal.exists() or db.exists():
        try:
            with TrialStore(db, journal_path=journal) as store:
                status = store.status(journal)
            summary = {
                "rows": status["rows"],
                "journal_lines": status["journal_lines"],
                "journal_trial_lines": status["journal_trial_lines"],
                "contaminated_rows": status.get("contaminated_rows", 0),
            }
        except Exception:  # noqa: BLE001 - a broken store must not block retiring it
            logger.warning("Could not summarize the store before archiving it", exc_info=True)

    manifest = {
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "label": label,
        "accounting_version_at_archive": ACCOUNTING_VERSION,
        "git_sha": audit.current_git_sha(),
        "journal_path": str(journal),
        "db_path": str(db),
        "destination": str(destination),
        **summary,
    }
    if dry_run:
        return {**manifest, "moved": [], "dry_run": True, "note": "Dry run: nothing was moved."}

    existing = [path for path in (journal, db) if path.exists()]
    if not existing:
        return {
            **manifest,
            "moved": [],
            "dry_run": False,
            "note": f"Nothing to archive: neither {journal.name} nor {db.name} exists.",
        }

    destination.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in existing:
        shutil.move(str(path), str(destination / path.name))
        moved.append(path.name)
    (destination / MANIFEST_NAME).write_text(json.dumps({**manifest, "moved": moved}, indent=2) + "\n")

    # The new journal's first line, so an empty campaign says why it is empty.
    audit.audit_log(
        ARCHIVE_TOOL,
        {"reason": reason, "archived_to": str(destination), "label": label, **summary},
        path=journal,
    )
    return {**manifest, "moved": moved, "dry_run": False, "note": "Both files moved together."}


def list_archives() -> List[Dict[str, Any]]:
    """Every retired era, newest first, with what its manifest recorded."""
    root = state_root() / ARCHIVE_DIRNAME
    if not root.is_dir():
        return []
    out = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        entry: Dict[str, Any] = {"name": directory.name, "path": str(directory)}
        manifest = directory / MANIFEST_NAME
        if manifest.exists():
            try:
                entry.update(json.loads(manifest.read_text()))
            except json.JSONDecodeError:
                entry["manifest_error"] = "unreadable"
        else:
            # Reported, never inferred: a directory somebody created by hand is not
            # evidence about what it holds.
            entry["manifest_error"] = "missing"
        out.append(entry)
    return out
