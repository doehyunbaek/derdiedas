#!/usr/bin/env python3
"""Validate dataset-selected weak verbs against German Wiktionary.

This runner combines:
- german-verbs.json for verb class and corpus rank
- goethe-cefr-levels.json for the requested Goethe level
- check_conjugations.py for generation and external validation
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import tempfile

from check_conjugations import validate

ROOT = Path(__file__).resolve().parent
VERBS_FILE = ROOT / "german-verbs.json"
CEFR_FILE = ROOT / "goethe-cefr-levels.json"
STATUSES = ("pass", "mismatch", "source-unavailable")


def main():
    parser = ArgumentParser()
    parser.add_argument("--level", default="A1", choices=("A1", "A2", "B1"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="minimum seconds between Wiktionary requests (default: 1.0)",
    )
    parser.add_argument("--max-retries", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    verbs = [
        verb for verb in json.loads(VERBS_FILE.read_text(encoding="utf-8"))["verbs"]
        if verb.get("rank") is not None and verb["rank"] <= 10_000
    ]
    levels = json.loads(CEFR_FILE.read_text(encoding="utf-8"))["verbs"]
    selected = [
        verb
        for verb in verbs
        if verb["class"] == "weak" and levels.get(verb["lemma"]) == args.level
    ]
    entries = [
        validate(
            verb["lemma"],
            verb["rank"],
            args.refresh,
            max(0.5, args.request_delay),
            args.max_retries,
        )
        for verb in selected
    ]
    summary = {
        status: sum(entry["status"] == status for entry in entries)
        for status in STATUSES
    }
    report = {
        "level": args.level,
        "scope": "earliest Goethe level, top 10,000 weak verbs in german-verbs.json",
        "externalSource": "German Wiktionary, Deutsch Verb Übersicht",
        "checked": len(entries),
        "summary": summary,
        "entries": entries,
    }
    output = args.output or (
        Path(tempfile.gettempdir())
        / "derdiedas"
        / f"weak-conjugation-{args.level.lower()}-report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Checked {len(entries)} {args.level} weak verbs: " + ", ".join(
        f"{count} {status}" for status, count in summary.items()
    ))
    print(f"Wrote {output}")
    for entry in entries:
        if entry["status"] in {"mismatch", "source-unavailable"}:
            print(f"{entry['status']}: {entry['lemma']} {entry['mismatches']}")


if __name__ == "__main__":
    main()
