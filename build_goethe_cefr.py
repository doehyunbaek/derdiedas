#!/usr/bin/env python3
"""Extract noun and verb lemmas from the Goethe A1–B1 Wortlisten.

The output contains all recognized Goethe headwords, not only this site's top
10,000 words. It also records which headwords are absent from those datasets.
Levels are earliest occurrence; the UI treats them cumulatively.
"""

from pathlib import Path
import json
import re
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "goethe-cefr-levels.json"
SOURCES = {
    "A1": "https://www.goethe.de/pro/relaunch/prf/de/A1_SD1_Wortliste_02.pdf",
    "A2": "https://www.goethe.de/pro/relaunch/prf/de/Goethe-Zertifikat_A2_Wortliste.pdf",
    "B1": "https://www.goethe.de/pro/relaunch/prf/de/Goethe-Zertifikat_B1_Wortliste.pdf",
}
HILFSVERBEN = {
    "sein", "haben", "werden", "können", "müssen", "wollen", "sollen",
    "dürfen", "mögen",
}


def load_lemmas(filename, key=None):
    data = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    entries = data[key] if key else data
    return {entry["lemma"] for entry in entries}


def pdf_text(url, directory, level):
    pdf = directory / f"{level.lower()}.pdf"
    text = directory / f"{level.lower()}.txt"
    urllib.request.urlretrieve(url, pdf)
    subprocess.run(["pdftotext", "-raw", pdf, text], check=True)
    return text.read_text(encoding="utf-8")


def extract_headwords(text, verb_inventory, level):
    """Extract noun entries directly and recognize verbs against a broad lexicon."""
    # Exclude introductory prose and appendices, which can also begin lines
    # with dictionary words. These markers surround each alphabetical list.
    if level == "A1":
        # A1's word-group inventory precedes the alphabetical list and already
        # contains level headwords such as "der Abend".
        text = text[text.find("\nZahlen\nWortgruppenliste\n"):]
        text = text[:text.find("\nLiteratur\n")]
    elif level == "A2":
        # A2 likewise includes categorized vocabulary before its alphabetical
        # list (then repeats many entries there).
        text = text[text.find("\nAnweisungssprache\nzur Prüfung\n"):]

    # pdftotext sometimes wraps a headword at a hyphen ("Straßen-\nbahn").
    # Rejoin that specific article + headword shape before extracting it.
    text = re.sub(
        r"^((?:der|die|das)\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]*)-\s*\n\s*([a-zäöüß][A-Za-zÄÖÜäöüß-]*)",
        r"\1\2",
        text,
        flags=re.MULTILINE,
    )
    found_nouns = set()
    for match in re.finditer(
        r"^(?:der|die|das)\s+([A-ZÄÖÜ][A-Za-zÀ-ÖØ-öø-ÿẞ-]*)",
        text,
        re.MULTILINE,
    ):
        candidate = match.group(1)
        # An ending hyphen means pdftotext split the headword around example
        # text; do not mistake that incomplete fragment for a noun.
        if not candidate.endswith("-"):
            found_nouns.add(candidate)

    # The broad class inventory has 10,000+ dictionary verbs, including words
    # outside this site's frequency cutoff. A1 prints bare infinitives followed
    # by examples; A2/B1 entries use a comma before their principal parts.
    found_verbs = set()
    separator = r"(?:,|\s)" if level == "A1" else r","
    for lemma in verb_inventory:
        if " " in lemma:
            continue
        escaped = re.escape(lemma)
        pattern = rf"^(?:\(sich\)\s+|sich\s+)?{escaped}(?:\s+\(sich\))?{separator}"
        if re.search(pattern, text, re.MULTILINE):
            found_verbs.add(lemma)
    return found_nouns, found_verbs


def main():
    top_nouns = load_lemmas("german-nouns.json")
    verb_data = json.loads((ROOT / "german-verbs.json").read_text(encoding="utf-8"))
    verb_records = verb_data["verbs"]
    top_verbs = {
        entry["lemma"] for entry in verb_records
        if entry.get("rank") is not None and entry["rank"] <= 10_000
    }
    verb_inventory = {entry["lemma"] for entry in verb_records if entry.get("class") != "unknown"}
    earliest_nouns = {}
    earliest_verbs = {}

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for level, url in SOURCES.items():
            text = pdf_text(url, directory, level)
            level_nouns, level_verbs = extract_headwords(text, verb_inventory, level)
            for lemma in level_nouns:
                earliest_nouns.setdefault(lemma, level)
            for lemma in level_verbs:
                earliest_verbs.setdefault(lemma, level)
            print(f"{level}: extracted {len(level_nouns)} nouns and {len(level_verbs)} verbs")

    missing_nouns = sorted(set(earliest_nouns) - top_nouns)
    missing_verbs = sorted(set(earliest_verbs) - top_verbs - HILFSVERBEN)
    data = {
        "source": "Goethe-Institut A1, A2, and B1 Wortlisten",
        "nouns": dict(sorted(earliest_nouns.items())),
        "verbs": dict(sorted(earliest_verbs.items())),
        "notInTop10000": {
            "nouns": missing_nouns,
            "verbs": missing_verbs,
        },
    }
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(earliest_nouns)} nouns and {len(earliest_verbs)} verbs to {OUTPUT.name}")
    print(f"Not in top 10,000: {len(missing_nouns)} nouns and {len(missing_verbs)} verbs")
    if missing_nouns:
        print("Missing nouns: " + ", ".join(missing_nouns))
    if missing_verbs:
        print("Missing verbs: " + ", ".join(missing_verbs))


if __name__ == "__main__":
    main()
