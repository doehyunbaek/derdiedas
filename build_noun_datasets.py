#!/usr/bin/env python3
"""Build ranked German common-noun datasets from Leipzig + Kaikki."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

import spacy

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "deu_mixed-typical_2011_1M" / "deu_mixed-typical_2011_1M-sentences.txt"
KAIKKI = ROOT / "kaikki.org-dictionary-German.jsonl"
COUNTS = ROOT / "german-noun-lemma-counts.json"
GENDERS = ROOT / "kaikki-german-noun-genders.json"
GENDER_ORDER = ("masculine", "feminine", "neuter")
ARTICLES = {"masculine": "der", "feminine": "die", "neuter": "das"}
# Prefer selected common standard senses over rare homographs and regional variants.
GENDER_OVERRIDES = {
    "Barometer": {"neuter"},
    "Butter": {"feminine"},
    "Disco": {"feminine"},
    "Embryo": {"neuter"}, # Pascal says "Embryo can be both"
    "Erkenntnis": {"feminine"},
    "Ersparnis": {"feminine"},  # neuter is Austrian
    "Euro": {"masculine"},
    "Foto": {"neuter"},
    "Gründung": {"feminine"},
    "Kunde": {"masculine"},
    "Messer": {"neuter"},
    "Meter": {"masculine"},
    "Mode": {"feminine"},
    "Moment": {"masculine"},
    "Pauschale": {"feminine"},  # neuter is Austrian
    "Polster": {"neuter"},  # masculine is Austrian
    "Poster": {"neuter"},
    "Silvester": {"neuter"},
    "Thermometer": {"neuter"},
    "Tor": {"neuter"},
    "Wende": {"feminine"},
    "Zepter": {"neuter"},
}


def sentence_texts():
    with CORPUS.open(encoding="utf-8") as source:
        for line in source:
            _, separator, text = line.rstrip("\n").partition("\t")
            if separator:
                yield text


def build_counts():
    if COUNTS.exists():
        print(f"Using existing {COUNTS.name}", flush=True)
        return Counter(json.loads(COUNTS.read_text(encoding="utf-8")))

    # The parser and NER are unnecessary for POS identification and lemmatization.
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    counts = Counter()
    documents = 0
    for doc in nlp.pipe(sentence_texts(), batch_size=512):
        documents += 1
        for token in doc:
            if token.pos_ == "NOUN" and token.is_alpha:
                lemma = token.lemma_.strip()
                if lemma and lemma.isalpha():
                    # German common-noun lemmas are conventionally capitalized.
                    lemma = lemma[0].upper() + lemma[1:]
                    counts[lemma] += 1
        if documents % 100_000 == 0:
            print(f"Tagged {documents:,} sentences", flush=True)

    COUNTS.write_text(
        json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved {len(counts):,} lemma counts to {COUNTS.name}", flush=True)
    return counts


def direct_genders(entry):
    """Return genders on lexical noun senses, excluding inflection/alt entries."""
    found = set()
    for sense in entry.get("senses") or ():
        tags = set(sense.get("tags") or ())
        if tags.intersection({"form-of", "alt-of"}):
            continue
        found.update(tags.intersection(GENDER_ORDER))
    # Some entries put lexical tags at entry level.
    tags = set(entry.get("tags") or ())
    found.update(tags.intersection(GENDER_ORDER))
    return found


def build_genders(wanted):
    if GENDERS.exists():
        print(f"Using existing {GENDERS.name}", flush=True)
        raw = json.loads(GENDERS.read_text(encoding="utf-8"))
        return {word: set(values) for word, values in raw.items()}

    genders = defaultdict(set)
    lines = 0
    with KAIKKI.open(encoding="utf-8") as source:
        for line in source:
            lines += 1
            entry = json.loads(line)
            word = entry.get("word")
            if entry.get("pos") == "noun" and word in wanted:
                genders[word].update(direct_genders(entry))
        print(f"Scanned {lines:,} Kaikki entries", flush=True)

    serializable = {
        word: [gender for gender in GENDER_ORDER if gender in values]
        for word, values in genders.items()
        if values
    }
    GENDERS.write_text(
        json.dumps(serializable, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Matched genders for {len(serializable):,} lemmas", flush=True)
    return {word: set(values) for word, values in serializable.items()}


def export(counts, genders):
    ranked = []
    for lemma, frequency in counts.items():
        noun_genders = GENDER_OVERRIDES.get(lemma, genders.get(lemma, ()))
        values = [gender for gender in GENDER_ORDER if gender in noun_genders]
        if not values:
            continue
        ranked.append((lemma, frequency, values))
    ranked.sort(key=lambda item: (-item[1], item[0].casefold(), item[0]))

    for size in (100, 1000, 10000):
        data = []
        for rank, (lemma, frequency, values) in enumerate(ranked[:size], 1):
            data.append({
                "rank": rank,
                "lemma": lemma,
                "articles": [ARTICLES[value] for value in values],
                "genders": [value[0] for value in values],
                "frequency": frequency,
            })
        output = ROOT / f"german-nouns-top-{size}.json"
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(data):,} entries to {output.name}", flush=True)

    print(f"Gender-annotated ranked lemmas available: {len(ranked):,}", flush=True)


def main():
    counts = build_counts()
    genders = build_genders(set(counts))
    export(counts, genders)


if __name__ == "__main__":
    main()
