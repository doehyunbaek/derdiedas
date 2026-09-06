#!/usr/bin/env python3
"""Enrich german-nouns.json with plural forms from the Kaikki German dictionary."""

from collections import defaultdict
from pathlib import Path
import argparse
import gzip
import json
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
NOUNS = ROOT / "german-nouns.json"
KAIKKI_URL = "https://kaikki.org/dictionary/German/kaikki.org-dictionary-German.jsonl.gz"


# Reviewed common learner senses, rather than whichever homograph occurs first.
# An empty list means no usual plural in the selected sense, not that a specialist
# or historical plural can never occur.
PLURAL_OVERRIDES = {
    "Bank": ["Banken", "Bänke"],
    "Wort": ["Wörter", "Worte"],
    "Schild": ["Schilder", "Schilde"],
    "Wasser": ["Wasser", "Wässer"],
    "Post": [],  # mail; not posts or historical postal services
    "Reis": [],  # rice; not das Reis (twig)
    "Gepäck": [],
    "Unterricht": [],
    "Kleidung": [],
    "Milch": [],
    "Sport": [],
}
PLURAL_ARTICLES = {"Schild": {"Schilder": "das", "Schilde": "der"}}


def set_plurals(noun, forms):
    if forms:
        noun["plurals"] = forms
        noun["pluralClasses"] = list(dict.fromkeys(plural_class(noun["lemma"], form) for form in forms))
    else:
        noun.pop("plurals", None)
        noun.pop("pluralClasses", None)
    noun.pop("pluralArticles", None)
    if noun["lemma"] in PLURAL_ARTICLES:
        noun["pluralArticles"] = PLURAL_ARTICLES[noun["lemma"]]


def apply_plural_overrides(nouns):
    for noun in nouns:
        if noun["lemma"] in PLURAL_OVERRIDES:
            set_plurals(noun, PLURAL_OVERRIDES[noun["lemma"]])


def plural_class(singular, plural):
    singular = singular.casefold()
    plural = plural.casefold()
    deumlaut = str.maketrans({"ä": "a", "ö": "o", "ü": "u"})

    def same_stem(stem):
        return stem == singular or stem.translate(deumlaut) == singular.translate(deumlaut)

    if same_stem(plural):
        return "unchanged"
    if plural.endswith("s"):
        return "s"
    if plural.endswith(("en", "n")):
        return "en"
    if plural.endswith("er"):
        return "er"
    if plural.endswith("e"):
        return "e"
    return "other"


def entry_plurals(entry):
    """Prefer dictionary headword plurals, falling back to nominative table forms."""
    headword_forms = []
    nominative_forms = []
    for item in entry.get("forms") or ():
        tags = set(item.get("tags") or ())
        form = item.get("form")
        if not form or "plural" not in tags:
            continue
        if form.startswith("die "):
            form = form[4:]
        if item.get("source") != "declension":
            headword_forms.append(form)
        elif "nominative" in tags:
            nominative_forms.append(form)
    return headword_forms or nominative_forms


def open_dictionary(path):
    if path:
        return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")
    temporary = tempfile.NamedTemporaryFile(prefix="kaikki-german-", suffix=".jsonl.gz", delete=False)
    temporary.close()
    print(f"Downloading {KAIKKI_URL}", flush=True)
    urllib.request.urlretrieve(KAIKKI_URL, temporary.name)
    return gzip.open(temporary.name, "rt", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dictionary", nargs="?", help="Kaikki German JSONL or JSONL.gz (downloaded when omitted)")
    parser.add_argument("--refresh-plurals", action="store_true", help="Apply reviewed corrections without downloading the dictionary")
    args = parser.parse_args()

    nouns = json.loads(NOUNS.read_text(encoding="utf-8"))
    if args.refresh_plurals:
        apply_plural_overrides(nouns)
        NOUNS.write_text(json.dumps(nouns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Applied reviewed plural corrections")
        return
    wanted = {noun["lemma"] for noun in nouns}
    plurals = defaultdict(list)
    with open_dictionary(args.dictionary) as source:
        for line in source:
            entry = json.loads(line)
            lemma = entry.get("word")
            if entry.get("pos") != "noun" or lemma not in wanted:
                continue
            for form in entry_plurals(entry):
                if form not in plurals[lemma]:
                    plurals[lemma].append(form)

    for noun in nouns:
        # Kaikki orders the usual dictionary plural first; later forms are often
        # rare, archaic, or restricted alternatives (for example Jahr after numerals).
        forms = plurals[noun["lemma"]][:1]
        set_plurals(noun, forms)

    apply_plural_overrides(nouns)
    NOUNS.write_text(json.dumps(nouns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Added plurals to {sum(bool(noun.get('plurals')) for noun in nouns):,}/{len(nouns):,} nouns")


if __name__ == "__main__":
    main()
