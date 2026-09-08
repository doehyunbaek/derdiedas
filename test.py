"""Offline regression suite and opt-in Wiktionary conjugation validation.

uv run python test.py
uv run python test.py weak-conjugations --level A1
"""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import tempfile
import unittest

import build_datasets as builder
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

def token(text, lemma, pos, tag, alpha=True):
    return SimpleNamespace(text=text, lemma_=lemma, pos_=pos, tag_=tag, is_alpha=alpha)


class DatasetTests(unittest.TestCase):
    def test_single_stream_collects_all_classes(self):
        documents = iter([
            [token("Häuser", "haus", "NOUN", "NN"), token("ist", "sein", "AUX", "VAFIN"),
             token("im", "in", "ADP", "APPRART"), token(".", ".", "PUNCT", "$.", False)],
            [token("steht", "stehen", "VERB", "VVFIN"), token("auf", "auf", "ADP", "PTKVZ"),
             token("auf", "auf", "ADP", "APPR"), token("bis", "bis", "SCONJ", "KOUS")],
        ])
        data = builder.count_documents(documents, "test", "test")
        self.assertEqual(data["nouns"], {"Haus": 1})
        self.assertEqual(data["verbs"], {"sein": 1, "stehen": 1})
        adp = data["prepositions"]
        self.assertEqual(adp["counts"], {"in": 1, "auf": 1})
        self.assertEqual(adp["surfaceForms"]["in"], {"im": 1})
        self.assertEqual(adp["excludedUses"]["auf"], {"ADP/PTKVZ": 1})
        self.assertEqual(adp["excludedUses"]["bis"], {"SCONJ/KOUS": 1})
        self.assertEqual(adp["sentences"], 2)
        self.assertEqual(adp["alphabeticTokens"], 7)

    def test_shared_cache_needs_no_spacy(self):
        data = {"schemaVersion": builder.SCHEMA_VERSION, "nouns": {}, "verbs": {}, "prepositions": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.json"
            path.write_text(json.dumps(data))
            with patch.object(builder, "COUNTS", path):
                self.assertEqual(builder.build_counts(), data)
                self.assertEqual(builder.selected_counts(("verbs",)), {"verbs": {}})

    def test_missing_legacy_inventory_builds_once(self):
        data = {"nouns": {"Haus": 1}, "verbs": {}, "prepositions": {}}
        with patch.object(builder, "load_shared_counts", return_value=None), \
             patch.object(builder, "legacy_counts", side_effect=[None, {}, {}]), \
             patch.object(builder, "build_counts", return_value=data) as build:
            self.assertEqual(builder.selected_counts(tuple(data)), data)
            build.assert_called_once_with()

    def test_legacy_counts_reused_without_tagging(self):
        with patch.object(builder, "load_shared_counts", return_value=None), \
             patch.object(builder, "legacy_counts", return_value={"gehen": 2}), \
             patch.object(builder, "build_counts") as build:
            self.assertEqual(builder.selected_counts(("verbs",)), {"verbs": {"gehen": 2}})
            build.assert_not_called()

    def test_recount_bypasses_existing_counts(self):
        with patch.object(builder, "build_counts", return_value={"nouns": {}}) as build, \
             patch.object(builder, "legacy_counts") as legacy:
            self.assertEqual(builder.selected_counts(("nouns",), recount=True), {"nouns": {}})
            build.assert_called_once_with(recount=True)
            legacy.assert_not_called()

    def test_export_only_requested_kind(self):
        with patch.object(builder, "selected_counts", return_value={"prepositions": {"test": 1}}) as counts, \
             patch.object(builder, "export_prepositions") as export:
            builder.main(["prepositions"])
            counts.assert_called_once_with(("prepositions",), False)
            export.assert_called_once_with({"test": 1})


class PrepositionTests(unittest.TestCase):
    def test_contractions(self):
        for surface, expected in builder.CONTRACTIONS.items():
            self.assertEqual(builder.normalize_preposition(SimpleNamespace(text=surface.capitalize(), lemma_=surface)), expected)

    def test_lemma_normalization(self):
        self.assertEqual(builder.normalize_preposition(SimpleNamespace(text="Nach", lemma_="nach")), "nach")

    def test_pos_and_particle_separation(self):
        for pos, tag, expected in [("ADP", "APPR", True), ("ADP", "APPRART", True), ("ADP", "APPO", True), ("ADP", "PTKVZ", False), ("PART", "PTKVZ", False), ("SCONJ", "KOUS", False)]:
            self.assertEqual(builder.is_preposition(SimpleNamespace(pos_=pos, tag_=tag)), expected)

    def test_groups(self):
        for lemma, group in {"in": "two-way", "auf": "two-way", "nach": "dative", "bei": "dative", "bis": "accusative", "wegen": "genitive"}.items():
            self.assertEqual(builder.CASE_BY_LEMMA[lemma], group)

    def test_export(self):
        data = {
            "model": "test", "spacyVersion": "test", "sentences": 3,
            "alphabeticTokens": 20, "counts": {"in": 4, "bei": 2, "zzz": 1},
            "surfaceForms": {"in": {"in": 2, "im": 2}, "bei": {"beim": 2}, "zzz": {"zzz": 1}},
            "excludedUses": {"auf": {"ADP/PTKVZ": 1, "PART/PTKVZ": 2}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(builder, "DATA", root):
                builder.export_prepositions(data)
            output = json.loads((root / "german-prepositions.json").read_text())
            self.assertEqual([item["rank"] for item in output["prepositions"]], [1, 2, 3])
            self.assertEqual(output["prepositions"][0]["frequency"], 4)
            self.assertEqual(output["prepositions"][0]["perMillionTokens"], 200000)
            self.assertEqual(output["prepositions"][2]["caseGroup"], "unclassified")
            self.assertEqual(output["excludedUses"], data["excludedUses"])


class SourceAndNounTests(unittest.TestCase):
    def test_download_reuses_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.gz"
            path.write_bytes(b"cached")
            with patch.object(builder.urllib.request, "urlretrieve") as fetch:
                self.assertEqual(builder.download("https://example.invalid/source", path), path)
                fetch.assert_not_called()

    def test_failed_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.gz"
            with patch.object(builder.urllib.request, "urlretrieve", side_effect=OSError("failed")):
                with self.assertRaises(OSError):
                    builder.download("https://example.invalid/source", path)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_explicit_dictionary_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("dictionary.jsonl", "dictionary.jsonl.gz"):
                path = Path(directory) / name
                text = '{"word": "Haus"}\n'
                if name.endswith(".gz"):
                    with builder.gzip.open(path, "wt", encoding="utf-8") as output:
                        output.write(text)
                else:
                    path.write_text(text)
                with builder.open_dictionary(path) as source:
                    self.assertEqual(source.read(), text)

    def test_gender_and_plural_rules(self):
        self.assertEqual(builder.direct_genders({"senses": [{"tags": ["feminine"]}, {"tags": ["form-of", "neuter"]}]}), {"feminine"})
        noun = {"lemma": "Bank"}
        builder.apply_plural_overrides([noun])
        self.assertEqual(noun["plurals"], ["Banken", "Bänke"])
        self.assertEqual(noun["pluralClasses"], ["en", "e"])

    def test_download_command_does_not_tag(self):
        with patch.object(builder, "ensure_corpus", return_value="sentences.txt") as corpus, \
             patch.object(builder, "download", return_value="dictionary.gz") as download, \
             patch.object(builder, "selected_counts") as counts:
            builder.main(["download"])
            corpus.assert_called_once_with()
            download.assert_called_once_with(builder.KAIKKI_URL, builder.KAIKKI)
            counts.assert_not_called()

    def test_plural_command_dispatch(self):
        with patch.object(builder, "enrich_noun_plurals") as enrich:
            builder.main(["plurals", "--refresh-plurals"])
            enrich.assert_called_once_with(None, True)


ROOT = Path(__file__).resolve().parent
VERBS_FILE = ROOT / "data" / "german-verbs.json"
CEFR_FILE = ROOT / "data" / "goethe-cefr-levels.json"
STATUSES = ("pass", "mismatch", "source-unavailable")


def run_weak_conjugations(argv=None):
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
    args = parser.parse_args(argv)

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


# --- Conjugation rules and cached Wiktionary validation ---

CACHE = ROOT / ".cache" / "wiktionary"
API = "https://de.wiktionary.org/w/api.php"
PERSONS = ("ich", "du", "er/sie/es", "wir", "ihr", "sie/Sie")
INSEPARABLE_PREFIXES = ("be", "emp", "ent", "er", "ge", "miss", "ver", "zer")
SEPARABLE_PREFIXES = (
    "zurück", "zusammen", "weiter", "wieder", "entgegen", "fest", "fort",
    "heim", "her", "hin", "hoch", "los", "nach", "nieder", "statt", "teil",
    "vor", "weg", "zu", "ab", "an", "auf", "aus", "bei", "ein", "mit",
)
# Ambiguous prefixes such as wieder- and über- require lexical information.
LEXICALLY_INSEPARABLE_VERBS = {"wiederholen", "übernachten"}
# Accidental prefix-shaped beginnings which are not productive prefixes.
LEXICALLY_UNPREFIXED_VERBS = {"antworten"}
# Compounds whose first verbal element behaves like a separable prefix.
SEPARABLE_COMPOUNDS = {"kennenlernen": ("kennen", "lernen")}
LAST_REQUEST_AT = 0.0

REFERENCE_FIELDS = {
    "present.ich": "Präsens_ich",
    "present.du": "Präsens_du",
    "present.er/sie/es": "Präsens_er, sie, es",
    "past.ich": "Präteritum_ich",
    "subjunctive2.ich": "Konjunktiv II_ich",
    "participle2": "Partizip II",
}

# --- Validation workflow --------------------------------------------------

def validate(lemma, rank, refresh=False, minimum_interval=1.0, max_retries=7):
    generated = conjugate_weak_verb(lemma)
    source = wiktionary_overview(fetch_wikitext(
        lemma, refresh, minimum_interval, max_retries
    ))
    if "error" in source:
        status = "source-unavailable"
        mismatches = []
    else:
        mismatches = []
        for path in REFERENCE_FIELDS:
            expected = nested_get(generated, path)
            attested = nested_get(source, path)
            # A dash marks a semantically unavailable person on defective
            # verbs such as impersonal regnen, not a contrary inflection.
            matches = expected in attested if isinstance(attested, list) else expected == attested
            if attested and attested != "—" and not matches:
                mismatches.append({"form": path, "generated": expected, "wiktionary": attested})
        status = "pass" if not mismatches else "mismatch"
    return {
        "lemma": lemma,
        "rank": rank,
        "familyStem": generated["familyStem"],
        "prefixBehavior": generated["prefixBehavior"],
        "endingPattern": generated["endingPattern"],
        "status": status,
        "generated": generated,
        "wiktionary": source,
        "mismatches": mismatches,
        "sourceUrl": f"https://de.wiktionary.org/wiki/{urllib.parse.quote(lemma)}",
    }

# --- Public conjugation API -----------------------------------------------

def check_strong_verb(
    lemma, expected, family_stem=None,
    refresh=False, minimum_interval=1.0, max_retries=7
):
    """Compare our stored strong conjugation metadata with Wiktionary.

    ``expected`` is the local dataset's ``principalParts`` object. Values are
    arrays because Wiktionary and Kaikki can attest multiple accepted forms.
    """
    family_stem = family_stem or lemma
    source_url = f"https://de.wiktionary.org/wiki/{urllib.parse.quote(lemma)}"
    source = wiktionary_overview(fetch_wikitext(
        lemma, refresh, minimum_interval, max_retries
    ))
    if "error" in source:
        return {
            "lemma": lemma,
            "familyStem": family_stem,
            "status": "source-unavailable",
            "wiktionary": source,
            "sourceUrl": source_url,
        }

    attested = {
        "present3": source.get("present", {}).get("er/sie/es", []),
        "preterite": source.get("past", {}).get("ich", []),
        "participle2": source.get("participle2", []),
        "subjunctive2": source.get("subjunctive2", {}).get("ich", []),
        "imperativeSingular": source.get("imperative", {}).get("singular", []),
        "imperativePlural": source.get("imperative", {}).get("plural", []),
        "auxiliaries": source.get("auxiliaries", []),
    }
    mismatches = []
    variant_differences = []
    for field, local_forms in expected.items():
        wiktionary_forms = attested.get(field, [])
        local_set = set(local_forms)
        wiktionary_set = set(wiktionary_forms)
        if not local_set.intersection(wiktionary_set):
            mismatches.append({
                "field": field,
                "dataset": local_forms,
                "wiktionary": wiktionary_forms,
            })
        elif local_set != wiktionary_set:
            variant_differences.append({
                "field": field,
                "datasetOnly": sorted(local_set - wiktionary_set),
                "wiktionaryOnly": sorted(wiktionary_set - local_set),
            })

    local_preterite = expected.get("preterite", [None])[0]
    wiktionary_preterite = attested["preterite"][0] if attested["preterite"] else None
    vowel_change = None
    if local_preterite and wiktionary_preterite:
        infinitive_vowels = vowel_pattern(stem_of(family_stem))
        vowel_change = {
            "dataset": (
                infinitive_vowels,
                vowel_pattern(strong_form_stem(lemma, family_stem, local_preterite)),
            ),
            "wiktionary": (
                infinitive_vowels,
                vowel_pattern(strong_form_stem(lemma, family_stem, wiktionary_preterite)),
            ),
        }

    return {
        "lemma": lemma,
        "familyStem": family_stem,
        "status": "pass" if not mismatches else "mismatch",
        "matches": not mismatches,
        "dataset": expected,
        "wiktionary": attested,
        "vowelChange": vowel_change,
        "mismatches": mismatches,
        "variantDifferences": variant_differences,
        "sourceUrl": source_url,
    }


def conjugate_weak_verb(lemma):
    """Conjugate a weak verb by resolving prefix behavior, then stem endings."""
    # 1. Resolve lexical prefix behavior and the base that receives endings.
    if lemma in LEXICALLY_INSEPARABLE_VERBS:
        prefix_behavior, prefix, base = "untrennbar", "", lemma
    elif lemma in LEXICALLY_UNPREFIXED_VERBS:
        prefix_behavior, prefix, base = "none", "", lemma
    elif lemma in SEPARABLE_COMPOUNDS:
        prefix, base = SEPARABLE_COMPOUNDS[lemma]
        prefix_behavior = "trennbar"
    elif prefix := next((value for value in SEPARABLE_PREFIXES if lemma.startswith(value)), ""):
        prefix_behavior, base = "trennbar", lemma[len(prefix):]
    elif lemma.endswith("ieren"):
        prefix_behavior, prefix, base = "-ieren", "", lemma
    elif lemma.startswith(INSEPARABLE_PREFIXES):
        prefix_behavior, prefix, base = "untrennbar", "", lemma
    else:
        prefix_behavior, prefix, base = "none", "", lemma

    # 2. Resolve the endings from the final sounds of the conjugated stem.
    stem = stem_of(base)
    ending_pattern = stem_paradigm(base)
    endings = {
        "regular": ("e", "st", "t", "en", "t", "en"),
        "sibilant-stem": ("e", "t", "t", "en", "t", "en"),
        "inserted-e": ("e", "est", "et", "en", "et", "en"),
    }[ending_pattern]
    present = [stem + ending for ending in endings]
    past_ich = stem + ("ete" if ending_pattern == "inserted-e" else "te")
    past = [past_ich, past_ich + "st", past_ich, past_ich + "n", past_ich + "t", past_ich + "n"]

    # 3. Apply separable-prefix placement to finite forms.
    if prefix_behavior == "trennbar":
        present = [form + " " + prefix for form in present]
        past = [form + " " + prefix for form in past]

    # 4. Form the participle according to prefix behavior.
    participle_suffix = "et" if ending_pattern == "inserted-e" else "t"
    if prefix_behavior == "trennbar":
        participle = prefix + "ge" + stem + participle_suffix
    elif prefix_behavior in {"untrennbar", "-ieren"}:
        participle = stem + participle_suffix
    else:
        participle = "ge" + stem + participle_suffix

    return {
        "familyStem": base,
        "prefixBehavior": prefix_behavior,
        "endingPattern": ending_pattern,
        "present": dict(zip(PERSONS, present)),
        "past": dict(zip(PERSONS, past)),
        "subjunctive2": dict(zip(PERSONS, past)),
        "participle2": participle,
    }

# --- Wiktionary reference data --------------------------------------------

def fetch_wikitext(lemma, refresh=False, minimum_interval=1.0, max_retries=7):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{urllib.parse.quote(lemma, safe='')}.txt"
    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8")
    query = urllib.parse.urlencode({
        "action": "parse", "page": lemma, "prop": "wikitext",
        "format": "json", "redirects": "1", "maxlag": "5",
    })
    request = urllib.request.Request(f"{API}?{query}", headers={
        "User-Agent": "derdiedas-conjugation-validator/1.0 (educational word-list validation)"
    })
    for attempt in range(max_retries + 1):
        wait_for_request_slot(minimum_interval)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if "error" in payload:
                code = payload["error"].get("code", "api-error")
                if code == "maxlag" and attempt < max_retries:
                    delay = min(60.0, 2 ** attempt) + random.uniform(0, 0.5)
                    print(f"Wiktionary maxlag for {lemma}; retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                return {"error": f"{code}: {payload['error'].get('info', '')}"}
            text = payload["parse"]["wikitext"]["*"]
            cache_file.write_text(text, encoding="utf-8")
            return text
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt >= max_retries:
                return {"error": f"HTTP {error.code} after {attempt + 1} attempts"}
            retry_after = error.headers.get("Retry-After")
            try:
                server_delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                server_delay = 0.0
            delay = max(server_delay, min(60.0, 2 ** attempt)) + random.uniform(0, 0.5)
            print(f"Wiktionary HTTP {error.code} for {lemma}; retrying in {delay:.1f}s")
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt >= max_retries:
                return {"error": f"{error} after {attempt + 1} attempts"}
            delay = min(60.0, 2 ** attempt) + random.uniform(0, 0.5)
            print(f"Wiktionary network error for {lemma}; retrying in {delay:.1f}s")
            time.sleep(delay)
    return {"error": "retry loop exhausted"}

def wiktionary_overview(wikitext):
    if isinstance(wikitext, dict):
        return wikitext
    start = wikitext.find("{{Deutsch Verb Übersicht")
    if start < 0:
        return {"error": "Deutsch Verb Übersicht not found"}
    block = wikitext[start:wikitext.find("\n}}", start)]
    fields = {}
    for line in block.splitlines()[1:]:
        if not line.startswith("|") or "=" not in line:
            continue
        key, value = line[1:].split("=", 1)
        fields[key.strip()] = clean_template_value(value)
    variants = lambda name: [
        value for key, value in fields.items()
        if (key == name or key.startswith(name + "*")) and value and value != "—"
    ]
    return {
        "present": {
            "ich": fields.get("Präsens_ich"),
            "du": fields.get("Präsens_du"),
            "er/sie/es": variants("Präsens_er, sie, es"),
        },
        "past": {"ich": variants("Präteritum_ich")},
        "subjunctive2": {"ich": variants("Konjunktiv II_ich")},
        "participle2": variants("Partizip II"),
        "imperative": {
            "singular": variants("Imperativ Singular"),
            "plural": variants("Imperativ Plural"),
        },
        "auxiliaries": variants("Hilfsverb"),
    }

def clean_template_value(value):
    value = re.sub(r"<!--.*?-->", "", value)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"\[\[(?:[^]|]*\|)?([^]]+)\]\]", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()

def wait_for_request_slot(minimum_interval):
    """Keep requests spaced out even when retries and normal calls interleave."""
    global LAST_REQUEST_AT
    remaining = minimum_interval - (time.monotonic() - LAST_REQUEST_AT)
    if remaining > 0:
        time.sleep(remaining)
    LAST_REQUEST_AT = time.monotonic()

# --- Low-level helpers ----------------------------------------------------

def nested_get(data, path):
    value = data
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def strong_form_stem(lemma, family_stem, form):
    """Isolate the family stem from an attested finite compound form."""
    word = form.split()[0]
    prefix = lemma[:-len(family_stem)] if lemma.endswith(family_stem) else ""
    if prefix and word.startswith(prefix):
        word = word[len(prefix):]
    return word[:-1] if word.endswith("e") else word


def vowel_pattern(stem):
    """Return the stem's vowel groups for principal-part comparison."""
    return tuple(re.findall(r"[aeiouäöüy]+", stem.lower()))


def stem_paradigm(lemma):
    stem = stem_of(lemma)
    if inserts_e(stem):
        return "inserted-e"
    if stem.endswith(("s", "ß", "x", "z")):
        return "sibilant-stem"
    return "regular"

def stem_of(infinitive):
    if infinitive.endswith(("eln", "ern")):
        return infinitive[:-1]  # handeln -> handel-, wandern -> wander-
    if infinitive.endswith("en"):
        return infinitive[:-2]
    return infinitive

def inserts_e(stem):
    """Whether dental endings need a linking e: arbeit-et, öffn-et."""
    return stem.endswith(("d", "t")) or bool(
        re.search(r"[^aeiouäöüyhlr][mn]$", stem)
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["weak-conjugations"]:
        run_weak_conjugations(sys.argv[2:])
    else:
        unittest.main()
