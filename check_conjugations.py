#!/usr/bin/env python3
"""Generate and validate German conjugation metadata against Wiktionary.

This module provides weak-conjugation rules plus validation helpers for weak
and strong verbs. Source wikitext is cached under .cache/wiktionary so repeated
checks are gentle on the API.
"""

from pathlib import Path
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
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
