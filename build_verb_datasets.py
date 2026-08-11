#!/usr/bin/env python3
"""Build ranked and conjugation-classified German verb datasets."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import re

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "deu_mixed-typical_2011_1M" / "deu_mixed-typical_2011_1M-sentences.txt"
UNIFIED = ROOT / "german-verbs.json"
KAIKKI = ROOT / "kaikki.org-dictionary-German.jsonl"
KAIKKI_VERBS = ROOT / ".cache" / "kaikki-german-verbs.jsonl.gz"

# These verbs have their own curated Hilfsverben tab.
EXCLUDED_LEMMAS = {
    "sein", "haben", "werden", "können", "müssen", "wollen", "sollen",
    "dürfen", "mögen",
    # Compounds treated here as constructions with the excluded auxiliary.
    "loswerden", "fertigwerden",
}
# Frequent tagging/spelling artifacts which unambiguously belong to an
# excluded auxiliary lemma but are not represented as forms in Wiktionary.
EXCLUDED_FORMS = {
    "mussen", "mussn", "muessen", "mußt", "müs", "kannstn",
}
# Strict strong base verbs, using Duden's criterion (ablaut plus an -en
# participle) and the numbered inventory at deutschplus.net. The inventory's
# weak/mixed section (mahlen onward) is deliberately excluded. Compounds are
# handled by inheritance below rather than counted as independent base verbs.
# Sources:
# https://www.duden.de/sprachwissen/sprachratgeber/Starke-und-schwache-Verben
# https://www.deutschplus.net/pages/Tabelle_starker_Verben
STRONG_BASES = {
    "backen", "fahren", "graben", "laden", "schaffen", "schlagen", "tragen",
    "wachsen", "waschen", "blasen", "braten", "fallen", "halten", "lassen",
    "raten", "schlafen", "empfangen", "fangen", "geschehen", "lesen", "sehen",
    "befehlen", "empfehlen", "stehlen", "gebären", "essen", "fressen",
    "genesen", "geben", "messen", "treten", "vergessen", "bergen", "bersten",
    "brechen", "erschrecken", "gelten", "helfen", "nehmen", "schelten",
    "sprechen", "stechen", "sterben", "treffen", "verderben", "werben",
    "werfen", "bewegen", "dreschen", "fechten", "flechten", "heben", "melken",
    "pflegen", "quellen", "scheren", "schmelzen", "schwellen", "weben",
    "gären", "wägen", "gehen", "stehen", "biegen", "bieten", "fliegen",
    "fliehen", "fließen", "frieren", "genießen", "gießen", "kriechen",
    "riechen", "schieben", "schießen", "schließen", "sieden", "sprießen",
    "stieben", "triefen", "verdrießen", "verlieren", "wiegen", "ziehen",
    "liegen", "beginnen", "gewinnen", "schwimmen", "rinnen", "sinnen",
    "spinnen", "glimmen", "klimmen", "binden", "dingen", "dringen", "finden",
    "gelingen", "klingen", "ringen", "schlingen", "schwinden", "schwingen",
    "singen", "sinken", "springen", "stinken", "trinken", "winden", "wringen",
    "zwingen", "bitten", "sitzen", "schinden", "bleiben", "gedeihen", "leihen",
    "meiden", "preisen", "reiben", "scheiden", "scheinen", "schreiben",
    "schreien", "schweigen", "speien", "steigen", "treiben", "weisen",
    "verzeihen", "beißen", "bleichen", "gleichen", "gleiten", "greifen",
    "kneifen", "leiden", "pfeifen", "reißen", "reiten", "scheißen",
    "schleichen", "schleifen", "schmeißen", "schneiden", "schreiten",
    "streichen", "streiten", "weichen", "heißen", "saufen", "saugen",
    "schnauben", "hauen", "laufen", "kommen", "stoßen", "tun", "rufen",
    "hängen", "erlöschen", "schwören", "lügen", "trügen",
}
CLASS_OVERRIDES = {
    # Kaikki merges homographs/legacy metadata here, but modern German reisen
    # is strictly weak: reiste, gereist.
    "reisen": "weak",
    # Modern standard usage is weak (schreckte, geschreckt); the strong
    # paradigm schrak/geschrocken is dated and should not define this binary UI.
    "schrecken": "weak",
}
# Duden's "unregelmäßiges Verb" category: the stem changes beyond the vowel
# alternation that defines a strong verb. Transparent derivatives inherit it.
# Wenden remains weak in its usual modern transitive/reflexive senses.
IRREGULAR_BASES = {
    "bringen", "denken", "kennen", "nennen", "rennen", "brennen", "senden",
    "wissen", "gehen", "stehen",
}
INSEPARABLE_PREFIXES = ("be", "emp", "ent", "er", "ge", "miss", "ver", "zer")

STRONG_STEM_OVERRIDES = {
    # Relics/compounds whose shared base is not a literal suffix in the modern
    # spelling, or whose base has its own Hilfsverben tab.
    "auserkiesen": "kiesen",
    "erkiesen": "kiesen",
    "misslingen": "gelingen",
    "verschleißen": "verschleißen",
    "zeihen": "zeihen",
}
PREFIXES = (
    "zusammen", "zwischen", "zurück", "weiter", "wieder", "hinter",
    "durch", "gegen", "nieder", "statt", "unter", "empor", "entgegen", "miss",
    "ab", "an", "auf", "aus", "be", "bei", "dar", "ein", "ent", "er",
    "fest", "fort", "frei", "heim", "her", "hin", "hoch", "los", "mit", "zer",
    "nach", "teil", "über", "um", "ver", "vor", "weg", "wider", "zu",
)


def sentence_texts():
    with CORPUS.open(encoding="utf-8") as source:
        for line in source:
            _, separator, text = line.rstrip("\n").partition("\t")
            if separator:
                yield text


def unified_records():
    if not UNIFIED.exists():
        return []
    return json.loads(UNIFIED.read_text(encoding="utf-8"))["verbs"]


def build_counts():
    records = unified_records()
    if records:
        print(f"Using raw counts from {UNIFIED.name}", flush=True)
        return Counter({
            record["lemma"]: record["rawFrequency"]
            for record in records if record.get("rawFrequency") is not None
        })

    import spacy
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    counts = Counter()
    documents = 0
    for doc in nlp.pipe(sentence_texts(), batch_size=512):
        documents += 1
        for token in doc:
            if token.pos_ in {"VERB", "AUX"} and token.is_alpha:
                lemma = token.lemma_.strip().lower()
                if lemma and lemma.isalpha():
                    counts[lemma] += 1
        if documents % 100_000 == 0:
            print(f"Tagged {documents:,} sentences", flush=True)

    return counts


def kaikki_lines():
    """Read either the noun-pipeline JSONL file or its downloaded gzip form."""
    if KAIKKI.exists():
        return KAIKKI.open(encoding="utf-8")
    compressed = KAIKKI.with_suffix(KAIKKI.suffix + ".gz")
    if compressed.exists():
        return gzip.open(compressed, "rt", encoding="utf-8")
    raise FileNotFoundError(
        f"Expected {KAIKKI.name} or {compressed.name} to build verb metadata"
    )


def build_principal_parts():
    """Extract independent local principal parts from the Kaikki verb export."""
    if not KAIKKI_VERBS.exists():
        print(f"No {KAIKKI_VERBS}; preserving existing principal parts", flush=True)
        return {
            record["lemma"]: record["principalParts"]
            for record in unified_records() if record.get("principalParts")
        }

    candidates = defaultdict(lambda: {
        "present3": Counter(),
        "preterite": Counter(),
        "participle2": Counter(),
        "subjunctive2": Counter(),
        "imperativeSingular": Counter(),
        "imperativePlural": Counter(),
        "auxiliaries": Counter(),
    })
    with gzip.open(KAIKKI_VERBS, "rt", encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            lemma = (entry.get("word") or "").lower()
            if entry.get("pos") != "verb" or not lexical_entry(entry, lemma):
                continue
            for form in entry.get("forms") or ():
                spelling = (form.get("form") or "").lower()
                tags = set(form.get("tags") or ())
                if not spelling:
                    continue
                weight = 3 if form.get("source") == "conjugation" else 1
                if {"indicative", "present", "singular", "third-person"} <= tags:
                    candidates[lemma]["present3"][spelling] += weight
                if {
                    "first-person", "indicative", "preterite", "singular"
                } <= tags:
                    candidates[lemma]["preterite"][spelling] += weight
                if {"participle", "past"} <= tags:
                    candidates[lemma]["participle2"][spelling] += weight
                if {
                    "first-person", "singular", "subjunctive-ii"
                } <= tags and "future" not in tags:
                    candidates[lemma]["subjunctive2"][spelling] += weight
                if {"imperative", "second-person", "singular"} <= tags:
                    candidates[lemma]["imperativeSingular"][spelling] += weight
                if {"imperative", "second-person", "plural"} <= tags:
                    candidates[lemma]["imperativePlural"][spelling] += weight

            for template in entry.get("head_templates") or ():
                if template.get("name") != "de-verb":
                    continue
                expansion = template.get("expansion") or ""
                match = re.search(r"auxiliary (haben|sein)(?: or (haben|sein))?", expansion)
                if match:
                    for auxiliary in match.groups():
                        if auxiliary:
                            candidates[lemma]["auxiliaries"][auxiliary] += 1

    result = {}
    for lemma, forms in candidates.items():
        if forms["preterite"] and forms["participle2"]:
            result[lemma] = {
                key: [form for form, _ in values.most_common()]
                for key, values in forms.items()
                if values
            }
    print(f"Extracted principal parts for {len(result):,} verbs", flush=True)
    return result


def lexical_entry(entry, lemma):
    senses = entry.get("senses") or ()
    if any(
        not set(sense.get("tags") or ()).intersection({"form-of", "alt-of"})
        for sense in senses
    ):
        return True
    return any(
        form.get("form", "").lower() == lemma
        and "infinitive" in (form.get("tags") or ())
        and form.get("source") == "conjugation"
        for form in entry.get("forms") or ()
    )


def resolve_class(entry):
    """Resolve explicit tags, then use principal parts where necessary."""
    tags = set(entry.get("tags") or ())
    for sense in entry.get("senses") or ():
        tags.update(sense.get("tags") or ())
    forms = entry.get("forms") or ()
    table_tags = {
        form.get("form", "").lower()
        for form in forms
        if "table-tags" in (form.get("tags") or ())
    }

    if "mixed" in tags or "mixed" in table_tags:
        return "mixed"
    strong = "strong" in tags or "strong" in table_tags
    weak = "weak" in tags or "weak" in table_tags
    irregular = "irregular" in tags or "irregular weak" in table_tags
    if weak and irregular or strong and weak:
        return "mixed"
    if strong:
        return "strong"
    if weak:
        return "weak"

    # Some entries provide no class tag but do provide diagnostic principal
    # parts. A -te preterite and -t participle are weak; an -en participle
    # with a non--te preterite is strong.
    past = [
        form.get("form", "").lower() for form in forms
        if "past" in (form.get("tags") or ())
        or "preterite" in (form.get("tags") or ())
    ]
    participles = [
        form.get("form", "").lower() for form in forms
        if {"participle", "past"} <= set(form.get("tags") or ())
    ]
    if any(form.endswith("te") for form in past) and any(
        form.endswith("t") for form in participles
    ):
        return "weak"
    if past and any(form.endswith("en") for form in participles):
        return "strong"
    return None


def unify_spelling_variants(counts):
    """Merge old/Swiss ss spellings with their standard ß equivalent.

    Only spellings actually attested in the normalized corpus are grouped, so
    genuine ss words such as lassen and müssen are not rewritten. The most
    frequent attested spelling becomes canonical and receives the total count.
    """
    groups = defaultdict(list)
    for lemma, frequency in counts.items():
        groups[lemma.replace("ß", "ss")].append((lemma, frequency))

    unified = Counter()
    merged = 0
    for variants in groups.values():
        canonical, _ = min(
            variants,
            key=lambda item: (-item[1], -item[0].count("ß"), item[0]),
        )
        unified[canonical] += sum(frequency for _, frequency in variants)
        merged += len(variants) - 1
    return unified, merged


def build_metadata(raw_counts):
    records = unified_records()
    if records:
        print(f"Using normalized counts and classes from {UNIFIED.name}", flush=True)
        normalized = Counter({
            record["lemma"]: record["frequency"]
            for record in records if record.get("frequency")
        })
        classes = {
            record["lemma"]: record["kaikkiClass"]
            for record in records if record.get("kaikkiClass")
        }
        return normalized, classes

    wanted = set(raw_counts)
    lexical = set()
    form_targets = defaultdict(Counter)
    classes = {}

    with kaikki_lines() as source:
        for line in source:
            entry = json.loads(line)
            if entry.get("pos") != "verb":
                continue
            lemma = (entry.get("word") or "").lower()
            is_lexical = lexical_entry(entry, lemma)
            if is_lexical:
                lexical.add(lemma)
                verb_class = resolve_class(entry)
                if verb_class:
                    classes[lemma] = verb_class
                # Conjugation tables are stronger evidence than form pages.
                for form in entry.get("forms") or ():
                    spelling = (form.get("form") or "").lower()
                    tags = set(form.get("tags") or ())
                    if spelling in wanted and tags.intersection({
                        "present", "preterite", "past", "participle",
                        "subjunctive-i", "subjunctive-ii",
                    }):
                        form_targets[spelling][lemma] += 3

            if lemma in wanted:
                for sense in entry.get("senses") or ():
                    if "form-of" not in (sense.get("tags") or ()):
                        continue
                    for relation in sense.get("form_of") or ():
                        target = (relation.get("word") or "").lower()
                        if target:
                            form_targets[lemma][target] += 1

    normalized = Counter()
    mapped = 0
    for lemma, frequency in raw_counts.items():
        if lemma in lexical:
            target = lemma
        elif form_targets[lemma]:
            best_score = max(form_targets[lemma].values())
            choices = sorted(
                candidate for candidate, score in form_targets[lemma].items()
                if score == best_score
            )
            lexical_choices = [candidate for candidate in choices if candidate in lexical]
            target = (lexical_choices or choices)[0]
            mapped += 1
        else:
            target = lemma
        normalized[target] += frequency

    normalized, spelling_merges = unify_spelling_variants(normalized)

    # Separable/inseparable derivatives inherit the base verb's class. Repeat
    # because a compound can contain more than one productive prefix.
    changed = True
    while changed:
        changed = False
        for lemma in lexical:
            if lemma in classes:
                continue
            candidates = [
                classes[lemma[len(prefix):]]
                for prefix in PREFIXES
                if lemma.startswith(prefix) and lemma[len(prefix):] in classes
            ]
            if candidates:
                classes[lemma] = candidates[0]
                changed = True

    print(f"Mapped {mapped:,} corpus forms to dictionary lemmas", flush=True)
    print(f"Unified {spelling_merges:,} ss/ß spelling variants", flush=True)
    print(f"Resolved classes for {len(classes):,} lemmas", flush=True)
    return normalized, classes


def is_strong(lemma):
    """Return whether a base or transparently prefixed derivative is strong."""
    if lemma in STRONG_BASES:
        return True
    return any(
        lemma.startswith(prefix)
        and len(lemma) > len(prefix)
        and is_strong(lemma[len(prefix):])
        for prefix in PREFIXES
    )


def strong_stem(lemma):
    """Return the core strong stem underlying this lemma."""
    if lemma in STRONG_STEM_OVERRIDES:
        return STRONG_STEM_OVERRIDES[lemma]
    matches = [base for base in STRONG_BASES if lemma == base or lemma.endswith(base)]
    return max(matches, key=len) if matches else None


def weak_stem(lemma, classes):
    """Return the simplest same-class verb underlying a weak derivative."""
    candidates = [
        lemma[len(prefix):]
        for prefix in PREFIXES
        if lemma.startswith(prefix)
        and len(lemma) > len(prefix)
        and classes.get(lemma[len(prefix):]) in {"weak", "mixed"}
    ]
    if not candidates:
        return lemma
    # Continue through stacked prefixes, e.g. wieder+be+leben -> leben.
    stems = [weak_stem(candidate, classes) for candidate in candidates]
    return min(stems, key=lambda stem: (len(stem), stem))


def weak_paradigm(lemma, stem):
    """Assign a weak verb to its visible conjugation/participle paradigm."""
    if lemma.endswith("ieren"):
        return "ieren"
    if any(lemma.startswith(prefix) for prefix in INSEPARABLE_PREFIXES):
        return "inseparable"
    if lemma != stem:
        return "separable"
    root = lemma[:-2] if lemma.endswith("en") else lemma
    if root.endswith(("d", "t")):
        return "inserted-e"
    return "regular"


def conjugation_class(lemma, classes):
    if any(lemma == base or lemma.endswith(base) for base in IRREGULAR_BASES):
        return "irregular"
    if lemma in CLASS_OVERRIDES:
        return CLASS_OVERRIDES[lemma]
    # Duden treats mixed verbs separately. In this binary UI they remain weak
    # because they take the weak dental endings. Direct dictionary metadata
    # takes precedence for compounds (e.g. beantragen is weak despite tragen).
    if lemma in STRONG_BASES:
        return "strong"
    if lemma in classes:
        return "strong" if classes[lemma] == "strong" else "weak"
    if is_strong(lemma):
        return "strong"
    if lemma.endswith(("en", "eln", "ern")):
        return "weak"
    return "unknown"


def verb_record(
    lemma, frequency, verb_class, classes, principal_parts,
    rank=None, raw_frequency=None
):
    """Build one canonical verb record shared by the UI and validators."""
    if verb_class == "strong":
        stem = strong_stem(lemma)
    elif verb_class == "irregular":
        matches = [base for base in IRREGULAR_BASES if lemma == base or lemma.endswith(base)]
        stem = max(matches, key=len)
    elif verb_class == "weak":
        stem = weak_stem(lemma, classes)
    else:
        stem = None
    record = {
        "lemma": lemma,
        "class": verb_class,
        "kaikkiClass": classes.get(lemma),
        "paradigm": (
            verb_class if verb_class in {"strong", "irregular", "unknown"}
            else weak_paradigm(lemma, stem)
        ),
        "stem": stem,
        "principalParts": (
            principal_parts.get(lemma)
            if verb_class in {"strong", "irregular"} or classes.get(lemma) == "mixed"
            else None
        ),
        "dudenCore": lemma in STRONG_BASES,
        "frequency": frequency,
    }
    if rank is not None:
        record["rank"] = rank
    if raw_frequency is not None:
        record["rawFrequency"] = raw_frequency
    return record


def export(raw_counts, counts, classes):
    principal_parts = build_principal_parts()
    candidates = (
        (lemma, frequency, conjugation_class(lemma, classes))
        for lemma, frequency in counts.items()
        if lemma not in EXCLUDED_LEMMAS and lemma not in EXCLUDED_FORMS
    )
    # Unresolved items in the corpus tail are overwhelmingly tagging errors,
    # truncated words, non-verbs, or unnormalized participles. Exclude them
    # instead of presenting "unknown" as a meaningful conjugation class.
    ranked = sorted(
        (item for item in candidates if item[2] != "unknown"),
        key=lambda item: (-item[1], item[0].casefold(), item[0]),
    )
    ranks = {lemma: rank for rank, (lemma, _, _) in enumerate(ranked, 1)}
    unified = [
        verb_record(
            lemma,
            counts.get(lemma, 0),
            conjugation_class(lemma, classes),
            classes,
            principal_parts,
            ranks.get(lemma),
            raw_counts.get(lemma),
        )
        for lemma in sorted(
            set(raw_counts) | set(counts) | set(classes),
            key=lambda value: (
                ranks.get(value) is None,
                ranks.get(value, 0),
                value.casefold(),
                value,
            ),
        )
    ]
    UNIFIED.write_text(
        json.dumps({
            "schemaVersion": 1,
            "description": "Unified German verb inventory, corpus metadata, and principal parts",
            "principalPartsSource": "Kaikki German verb export (Wiktextract)",
            "verbs": unified,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(unified):,} entries to {UNIFIED.name}", flush=True)

    print(f"Ranked normalized verb lemmas available: {len(ranked):,}", flush=True)


def main():
    raw_counts = build_counts()
    counts, classes = build_metadata(raw_counts)
    export(raw_counts, counts, classes)


if __name__ == "__main__":
    main()
