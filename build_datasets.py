"""Leipzig/Kaikki downloads, caches, tagging, and German dataset exports.

Run without arguments to export all datasets; use --help for individual tasks.
Curated A1 cards and Goethe extraction are independent of this pipeline.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import gzip
import json
import re
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# --- Sources and downloads ---

CACHE = Path.home() / ".cache" / "derdiedas"
CORPUS_NAME = "deu_mixed-typical_2011_1M"
CORPUS = CACHE / CORPUS_NAME / f"{CORPUS_NAME}-sentences.txt"
KAIKKI = CACHE / "kaikki.org-dictionary-German.jsonl.gz"
CORPUS_URL = f"https://downloads.wortschatz-leipzig.de/corpora/{CORPUS_NAME}.tar.gz"
KAIKKI_URL = "https://kaikki.org/dictionary/German/kaikki.org-dictionary-German.jsonl.gz"


def download(url, destination):
    """Publish only complete downloads; reuse existing files."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        print(f"Downloading {url} to {destination}", flush=True)
        urllib.request.urlretrieve(url, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def ensure_corpus():
    if CORPUS.exists():
        return CORPUS
    archive = download(CORPUS_URL, CACHE / f"{CORPUS_NAME}.tar.gz")
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    # Copy only the required member, without extracting arbitrary archive paths.
    with tarfile.open(archive, "r:gz") as source:
        member = next(
            item for item in source
            if item.isfile() and Path(item.name).name == CORPUS.name
        )
        with tempfile.NamedTemporaryFile(dir=CORPUS.parent, delete=False) as output:
            temporary_path = Path(output.name)
            try:
                with source.extractfile(member) as sentences:
                    shutil.copyfileobj(sentences, output)
                output.close()
                temporary_path.replace(CORPUS)
            finally:
                temporary_path.unlink(missing_ok=True)
    return CORPUS


def open_dictionary(path=None):
    """Open an explicit dictionary or download/reuse the shared gzip cache."""
    path = Path(path) if path is not None else download(KAIKKI_URL, KAIKKI)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")

# --- Noun genders and export ---

NOUN_COUNTS = DATA / "german-noun-lemma-counts.json"
GENDERS = DATA / "kaikki-german-noun-genders.json"
GENDER_ORDER = ("masculine", "feminine", "neuter")
ARTICLES = {"masculine": "der", "feminine": "die", "neuter": "das"}
# Prefer selected common standard senses over rare homographs and regional variants.
GENDER_OVERRIDES = {
    # Common learner meanings; do not merge rare homographs into these entries.
    "Alter": {"neuter"},  # age
    "Automat": {"masculine"},
    "Dame": {"feminine"},
    "Fax": {"neuter"},
    "Laden": {"masculine"},  # shop, not nominalized laden
    "Mensch": {"masculine"},  # neuter is regional/derogatory
    "Mittag": {"masculine"},
    "Morgen": {"masculine"},  # morning, not the nominalized adverb
    "Mund": {"masculine"},
    "Ort": {"masculine"},  # place
    "Post": {"feminine"},  # mail, not a social-media post
    "Prospekt": {"masculine"},
    "Reis": {"masculine"},  # rice, not a twig (das Reis)
    "Salat": {"masculine"},
    "Taxi": {"neuter"},
    "Tee": {"masculine"},  # tea, not a golf tee
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
    with open_dictionary() as source:
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


def export_nouns(counts, genders):
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
        output = DATA / f"german-nouns-top-{size}.json"
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(data):,} entries to {output.name}", flush=True)

    print(f"Gender-annotated ranked lemmas available: {len(ranked):,}", flush=True)


def refresh_gender_overrides(path):
    """Correct the published dataset without rebuilding counts or losing plurals."""
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for noun in data:
        override = GENDER_OVERRIDES.get(noun["lemma"])
        if override is None:
            continue
        values = [gender for gender in GENDER_ORDER if gender in override]
        articles = [ARTICLES[value] for value in values]
        genders = [value[0] for value in values]
        if noun["articles"] != articles or noun["genders"] != genders:
            noun.update(articles=articles, genders=genders)
            changed += 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed

# --- Noun plurals ---

NOUNS = ROOT / "data" / "german-nouns.json"


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


def enrich_noun_plurals(dictionary=None, refresh=False):

    nouns = json.loads(NOUNS.read_text(encoding="utf-8"))
    if refresh:
        apply_plural_overrides(nouns)
        NOUNS.write_text(json.dumps(nouns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Applied reviewed plural corrections")
        return
    wanted = {noun["lemma"] for noun in nouns}
    plurals = defaultdict(list)
    with open_dictionary(dictionary) as source:
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

# --- Verb metadata and export ---

UNIFIED = ROOT / "data" / "german-verbs.json"
KAIKKI_VERBS = ROOT / ".cache" / "kaikki-german-verbs.jsonl.gz"

# These verbs have their own curated Hilfsverben tab.
EXCLUDED_LEMMAS = {
    "sein", "haben", "werden", "können", "müssen", "wollen", "sollen",
    "dürfen", "mögen",
    # Compounds treated here as constructions with the excluded auxiliary.
    "loswerden", "fertigwerden",
    # Covered by nehmen/halten rather than standalone verb entries.
    "gefangennehmen", "gefangengehalen",
}
# Frequent tagging/spelling artifacts which unambiguously belong to an
# excluded auxiliary lemma but are not represented as forms in Wiktionary.
EXCLUDED_FORMS = {
    "mussen", "mussn", "muessen", "mußt", "müs", "kannstn",
}
# Unambiguous corpus/tagger artifacts. Correct misspellings of real verbs are
# folded into the canonical lemma; adjective-derived pseudo-verbs are dropped.
LEMMA_CORRECTIONS = {
    "emfehlen": "empfehlen",
    "emnpfehlen": "empfehlen",
    "vermuteen": "vermuten",
}
NON_VERB_LEMMAS = {
    "emfehlensweren", "empfehlensweren", "empfehlenswern",
    "umfängen", "verfängen",
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
    "empfangen": "fangen",
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


def unified_records():
    if not UNIFIED.exists():
        return []
    return json.loads(UNIFIED.read_text(encoding="utf-8"))["verbs"]


def normalize_corpus_lemma(lemma):
    """Canonicalize known tagger misspellings and reject pseudo-verbs."""
    if lemma in NON_VERB_LEMMAS:
        return None
    return LEMMA_CORRECTIONS.get(lemma, lemma)


def kaikki_lines():
    """Read the shared cached Kaikki dictionary."""
    return open_dictionary()


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
        normalized = Counter()
        for record in records:
            lemma = normalize_corpus_lemma(record["lemma"])
            if lemma and record.get("frequency"):
                normalized[lemma] += record["frequency"]
        classes = {
            record["lemma"]: record["kaikkiClass"]
            for record in records
            if record.get("kaikkiClass")
            and normalize_corpus_lemma(record["lemma"]) == record["lemma"]
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
    if lemma in NON_VERB_LEMMAS:
        return "unknown"
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
            # Fragen uses regular standard forms; regional frägt/frug should
            # not trigger the special principal-parts display.
            if verb_class in {"strong", "irregular"} or (classes.get(lemma) == "mixed" and lemma != "fragen")
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


def export_verbs(raw_counts, counts, classes):
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
            (set(raw_counts) | set(counts) | set(classes))
            - {"gefangennehmen", "gefangengehalen", "umfängen", "verfängen"},
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

# --- Preposition normalization and export ---

# Explicit surface normalization, applied only to tokens tagged ADP.
CONTRACTIONS = {
    "am": "an", "ans": "an", "aufs": "auf", "beim": "bei",
    "durchs": "durch", "fürs": "für", "hinterm": "hinter", "hinters": "hinter",
    "im": "in", "ins": "in", "überm": "über", "übers": "über",
    "ums": "um", "unterm": "unter", "unters": "unter", "vom": "von",
    "vorm": "vor", "vors": "vor", "zum": "zu", "zur": "zu",
}
# Common standard government, not an exhaustive inventory of regional/rare uses.
GROUPS = {
    "accusative": "bis durch für gegen ohne um wider",
    "dative": "aus außer bei entgegen entsprechend gegenüber gemäß mit nach samt seit von zu zuliebe",
    "genitive": "angesichts anhand anlässlich anstatt außerhalb beiderseits bezüglich diesseits einschließlich entlang innerhalb jenseits kraft längs mittels oberhalb seitens statt trotz unterhalb unweit während wegen zugunsten zwecks",
    "two-way": "an auf hinter in neben über unter vor zwischen",
}
NOTES = {
    "bis": "Usually accusative; often followed by another preposition, which governs its own case (bis zu + dative).",
    "entlang": "Genitive before the noun; accusative when placed after it. Dative also occurs.",
    "trotz": "Usually genitive; dative also occurs.",
    "wegen": "Standard genitive; dative is common colloquially.",
    "während": "Genitive as a preposition; also a conjunction (excluded here).",
}
CASE_BY_LEMMA = {word: group for group, words in GROUPS.items() for word in words.split()}


def is_preposition(token):
    # Some model mappings label separable particles ADP; STTS is decisive here.
    return token.pos_ == "ADP" and token.tag_ != "PTKVZ"


def normalize_preposition(token):
    surface = token.text.lower()
    return CONTRACTIONS.get(surface, token.lemma_.strip().lower())


def export_prepositions(data):
    records = []
    for rank, (lemma, frequency) in enumerate(sorted(data["counts"].items(), key=lambda item: (-item[1], item[0])), 1):
        records.append({
            "rank": rank, "lemma": lemma, "frequency": frequency,
            "perMillionTokens": round(frequency / data["alphabeticTokens"] * 1_000_000, 2),
            "caseGroup": CASE_BY_LEMMA.get(lemma, "unclassified"),
            "note": NOTES.get(lemma, ""),
            "surfaceForms": data["surfaceForms"][lemma],
        })
    output = DATA / "german-prepositions.json"
    output.write_text(json.dumps({
        "schemaVersion": 1,
        "source": "Leipzig deu_mixed-typical_2011_1M",
        "method": "Alphabetic ADP tokens excluding PTKVZ verb particles; lowercase spaCy lemmas with explicit contraction overrides. Ranks are within prepositions, not all words.",
        "caseSource": "Manually curated common standard government; not inferred from corpus. Unclassified entries may include tagging errors.",
        **{key: data[key] for key in ("model", "spacyVersion", "sentences", "alphabeticTokens", "excludedUses")},
        "contractions": CONTRACTIONS,
        "prepositions": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} entries to {output}", flush=True)

# --- Shared corpus tagging and cache ---

COUNTS = CACHE / "german-corpus-counts-v1.json"
SCHEMA_VERSION = 1


def sentence_texts():
    with ensure_corpus().open(encoding="utf-8") as source:
        for line in source:
            _, separator, text = line.rstrip("\n").partition("\t")
            if separator:
                yield text


def count_documents(documents, model, spacy_version):
    """Collect all three inventories from the same stream of tagged documents."""

    nouns, verbs, prepositions = Counter(), Counter(), Counter()
    surfaces, excluded = defaultdict(Counter), defaultdict(Counter)
    tracked = set(CASE_BY_LEMMA) | set(CONTRACTIONS)
    sentences = tokens = 0
    for doc in documents:
        sentences += 1
        for token in doc:
            if not token.is_alpha:
                continue
            tokens += 1
            lemma = token.lemma_.strip()
            if token.pos_ == "NOUN" and lemma and lemma.isalpha():
                nouns[lemma[0].upper() + lemma[1:]] += 1
            if token.pos_ in {"VERB", "AUX"}:
                verb = normalize_corpus_lemma(lemma.lower())
                if verb and verb.isalpha():
                    verbs[verb] += 1
            surface = token.text.lower()
            if is_preposition(token):
                preposition = normalize_preposition(token)
                if preposition and preposition.isalpha():
                    prepositions[preposition] += 1
                    surfaces[preposition][surface] += 1
            elif surface in tracked:
                excluded[surface][f"{token.pos_}/{token.tag_}"] += 1
        if sentences % 100_000 == 0:
            print(f"Tagged {sentences:,} sentences", flush=True)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": "deu_mixed-typical_2011_1M",
        "nouns": dict(nouns), "verbs": dict(verbs),
        "prepositions": {
            "model": model, "spacyVersion": spacy_version,
            "sentences": sentences, "alphabeticTokens": tokens,
            "counts": dict(prepositions), "surfaceForms": dict(surfaces),
            "excludedUses": dict(excluded),
        },
    }


def load_shared_counts():
    if COUNTS.exists():
        data = json.loads(COUNTS.read_text(encoding="utf-8"))
        if data.get("schemaVersion") == SCHEMA_VERSION:
            return data
    return None


def build_counts(recount=False):
    if not recount:
        data = load_shared_counts()
        if data is not None:
            print(f"Using shared counts from {COUNTS}", flush=True)
            return data
    import spacy
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    data = count_documents(
        nlp.pipe(sentence_texts(), batch_size=512),
        f"de_core_news_sm {nlp.meta['version']}", spacy.__version__,
    )
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=COUNTS.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.close()
            temporary.replace(COUNTS)
        finally:
            temporary.unlink(missing_ok=True)
    return data


def legacy_counts(kind):
    """Reuse existing counts when a shared cache has not yet been built."""
    if kind == "nouns" and NOUN_COUNTS.exists():
        return json.loads(NOUN_COUNTS.read_text(encoding="utf-8"))
    if kind == "verbs":
        records = unified_records()
        if any(record.get("rawFrequency") is not None for record in records):
            counts = Counter()
            for record in records:
                lemma = normalize_corpus_lemma(record["lemma"])
                frequency = record.get("rawFrequency")
                if lemma and frequency is not None:
                    counts[lemma] += frequency
            return dict(counts)
    if kind == "prepositions":
        path = CACHE / "german-adp-counts-spacy-3.8.0.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def selected_counts(kinds, recount=False):
    if recount:
        shared = build_counts(recount=True)
        return {kind: shared[kind] for kind in kinds}
    shared = load_shared_counts()
    if shared is not None:
        return {kind: shared[kind] for kind in kinds}
    existing = {kind: legacy_counts(kind) for kind in kinds}
    if any(value is None for value in existing.values()):
        # A missing inventory triggers exactly one pass collecting all kinds.
        shared = build_counts()
        return {kind: shared[kind] for kind in kinds}
    return existing

# --- Command line ---

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=("all", "nouns", "verbs", "prepositions", "plurals", "download"))
    parser.add_argument("--recount", action="store_true", help="Retag all word classes in one pass, bypassing existing counts")
    parser.add_argument("--refresh-genders", action="store_true", help="Only apply reviewed noun gender corrections")
    parser.add_argument("--refresh-plurals", action="store_true", help="Only apply reviewed noun plural corrections")
    parser.add_argument("--dictionary", type=Path, help="Explicit JSONL/JSONL.gz dictionary for plural enrichment")
    args = parser.parse_args(argv)
    if args.refresh_genders and (args.dataset not in {"all", "nouns"} or args.recount or args.refresh_plurals):
        parser.error("--refresh-genders requires nouns (or no dataset) and cannot be combined with other refresh options")
    if args.refresh_plurals and (args.dataset not in {"all", "nouns", "plurals"} or args.recount):
        parser.error("--refresh-plurals requires nouns/plurals (or no dataset) and cannot be combined with --recount")
    if args.dictionary and (args.dataset != "plurals" or args.refresh_plurals):
        parser.error("--dictionary is only for full plural enrichment: plurals --dictionary PATH")
    if args.recount and args.dataset in {"download", "plurals"}:
        parser.error("--recount requires all, nouns, verbs, or prepositions")
    DATA.mkdir(parents=True, exist_ok=True)
    if args.refresh_genders:
        print(f"Corrected genders for {refresh_gender_overrides(NOUNS)} published nouns")
        return
    if args.refresh_plurals or args.dataset == "plurals":
        enrich_noun_plurals(args.dictionary, args.refresh_plurals)
        return
    if args.dataset == "download":
        print(ensure_corpus())
        print(download(KAIKKI_URL, KAIKKI))
        return
    kinds = ("nouns", "verbs", "prepositions") if args.dataset == "all" else (args.dataset,)
    counts = selected_counts(kinds, args.recount)
    for kind in kinds:
        if kind == "nouns":
            export_nouns(Counter(counts[kind]), build_genders(set(counts[kind])))
        elif kind == "verbs":
            raw = Counter(counts[kind])
            normalized, classes = build_metadata(raw)
            export_verbs(raw, normalized, classes)
        else:
            export_prepositions(counts[kind])


if __name__ == "__main__":
    main()
