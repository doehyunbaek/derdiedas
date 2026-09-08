# Der · Die · Das

A static German vocabulary reference for exploring frequency-ranked nouns and verbs, with Python scripts for building and checking the underlying datasets.

## Run locally

From this directory:

```sh
uv run python -m http.server 8000
```

Open <http://localhost:8000>. The app uses plain HTML, CSS, and JavaScript—no frontend build or backend is required. Serve it over HTTP rather than opening `index.html` directly, since it fetches local JSON files.

## What's in the app

- **Nouns:** grouped by grammatical gender (`der`, `die`, `das`) or plural ending, with corpus ranks and frequencies.
- **Verbs:** grouped into weak, strong, and irregular classes, with stem families, conjugation patterns, and principal parts where available.
- **Auxiliary and modal verbs:** a separate reference section for `sein`, `haben`, `werden`, `können`, `müssen`, `wollen`, `sollen`, `dürfen`, and `mögen`.
- **Prepositions:** 21 A1 reference cards styled like Hilfsverben, based on the preposition summary in *Spektrum Deutsch A1+*, printed pp. 268–269 (PDF pp. 270–271). Each card explains selected uses and cases with original German examples, English translations, and relevant contractions. Cards are sorted by Leipzig preposition frequency (including contractions, across all meanings). Search by word, case, or usage.
- **Filters:** top-ranked vocabulary limits, cumulative Goethe A1/A2/B1 vocabulary, and regular-expression search; nouns also have ending-heuristic shortcuts.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Complete browser UI, styles, and client-side logic. |
| `data/german-nouns.json` | Ranked nouns with articles, genders, frequencies, and available plural forms/classifications. |
| `data/german-verbs.json` | Unified verb inventory with classes, paradigms, stems, corpus metadata, and principal parts. Includes unranked records beyond the displayed vocabulary. |
| `data/prepositions-a1.json` | Manually curated Spektrum A1 usage cards loaded by the UI; independent of corpus ranks and Goethe word-list labels. |
| `data/german-prepositions.json` | Retained research dataset: POS-filtered preposition frequencies, contraction breakdowns, reviewed case groups, and excluded non-preposition uses. Also supplies the frequency ordering for the A1 reference view. |
| `build_datasets.py` | All Leipzig/Kaikki downloads, caching, shared spaCy counting, noun genders/plurals, verb metadata, preposition rules, and exports. |
| `test.py` | Offline regression suite, conjugation rules, cached Wiktionary weak/strong validation helpers, and opt-in weak-conjugation runner. |
| `data/goethe-cefr-levels.json` | Earliest recognized Goethe level for noun and verb headwords, plus headwords missing from the top-10,000 datasets. |
| `build_goethe_cefr.py` | Downloads Goethe word-list PDFs and extracts A1–B1 noun/verb headwords. |

## Data sources and maintenance

All application JSON files live in `data/`. The generated files are included; rebuilding them is **not necessary to run the app**. Builders read and write this directory; external source/tagging caches remain in their cache directories.

- **Frequency:** Leipzig's `deu_mixed-typical_2011_1M` sentence corpus, tagged and lemmatized with spaCy's `de_core_news_sm` model.
- **Lexical metadata:** [Kaikki's German dictionary](https://kaikki.org/dictionary/German/) (Wiktionary/Wiktextract).
- **CEFR vocabulary:** Goethe-Institut A1, A2, and B1 *Wortlisten*, linked in `build_goethe_cefr.py`.
- **Conjugation checks:** [German Wiktionary](https://de.wiktionary.org/), with responses cached in `.cache/wiktionary/`.

Source files are downloaded on demand and reused in `~/.cache/derdiedas`. To download both in advance:

```sh
uv run python build_datasets.py download
```

The cache keeps the Leipzig archive and extracted sentence file, plus the compressed Kaikki dictionary (read directly without unpacking). Corpus tagging additionally requires spaCy and its German model.

The noun builder produces `data/german-nouns-top-{100,1000,10000}.json`, not the published filename. To publish a fresh top-10,000 list, copy `data/german-nouns-top-10000.json` to `data/german-nouns.json`, then run `uv run python build_datasets.py plurals`. This replaces the existing noun dataset.

The verb builder can reuse raw counts and class metadata from the existing unified dataset. Principal parts are extracted from `.cache/kaikki-german-verbs.jsonl.gz` when available; otherwise existing principal parts are preserved.

Useful maintenance commands:

```sh
# Apply reviewed corrections without downloading source data.
uv run python build_datasets.py nouns --refresh-genders
uv run python build_datasets.py plurals --refresh-plurals

# Re-extract plurals from a local dictionary (omit the path to reuse the shared cache).
uv run python build_datasets.py plurals --dictionary /path/to/kaikki.org-dictionary-German.jsonl.gz

# Refresh verb metadata using available datasets/source files.
uv run python build_datasets.py verbs

# Rebuild Goethe levels; requires internet access and pdftotext (Poppler).
uv run python build_goethe_cefr.py

# Validate A1 weak verbs against Wiktionary.
uv run python test.py weak-conjugations --level A1 --output /tmp/weak-conjugations.json
```

`uv run python test.py` runs only offline tests. The opt-in `weak-conjugations` command supports `--level A1|A2|B1`, `--refresh`, request delays, and retries. Unlike the UI's cumulative filters, it selects only verbs whose **earliest** Goethe level matches the requested level. Reports distinguish passes, mismatches, and unavailable sources.

## Unified dataset builder

```sh
uv run python build_datasets.py              # export all three datasets
uv run python build_datasets.py nouns        # export one dataset
uv run python build_datasets.py verbs
uv run python build_datasets.py prepositions
```

Counts are reused from `~/.cache/derdiedas/german-corpus-counts-v1.json`. Before that cache exists, the CLI can reuse legacy noun counts, verb `rawFrequency` records, and the existing ADP cache. If any requested inventory is missing, one spaCy pass collects **all three** inventories and saves the shared cache; only requested datasets are exported. `--recount` bypasses existing counts and repeats that shared pass. It does not invalidate lexical metadata caches or change the curated A1 cards.

For tagging, supply spaCy and the German model as below (use `all` instead of `prepositions` to export everything).

### Preposition corpus data

This optional analysis pipeline does not regenerate the manually curated `prepositions-a1.json` reference cards. The source PDF is private and is not distributed with the app.

```sh
uv run --with 'spacy>=3.8,<3.9' \
  --with 'https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl' \
  python build_datasets.py prepositions
uv run python test.py
```

The builder uses `de_core_news_sm` with parser and NER disabled, matching the noun/verb tagging methodology. It counts alphabetic `ADP` tokens, explicitly excludes `PTKVZ` separable verb particles, and normalizes contractions such as `im → in` and `beim → bei`. Excluded uses of reviewed prepositions/contractions are recorded by POS and fine-grained tag. Add `--recount` to replace cached counts after tagging or normalization changes.

Ranks are within the preposition inventory, not all corpus words. Case groups are manually curated common standard government, not automatically inferred case distributions. Notes flag selected alternative uses; unclassified ADP lemmas remain in the research dataset for review and may include tagging errors. Multiword prepositions are not reconstructed, and no CEFR labels are inferred.

## Interpretation notes

Frequency ranks reflect this corpus, not a universal learning order. Gender and plural overrides favor selected common learner meanings over rare homographs or regional variants. Multiple retained genders or plurals can represent different meanings; nouns without a usual plural in the selected sense are omitted from the plural view. CEFR labels are extracted word-list membership, not a complete proficiency classification.
