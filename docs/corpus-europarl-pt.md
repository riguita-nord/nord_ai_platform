# Europarl Portuguese source

Downloaded archive: `data/corpus/europarl-pt.xml.bz2` (not committed; `data/` is gitignored).

Source: Språkbanken, *Europarl: Swedish-Portuguese*, release v3, Portuguese XML download: <https://spraakbanken.gu.se/resurser/meningsmangder/europarl-pt.xml.bz2>.

Språkbanken identifies the Portuguese resource as European Portuguese and lists it under CC BY 4.0. The downloadable Portuguese XML archive is about 90.33 MB. The Portulan CLARIN corpus documentation describes approximately 58.3 million European Portuguese tokens, from European Parliament sessions (1996–2011); its domain is formal and political, not everyday conversation.

Attribution: Koehn, Philipp (2014). *Europarl: Swedish-Portuguese* (release v3, updated 2014-04-29). Dataset enriched and distributed by Språkbanken. DOI: <https://doi.org/10.23695/7aja-mv27>. The original Europarl corpus reference is Koehn (2005), *Europarl: A Parallel Corpus for Statistical Machine Translation*, MT Summit X, pp. 79–86.

The current preparation command extracts the first 500,000 sentence records as UTF-8 text:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_europarl_corpus.py --max-sentences 500000
```

The derived file remains local at `data/corpus/europarl-pt.txt`. Retain attribution and the CC BY 4.0 notice with any redistributed derivative. Do not treat this parliamentary corpus as a representative conversational corpus.
