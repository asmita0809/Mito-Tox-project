# Data

## What is committed, and what is not

`data/processed/` **is committed.** It is small, and it is what makes the modelling
reproducible without a network round-trip.

`data/external/` **is gitignored.** It holds the raw API responses (one file is 26 MB), and
they are regenerable exactly by the download scripts. The checksums below let you confirm
you fetched the same bytes.

## Tox21 — the training source

Downloaded by `scripts/download_tox21.py` from PubChem BioAssay, 2026-09-01.

The PUG REST data-table endpoint (`/rest/pug/assay/aid/<aid>/CSV`) returns **HTTP 400
PUGREST.BadRequest** for both assays. The endpoint that works is the classic download CGI:

```
https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi
    ?query=download&record_type=datatable&actvty=all&response_type=save&aid=<aid>
```

The script fetches both and saves both, so the failure is visible in the manifest rather than
silently producing a truncated table. `download_manifest.json` reports `status: incomplete`
for exactly this reason — the two PUG REST data-table views failed; the eight files below all
downloaded and verified.

| file | bytes | sha256 (first 16) |
|---|---|---|
| `aid720637_description.json` | 8,285 | `d11fe01be1fbb4ba` |
| `aid720637_summary.json` | 3,871 | `d01d3337537f5b75` |
| `aid720637_concise.csv` | 1,712,129 | `9d9f011f3588eb69` |
| `aid720637_datatable_pcget.csv` | 1,732,929 | `ae2529f9ed5f6699` |
| `aid720634_description.json` | 686,530 | `60bd41c2ba11a189` |
| `aid720634_summary.json` | 4,776 | `ef31dc25b797acd8` |
| `aid720634_concise.csv` | 1,837,435 | `2225281582628ea0` |
| `aid720634_datatable_pcget.csv` | 25,941,174 | `6e427d0a387a3cad` |

AID 720637 contains 10,486 sample records; AID 720634 contains the same 10,486 substances,
joinable one-to-one on `PUBCHEM_SID`.

Tox21 data is a public-domain work of the U.S. government. It is not redistributed here
because it is large and freely available from the source of record.

## MitoTox — provenance analysis only, not a training source

Downloaded by `scripts/download_mitotox.py` from `https://www.mitotox.org/api/`, 2026-09-01.
All five endpoints verified: declared record counts matched downloaded counts exactly.

| file | records | sha256 (first 16) |
|---|---|---|
| `mitotox_compounds.json` | 1,453 | `a54079d694efda51` |
| `mitotox_records.json` | 4,617 | `2da7500370861c06` |
| `mitotox_functions.json` | 249 | `96782dd4acb0a9a2` |
| `mitotox_targets.json` | 876 | `828f6ad8449829aa` |
| `mitotox_papers.json` | 334 | `c8ff75c00f42d5d1` |

This was downloaded to be a second, independent training source. The provenance analysis
(`scripts/inspect_mitotox_provenance.py`) showed it is not independent: 1,861 of the 1,997
membrane-potential records come from one paper, the Tox21 10K follow-up. It is retained in
the repository because the analysis that ruled it out is part of the result.

MitoTox is an academic database; see mitotox.org for its terms. Nothing from it is
redistributed here.

## Processed files

| file | rows | what it is |
|---|---|---|
| `tox21_mmp_observations.csv` | 10,486 | every raw sample record, nothing dropped |
| `tox21_mmp_compounds.csv` | 8,099 | compound level, all structure diagnostics |
| `tox21_mmp_training.csv` | 5,938 | the modelling set, labelled |
| `tox21_mmp_excluded.csv` | 1,637 | excluded parents, flagged, **no label column** |
| `splits.csv` | 5,938 | subset and CV fold per compound |
| `features_morgan.npz` | 5,938 × 2048 | binary Morgan fingerprints |
| `features_maccs.npz` | 5,938 × 167 | MACCS keys |
| `features_descriptors.csv` | 5,938 × 6 | the six descriptors |
| `structural_alerts.csv` | 5,938 × 13 | SMARTS alert matches (**not** model features) |
| `test_nn_similarity.csv` | 1,205 | each test compound's nearest training neighbour |
| `oof_predictions_dev.npz` | 4,733 × 108 | out-of-fold predictions for every configuration |
| `feature_metadata.json` | — | every fingerprint and descriptor setting |
