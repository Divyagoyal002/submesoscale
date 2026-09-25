# IRIS National Fair 2026–27: project notes

## How each requirement is covered

| ID | Requirement | Where it is done | How it is checked |
|---|---|---|---|
| FR-1 | Ingest ADT and SST | `data/copernicus.py`, `submeso download` | Dataset IDs checked against the catalogue; `subset()` call matches its signature |
| FR-2 | OSSE training data | `data/osse.py`, `data/synthetic.py` | `tests/test_pipeline_smoke.py` |
| FR-3 | CNN super-resolution | `models/networks.py` | `tests/test_model_and_loss.py` |
| FR-4 | Physics-informed loss | `models/losses.py` | Tests check that the loss is 0 for a perfect prediction, ignores land, and ignores a constant offset |
| FR-5 | Geostrophic currents | `physics.py`, `inference.py` | An analytic test against a known wave; a warm eddy gives anticyclonic vorticity |
| FR-6 | Drifter validation | `validation/drifters.py` | Tested on synthetic drifters; the GDP download was tested on real data |
| FR-7 | Spectral analysis | `validation/spectra.py` | Recovers a known k⁻¹¹ᐟ³ slope in the tests |
| FR-8 | Visualisation | `viz/plots.py`, `submeso visualize` | Maps, spectra, scatter plots and an MP4/GIF animation |

| Non-functional requirement | How it is met |
|---|---|
| Reproducibility | Everything is scripted through the CLI; seeds are fixed; each run saves its config; CI runs the tests |
| Compute | One region, patches of 64 px, about 1–2 h on a single GPU; a Colab notebook is provided |
| Data provenance | Open data only; see `docs/data_sources.md` |
| Interpretability | Quantitative metrics (RMSE, effective resolution, band-energy ratio, drifter statistics), not visual sharpness alone |

## Originality (what makes this more than a reproduction)

1. **A different region and period.** The Alboran–Balearic sub-basin in 2022–2024 overlaps the SWOT
   era. The reference study covered the whole Mediterranean from 2008 to 2019. A Black Sea configuration
   is also included.
2. **A new loss.** Two terms are added to the geostrophic-velocity term: a Rossby-number (ζ/f) term and
   an isotropic log-spectrum term.
3. **New diagnostics.** A 10–50 km band-energy ratio tests directly whether the added detail is real
   energy or an artefact, alongside the effective resolution (spectral score).
4. **A synthetic SQG test bed.** It gives a controlled experiment in which SST is known to carry
   fine-scale information, useful for ablation studies.

## Suggested experiments for the paper

- **Loss ablation:** ADT only, then + geo, then + vort, then + spec. Compare `metrics_osse.json`.
- **Architecture:** `-o model.name=srcnn` compared with `resnet`.
- **Does SST help?** Retrain with the SST channel set to zero, for example by overriding
  `osse.sst_noise_k` to a large value, and compare.
- **Region transfer:** train on the Western Mediterranean and evaluate on the Black Sea, or the reverse.

## Before you submit

- [ ] **Anonymity:** no school, city or state appears in the video, paper, abstract, figures, **or in
      this repository** (commit author names, README, notebook outputs). Check `git log` and the
      GitHub profile shown on the repo before sharing the link.
- [ ] Decide on the category: Systems Software or Earth & Environmental Sciences.
- [ ] Put the numbers from the **real** data (`runs/wmed_sr/metrics_*.json`) in the paper, not the demo numbers.
- [ ] Cite every dataset (see `docs/data_sources.md`) and the reference study.
