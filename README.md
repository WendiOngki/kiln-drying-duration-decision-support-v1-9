# Kiln-Drying Duration Decision Support

Batch-level prediction of wood kiln-drying duration for loading and downstream scheduling, developed with a leakage-controlled temporal evaluation protocol.

**Live prototype:** https://kiln-drying-duration-decision-support.streamlit.app/

> **Status.** This repository accompanies a research manuscript that is currently under review. The model is a *candidate* decision-support model. It has been evaluated retrospectively on historical batches only and has not yet been validated prospectively. Predictions require operator review and do not replace moisture-content inspection.

---

## Problem

In wood-furniture manufacturing, kiln drying sits between sawmilling and all downstream machining. At the case manufacturer (a wood-furniture manufacturer in Klaten Regency, Indonesia), each kiln charge is planned with a fixed 14-day allowance. In the historical records, realized durations ranged from 9 to 43 days, and 149 of 166 batches (89.8%) took longer than 14 days.

This project estimates, **at the moment a kiln is loaded**, how long a charge will take, using information the plant already records: species, kiln, load, board-thickness composition, a target moisture-content (MC) proxy, calendar features and a 14-day weather summary.

## Data

- 755 thickness-level production records, aggregated to **166 kiln batches** (149 mahogany, 17 teak), with entry dates from January 2024 to March 2026.
- Daily weather (relative humidity, precipitation, maximum and minimum temperature) from **NASA POWER**, averaged over the first 14 days after entry.
- The production records are confidential and are **not included** in this repository.

The unit of analysis is one batch, so rows of the same batch never fall on both sides of a train/test split.

## Method

| Step | Description |
|---|---|
| Aggregation | Thickness-level rows → one row per batch; board-weighted mean and SD of thickness, maximum thickness, thickness-class shares |
| Features | 26 inputs: species, kiln, volume, boards, volume per board, thickness composition, entry month, calendar-based rainy-season indicator (Oct–Mar), target-MC proxy (MC at the thickest class) and its interaction with maximum thickness, five 14-day weather features |
| Locked audit | The 34 most recent batches (entries from 30 Oct 2025) are locked away; 4 batches overlapping the audit start are purged; 128 development batches remain |
| Model selection | Purged expanding-window temporal cross-validation (4 valid folds) on the development set |
| Learners | Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, all with fixed (untuned) hyperparameters |
| Selection rule | Lowest mean temporal-CV MAE (declared in advance) → **Extra Trees** |
| Final evaluation | One refit on the 128 development batches, one evaluation on the locked audit, against operational baselines, with 2,000-sample bootstrap intervals |

All preprocessing (imputation, scaling, one-hot encoding) is fitted inside each training set through a single scikit-learn pipeline.

## Results

**Locked out-of-time audit (34 most recent batches)**

| Predictor | MAE (days) | RMSE (days) | R² | MAE reduction by model |
|---|---|---|---|---|
| **Extra Trees** | **3.21** [2.33, 4.20] | 4.23 | 0.146 [−0.282, 0.427] | — |
| Development-period mean | 4.67 | 6.39 | −0.951 | 31.3% |
| Fixed 14-day planning rule | 9.59 | 10.62 | −4.389 | 66.5% |

Brackets: 95% bootstrap percentile intervals.

- **Planning tolerance on the audit:** 55.9% of batches within ±3 days, 85.3% within ±5 days, 94.1% within ±7 days.
- **Temporal CV (development set):** MAE 3.55 ± 3.04 days, R² 0.105.
- **R² is modest** because the duration level shifted: mean duration was 19.13 days in the development period and 23.59 days in the audit period. R² is not an accuracy percentage.
- **Random CV is optimistic:** repeated random 5-fold CV on the same data gave MAE 2.08 days and R² 0.588, which overstates what can be expected on future batches.
- **Weather** lowered temporal-CV MAE by 0.12 days (3.67 → 3.55). The gain was obtained with realized weather and is therefore an upper bound for forecast-based use.
- **Constrained data augmentation** gave no stable improvement across 30 seeds (mean ΔMAE +0.019 days, 95% interval −0.074 to +0.113). The final model is trained on observed batches only.

Permutation importance and all other analyses are associational and should not be read as causal effects.

## Prototype application

The Streamlit app (`app.py`) has three tabs:

1. **Prediction**
   - drying start date and 14-day weather (automatic or manual);
   - batch specification (species, kiln, volume, boards, boards per thickness class, target MC of the thickest boards);
   - planning estimate: model prediction, planning duration rounded up to whole days, planned completion date and difference from the 14-day plan;
   - empirical planning buffer of ±3, ±5 or ±7 days, based on the audit tolerance shares (not a statistical prediction interval);
   - warnings when inputs fall outside the range of the training data;
   - downloadable JSON record with an empty `actual_exit_date` field for later prospective validation.
2. **Model performance:** audit metrics, comparison with baselines and error-tolerance shares.
3. **About & limitations:** intended use, inputs that need care and main limitations.

**Weather sources in the app.** In automatic mode, NASA POWER is used when the 14-day window lies entirely in the past, and Open-Meteo forecasts are used when it lies within the forecast horizon. Windows that span both are rejected, and the user is asked to enter values manually.

## Repository structure

```
kiln-drying-duration-decision-support-v1-9/
├── app.py                                   # Streamlit decision-support prototype
├── requirements.txt
├── models/
│   ├── candidate_target_mc_duration_model.joblib   # Extra Trees pipeline fitted on the 128 development batches
│   ├── model_contract.json                  # feature list and model metadata
│   └── environment_manifest.json            # library versions used to fit the model
└── outputs/
    ├── audit_metrics.csv                    # locked-audit metrics for the model and baselines
    └── audit_bootstrap_samples.csv          # bootstrap samples for the audit intervals
```

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The original dataset is not needed to run the app. Manual weather entry works without any configuration. For automatic weather retrieval, create `.streamlit/secrets.toml` (do not commit it):

```toml
[weather]
latitude = 0.0      # coordinates of the study area
longitude = 0.0
location_label = "study area"
```

**Version note.** The saved model must be loaded with the scikit-learn version used to fit it (1.4.2; see `models/environment_manifest.json`). `requirements.txt` pins this version.

## Reproducing the analysis

The analysis was carried out in a single Python notebook (version 1.9.0): data audit, batch aggregation, temporal cross-validation, locked audit, bootstrap, permutation importance and the augmentation analysis, with seed 42 throughout. The notebook is not included here because its stored outputs contain confidential production records; it is available from the author on reasonable request. Library versions are listed in `models/environment_manifest.json` (Python 3.10, pandas 2.1.4, NumPy 1.26.4, scikit-learn 1.4.2, XGBoost 3.1.2, Matplotlib 3.7.5, seaborn 0.13.2).

## Limitations

- Single plant, 166 batches, only 17 teak batches; no prospective or external validation yet.
- The target-MC input is a proxy: historical records store the MC measured at release, which is used in place of a target set before drying.
- Weather features in development are realized observations; in use, forecasts add uncertainty.
- Entry month and the rainy-season indicator are calendar proxies. The actual onset of the rainy season varies from year to year.
- Initial moisture content, wood density and in-kiln climate were not recorded.
- Recent conditions can fall outside the training range; the app flags such inputs as extrapolation.

## Earlier version

An earlier prototype, developed for an undergraduate thesis, used a random hold-out split with data augmentation and an XGBoost model. That workflow has been superseded by the temporal evaluation described here, and its results are not comparable with the results above.

## Citation

Manuscript under review. Citation details will be added after publication.

## License

[To be added]

## Author

Wendi Ongki Nugroho Saputro
