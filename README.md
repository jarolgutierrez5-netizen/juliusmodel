# HR Signal Tracker

A GitHub Pages dashboard for daily MLB home-run projections and outcome tracking.

## Automated schedule

The GitHub Actions workflow runs twice each day during the MLB season:

- `13:30 UTC` builds the daily projection file.
- `11:00 UTC` settles the prior day's results.

GitHub's scheduler is best-effort. You can also run either workflow manually from the **Actions** tab.

## Deploy

1. Create an empty GitHub repository.
2. Upload this project or push it with Git.
3. In **Settings → Pages**, select **Deploy from a branch**, then choose `main` and `/ (root)`.
4. In **Actions**, allow workflows to have read and write permissions so the scheduled job can commit generated data.

## Data model

Daily files live in `data/daily/YYYY-MM-DD.json`. The app reads every daily file listed in `data/manifest.json` and shows a cumulative calibration summary.

The default scripts use the public MLB Stats API for schedule, lineups, and final box scores. The projection function is deliberately conservative until you add a Statcast/contact-quality data source and any desired weather or park inputs.


## Accuracy modules included

The daily model now integrates:

- `playing_time_model.py` for expected PA, confirmed-lineup treatment, and substitution risk.
- `bullpen_model.py` for bullpen HR/barrel context and workload readiness once a bullpen feed is connected.
- `environment_model.py` for park, roof, temperature, and wind adjustments with neutral fallbacks.
- `minor_league_prior.py` for partially pooled priors on recent call-ups and limited-PA hitters.
- `pitch_shape_matchups.py` for hitter power against the starter's pitch mix.
- `calibration.py` for Brier score and probability-bucket calibration.
- `miss_review.py` for preliminary postgame miss labels.

The current public-data pipeline uses confirmed MLB lineups, season stats, and Statcast. Some accuracy modules intentionally run at neutral weight until a verified same-day weather, roof, bullpen, and minor-league data feed is connected. Every neutral fallback is written to `missing_inputs` rather than hidden.


## Number of displayed players

The model now defaults to **15 daily players** instead of five. It aims for roughly 60% established profiles and 40% under-the-radar profiles when the data supports both groups.

To change the count in GitHub Actions, set an environment variable for the build step:

```yaml
env:
  MAX_CALLS: "20"
```


## Trained-model workflow

The dashboard now distinguishes a **transparent prototype** from a trained model. Daily calls use the calibrated model only when `models/hr_pa_calibrated.joblib` exists. Otherwise the site labels the run as `transparent-prototype` and preserves the interpretable feature-based fallback.

To train, first build a historical plate-appearance dataset containing the columns documented at the top of `scripts/train_hr_model.py`. Then run:

```bash
python scripts/train_hr_model.py --input data/training/pa_features.parquet
python scripts/backtest_hr_model.py \
  --input data/training/pa_features.parquet \
  --model models/hr_pa_calibrated.joblib \
  --output data/backtest_report.json
```

The training script uses a **chronological** split and Platt calibration on later data, rather than random splitting. Do not publish calibrated probabilities until `data/training_report.json` and the independent holdout report show acceptable Brier score and probability-bucket calibration.
