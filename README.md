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
