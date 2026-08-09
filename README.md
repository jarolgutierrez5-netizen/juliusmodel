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
