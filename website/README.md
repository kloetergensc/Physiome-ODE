# Leaderboard website

The site at https://kloetergensc.github.io/Physiome-ODE/ is a static page served by GitHub Pages from `docs/`.
Nothing runs on a server: the page loads JSON files and computes every statistic (Elo, ranks, win rates) in the browser.

```
website/                     sources, edited by hand
  models.json                model metadata; list order fixes each model's color
  datasets.csv               short ids (DUP01, ...) for the 50 datasets
  results/<model>.csv        one results file per model
  build_leaderboard.py       -> docs/data/results.json (+ README table)
  generate_samples.py        -> docs/samples/<id>.json
docs/                        what GitHub Pages serves; only the scripts write data/ and samples/
  index.html, assets/        page, styles, script, self-hosted fonts
  data/results.json
  samples/<id>.json
```

## Add or update a model

1. Put the results in `website/results/<model>.csv`, one row per dataset and fold:

   ```
   dataset,fold,test_mse
   dupont_1991a,0,0.948
   dupont_1991a,1,0.957
   ```

   `dataset` is the Physiome model name used by the training scripts (`-dset`). An already aggregated file
   with the columns `dataset,mse_mean,mse_std` also works; the paper models are stored that way because only
   the rounded means and standard deviations from the paper are available. For fold files the script reports
   the mean and the sample standard deviation over folds.

2. Add an entry to `website/models.json` (`name`, `results`, `family`, `reference`, `code`, and `status`:
   `paper`, `verified` or `self-reported`). New models go at the end so existing models keep their colors.
   The palette has 8 slots; a 9th model is drawn in a neutral gray, so decide on a scheme before then.

3. Rebuild and check:

   ```
   python website/build_leaderboard.py --readme
   ```

   A model that is missing any of the 50 datasets is left out, with a warning. The script uses only the
   standard library.

4. Commit `website/`, `docs/data/results.json` and `README.md`. GitHub Pages redeploys within a minute or two.

## Dataset samples

`generate_samples.py` regenerates the first 12 instances of every dataset with the benchmark's seeds and
solver settings (needs the training environment: numpy, scipy). Rerun it only if the datasets change:

```
python website/generate_samples.py              # all datasets, ~15 s on 32 cores
python website/generate_samples.py --only LEN01
```

## Preview locally

The page fetches its data, so open it through a web server rather than as a file:

```
python -m http.server -d docs 8000             # then visit http://localhost:8000
```

## Publishing

Repository Settings → Pages → Build and deployment: "Deploy from a branch", branch `main`, folder `/docs`.
For a custom domain, add it under the same settings page (GitHub writes `docs/CNAME`) and point a DNS record
at GitHub Pages.
