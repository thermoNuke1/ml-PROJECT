# Google Colab Reproducibility

This project now has an explicit Colab path aimed at the course guideline that the notebook and code should be reproducible by the teaching team.

## What is reproducible in Colab

There are two supported levels:

1. `Quick repro`: runs entirely from tracked repo files and finishes on a normal Colab runtime.
2. `Full repro`: reruns the larger preprocessing and training pipeline from the original Lichess PGN source or from a filtered rapid-only PGN supplied to Colab.

The quick path proves that the core scripts execute cleanly in Colab. The full path is the one to use when regenerating tables that are meant to closely match the report.

## Quick repro

Open the project notebook or a fresh Colab notebook and run:

```python
!git clone https://github.com/thermoNuke1/ml-PROJECT.git
%cd /content/ml-PROJECT
!pip install -q -r requirements.txt
!python scripts/run_colab_repro.py --run-rnn-smoke --rnn-epochs 2
```

This does three things:

1. Prints archived larger-scale metrics that are tracked in `artifacts/reported_results/`.
2. Rebuilds a small board-feature logistic-regression baseline from the tracked mini PGN in `artifacts/sample_data/`.
3. Optionally runs a tiny shard-based PyTorch RNN smoke test to confirm the recurrent pipeline works in Colab too.

The quick repro writes fresh outputs into `artifacts/colab_runs/`.

## Full repro

For larger experiments, use the same repo clone and dependency install, then choose one of the following paths.

### Path A: start from the raw Lichess monthly PGN

If you have the February 2026 Lichess PGN available in Colab or Google Drive:

```python
!python scripts/filter_lichess_pgn.py \
    --input /content/lichess_db_standard_rated_2026-02.pgn \
    --output data/lichess_rapid_10_0_completed.pgn \
    --time-control 600+0 \
    --termination Normal
```

Then run the desired downstream experiment.

Board-logistic 1k repro:

```python
!python scripts/extract_lichess_board_features.py \
    --input data/lichess_rapid_10_0_completed.pgn \
    --output data/dev_board_features_1000_games.csv \
    --max-games 1000

!python scripts/train_baseline_model.py \
    --input data/dev_board_features_1000_games.csv \
    --model logreg \
    --metrics-output data/dev_board_logreg_metrics_1000_games.json
```

Shard-based PyTorch RNN repro on 250k games:

```python
!python scripts/extract_rnn_game_shards_parallel.py \
    --input data/lichess_rapid_10_0_completed.pgn \
    --output-dir data/rnn_shards_250000 \
    --max-games 250000 \
    --workers 2 \
    --chunk-games 200 \
    --games-per-shard 2000

!python scripts/train_simplernn_from_shards_torch.py \
    --input-dir data/rnn_shards_250000 \
    --output data/torch_rnn_landmark_board_250000.json \
    --model-type rnn \
    --device cuda
```

### Path B: start from a pre-filtered rapid-only PGN

If Colab storage is tight, upload or mount only the already-filtered `600+0` and `Termination="Normal"` PGN, then start directly from feature extraction or shard extraction.

## Tracked reproducibility assets

The repo now includes small tracked files that are safe to keep in Git:

- `artifacts/sample_data/colab_dev_60_games.pgn`
- `artifacts/sample_data/sample_rapid_10_0_completed.pgn`
- `artifacts/reported_results/dev_board_logreg_metrics_1000_games.json`
- `artifacts/reported_results/torch_rnn_landmark_board_250000_with_history.json`
- `artifacts/reported_results/simplernn_scaling_summary.csv`

These are intentionally separate from `data/`, because `data/` remains ignored to avoid accidentally committing multi-gigabyte intermediate files.

## Guideline alignment

This setup is designed to match the course expectations:

- Dependencies are explicit through `requirements.txt`.
- The teaching team can run a lightweight end-to-end Colab reproduction without local files.
- The repo contains archived official metrics for the tables that would be expensive to regenerate in every Colab session.
- The document above also includes the exact commands needed to regenerate the larger experiments when the underlying PGN is available.
