# Road Accident Prediction

A PyTorch research notebook for predicting critical road events from dashcam video using the Nexar collision prediction dataset. It compares temporal models and late fusion on video windows ending **0.5, 1.0, and 1.5 seconds before an event**.

The notebook runs an experiment across three predefined training seeds, freezes all tuning decisions before final test inference, and exports tables, figures, and statistical comparisons. Saved execution output is cleared for a readable repository; trained weights, dataset files, and experiment results are not included.

## Models and evaluation

| Component | Implementation |
| --- | --- |
| Temporal appearance model | EfficientNet-B0 with attention-based bidirectional GRU |
| Temporal shift model | MobileNetV3 Large with temporal shift modules |
| 3D model | Compact 3D CNN inspired by X3D; not canonical X3D-XS |
| Static control | Single-frame EfficientNet |
| Fusion | Equal and tuning-optimized two- and three-branch probability fusion |

The source-video split is stratified by label: 60% training, 20% tuning, and 20% test. Training seeds are `42`, `123`, and `2026`; the data split seed stays fixed at `42`. Model selection uses tuning horizon-mean average precision. Reporting includes AP by horizon, calibration, operating thresholds, paired video bootstrap intervals, permutation tests, and Holm-adjusted comparisons.

## Run on Kaggle

1. Upload `road-accident-prediction.ipynb` to a Kaggle notebook.
2. Attach the Nexar collision prediction competition data using your own authorized access.
3. Enable a GPU accelerator. Enable internet for the initial pretrained Torchvision weight downloads, or provide an existing weight cache. Setting `PRETRAINED=False` changes the experiment and should use a fresh output directory.
4. Check the three paths in the first configuration cell:

   ```python
   RAW_CSV_PATH = '/kaggle/input/competitions/nexar-collision-prediction/train.csv'
   RAW_VIDEO_DIR = '/kaggle/input/competitions/nexar-collision-prediction/train'
   WORK_DIR = '/kaggle/working/nexar_multiseed'
   ```

5. Keep `EXECUTION_MODE='one_click'` and `STAGES=()`, then run all cells in order.

The CSV must contain `id`, `target`, and `time_of_event`. The optional `time_of_alert` column enables alert-relative diagnostics. Videos must be named `<normalized_id>.mp4`; numeric IDs are zero-padded to five characters. Unreadable videos and videos without sufficient temporal coverage are excluded by the audit.

For an initial pilot, set `DRY_RUN=True`, reduce `EPOCHS`, and choose a separate `WORK_DIR`. Pilot results are not confirmatory results. Return to a fresh directory for the full experiment.

## Local setup

Python 3.12 is recommended; the supplied notebook records Python 3.12.12. A CUDA-capable GPU is recommended for training. CPU execution is supported by the notebook but a full run is expensive.

```bash
git clone https://github.com/nazzmullxd/road-accident-prediction.git
cd road-accident-prediction
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and open the notebook:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab road-accident-prediction.ipynb
```

Use a compatible PyTorch/Torchvision pair for your CUDA environment. Update the first cell to point at your local CSV, video directory, and writable output directory. On Windows, start with `NUM_WORKERS=0` and `PERSISTENT_WORKERS=False` to avoid notebook multiprocessing issues; Kaggle/Linux is the primary training environment.

## Experiment workflow

1. Audit videos and freeze the source-video split.
2. Extract and verify a shared, memory-mapped RGB `uint8` cache.
3. Train all declared model seeds, including event-centered diagnostics and strictly pre-event models.
4. Generate tuning predictions and freeze model, fusion, calibration, and threshold selections.
5. Verify all selection locks before opening the final test gate.
6. Generate sealed test predictions, per-seed reports, measured inference benchmarks, and aggregate results across all declared seeds.

Each video caches four windows of 16 frames at 224 × 224 pixels, approximately 9.19 MiB per video before small file overheads. The default cache limit is 16 GiB with 1.5 GiB reserved free space; raw videos and checkpoints require additional storage. The default full run trains seven model/regime combinations per seed for up to 40 epochs each. Runtime depends on hardware and data eligibility.

The one-click orchestrator resumes verified artifacts in the same output directory. Use a new `WORK_DIR` when changing scientific settings. Do not delete locks or overwrite test predictions to reselect models after inspecting test outcomes.

## Generated outputs

```text
WORK_DIR/
├── manifests/             # Audit, split, cache, and protocol records
├── uint8_cache/           # Shared extracted frames
├── runs/seed_<seed>/
│   ├── checkpoints/
│   ├── predictions/
│   ├── selection/         # Frozen tuning selections and completion records
│   ├── tables/
│   ├── figures/
│   ├── stats/
│   └── logs/
├── tables/               # Across-seed results and mean/SD summaries
├── figures/              # Aggregate figures (PDF and PNG)
├── stats/                # Across-seed comparisons and method metadata
└── one_click_progress.json
```

Key aggregate files include `tables/multiseed_table_II_mean_sd.csv`, `stats/multiseed_H1_H2.csv`, and `figures/fig_2_multiseed_horizons.pdf`. Outputs stay outside version control by default.

## Repository checks

```bash
python -m pip install -r requirements-dev.txt
python scripts/validate_notebook.py
```

GitHub Actions runs the same lightweight checks: notebook schema, Python syntax, clean notebook output, and the notebook's temporal-window/source-ID checks without training or downloading data. These checks do not establish model performance or validate a complete GPU training run.

## Scope

This is a research experiment. Source IDs are disjoint across splits, but near-duplicate scenes across different IDs are not ruled out. Event-centered evaluation is a diagnostic and does not establish pre-event prediction quality. Reported cached-input inference timings exclude raw video decoding and camera acquisition. Results do not establish deployment safety or generalization to other datasets.

Dataset access and pretrained weights remain subject to their respective terms. No dataset files or trained checkpoints are redistributed here.
