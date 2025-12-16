# Deep Learning IDS on NSL-KDD (Binary Classification)

This project trains and evaluates intrusion detection models on the **NSL-KDD** dataset for **binary classification**:
**normal (0)** vs **attack (1)**. We compare a Logistic Regression baseline with a PyTorch MLP and evaluate an
imbalance-handling extension (class weighting / `pos_weight`). The main metrics are **attack-class Recall** and **F1**.

## Repository Structure

* `scripts/` — runnable scripts for download, preprocessing, training, and table generation
* `src/` — helper modules (dataset parsing, utilities)
* `data/` — dataset (downloaded in Colab; not committed to git)
* `results/` — generated metrics + LaTeX table + training curve
* `report/` — AAAI 2023 report source (LaTeX)
* `Final_Project_Deep_Learning.pdf` — final compiled paper (for convenience)

## Google Colab (recommended)

These steps are written for **Google Colab**. You can either **clone the repo** or **upload a zip** of the project.

### 0) (Optional) Enable GPU

Colab menu: **Runtime → Change runtime type → Hardware accelerator → GPU**.

### 1A) Option A — Clone from GitHub

In a Colab cell:

```bash
!git clone <YOUR_NEW_REPO_URL_HERE>
%cd <YOUR_REPO_FOLDER_NAME>/IDS_Project
!pip -q install -r requirements.txt
```

### 1B) Option B — Upload a zip instead

If you downloaded the project as a zip, upload it in Colab and unzip:

```python
from google.colab import files
uploaded = files.upload()  # choose your .zip file
```

```bash
!unzip -q <YOUR_ZIP_FILE_NAME>.zip -d .
%cd IDS_Project
!pip -q install -r requirements.txt
```

## Reproduce end-to-end 

> Run the following commands from the directory that contains `scripts/`, `src/`, `results/`, and `report/`
> (e.g., after `cd IDS_Project` as shown above).

### 1) Download NSL-KDD

```bash
python scripts/download_nslkdd.py --out_dir data/raw
```

Expected files:

* `data/raw/KDDTrain+.txt`
* `data/raw/KDDTest+.txt`

### 2) Preprocess - one-hot + scaling + train/val split

```bash
python scripts/preprocess.py --raw_dir data/raw --out_dir data/processed --val_size 0.15 --seed 42
```

Outputs:

* `data/processed/train.npz`
* `data/processed/val.npz`
* `data/processed/test.npz`
* `data/processed/preprocessor.joblib`
* `data/processed/meta.json`

### 3) Train Logistic Regression baselines

Unweighted:

```bash
python scripts/train_baseline.py --data_dir data/processed --results_dir results --model logreg --class_weight none --seed 42
```

Balanced class weights:

```bash
python scripts/train_baseline.py --data_dir data/processed --results_dir results --model logreg --class_weight balanced --seed 42
```

### 4) Train MLP

Unweighted:

```bash
python scripts/train_mlp.py --data_dir data/processed --results_dir results --imbalance none --seed 42
```

Weighted loss (`pos_weight`):

```bash
python scripts/train_mlp.py --data_dir data/processed --results_dir results --imbalance pos_weight --seed 42
```

This also generates:

* `results/mlp_training_curve.pdf`
* `results/mlp_history.json`

### 5) Generate the LaTeX results table

```bash
python scripts/make_results_table.py --results_dir results --out results/results_table.tex
```

## Outputs

After running the pipeline above, you should have (names may vary slightly by script version):

* `results/baseline_logreg_none_metrics.json`
* `results/baseline_logreg_balanced_metrics.json`
* `results/mlp_none_metrics.json`
* `results/mlp_pos_weight_metrics.json`
* `results/results_table.tex`
* `results/mlp_training_curve.pdf`

## Optional - Build the report in Colab 

If you just need the paper, the compiled PDF is already included: `Final_Project_Deep_Learning.pdf`.

If you want to **recompile** `report/main.tex` in Colab:

1. Put AAAI style files into `report/`:

* `aaai23.sty`
* `aaai23.bst`

2. Install LaTeX (first time only per Colab runtime):

```bash
!apt-get -qq update
!apt-get -qq install -y texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra latexmk
```

3. Copy generated artifacts into `report/` (so `\input{results_table.tex}` and the figure resolve):

```bash
%cd report
!cp ../results/results_table.tex .
!cp ../results/mlp_training_curve.pdf .
```

4. Compile:

```bash
!latexmk -pdf -bibtex main.tex
```

The output will be:

* `report/main.pdf`

## Notes

* The raw dataset is not committed. In Colab, re-running from scratch is easiest: delete the `data/` folder and rerun.
* Attack is treated as the positive class (`pos_label=1`) for precision/recall/F1.
