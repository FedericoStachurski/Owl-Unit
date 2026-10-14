# Owl Unit — DeepLabCut 3 training starter

Copy `src/training/*.py` into your existing `~/Owl-Unit/src/training/` folder. Keep your other files in place.

## Install

```bash
conda activate owl-unit
python -m pip install 'deeplabcut' tables opencv-python
```

## Run (uses existing `configs/training/Siviglia.toml`)

```bash
python src/training/prepare_dlc_dataset.py Siviglia
python src/training/train_dlc.py Siviglia --create-dataset
python src/training/train_dlc.py Siviglia --train
```

The first command converts saved, EXIF-normalised and manually rotated images to DeepLabCut's labelled-data format. The placeholder reference AVI is only used to initialize the DeepLabCut project; it is *not* training data.

Training settings such as `epochs`, `batch_size`, `device` and `train_fraction` are read from the dataset TOML. DLC performs a train/test split using the configured `train_fraction`; the `validation_fraction` and `test_fraction` fields of the starter TOML are **not** independently applied by this initial integration. Do not treat reported DLC test error as an untouched three-way holdout. We can add a dedicated held-out test split next.

The generated DeepLabCut project lives in `data/processed/dlc/Siviglia/...` and the `owl_unit_dlc_project.json` file contains the exact `config.yaml` path. The original photographs and annotation CSV files are never modified. Repeat `prepare` only before beginning training, since it replaces the exported labelled images and `.h5` file.

**Be aware:** this creates a fresh DLC project, with DLC's native project creation and dataset APIs. DLC versions differ, so if an import/configuration error occurs, share the traceback and `python -c 'import deeplabcut; print(deeplabcut.__version__)'`.
