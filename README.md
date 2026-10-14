# 🦉 Owl Unit

**Dataset-aware owl facial landmark annotation and DeepLabCut training**

Owl Unit is a Python project for building an annotated dataset of owl faces and training a landmark detector. It combines a manual **napari** annotation interface with a command-line workflow for converting annotations to **DeepLabCut (DLC)** format and training a PyTorch-based model.

The project is designed to work with multiple photographic collections—for example, **Siviglia** and **Scotland**—without mixing their annotations or model files. Its principal targets are **eight points along the facial-disc boundary plus the beak tip**.

> **Current scope:** Manual annotation, dataset-specific training configuration, DLC project preparation, training dataset creation, and model training are implemented in the current scripts. Automated evaluation, inference, and publication-ready measurement tools are planned work, not existing CLI commands.

---

## Contents

1. [Overview](#overview)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [CLI command reference](#cli-command-reference)
6. [Annotation workflow](#annotation-workflow)
7. [Landmark definitions](#landmark-definitions)
8. [Annotation file formats](#annotation-file-formats)
9. [Dataset-specific configuration](#dataset-specific-configuration)
10. [DeepLabCut data preparation](#deeplabcut-data-preparation)
11. [Training](#training)
12. [Adding a new dataset](#adding-a-new-dataset)
13. [Data management and Git](#data-management-and-git)
14. [Troubleshooting](#troubleshooting)
15. [Limitations and next steps](#limitations-and-next-steps)

---

## Overview

The complete pipeline is:

```text
Photographic collection (e.g. Siviglia)
                 │
                 ▼
       napari manual annotation
          ├─ Rotate image upright
          ├─ Record facing direction
          ├─ Annotate 8 facial-disc points
          └─ Annotate beak tip
                 │
                 ▼
       Dataset-specific CSV files
          ├─ owl_manifest.csv
          └─ owl_keypoints.csv
                 │
                 ▼
     Dataset-specific training TOML
                 │
                 ▼
     Convert to DeepLabCut project
          ├─ Correctly rotated images
          ├─ DLC-format landmark table
          └─ Source-image mapping
                 │
                 ▼
      Create DLC training dataset
                 │
                 ▼
      Train keypoint detection model
```

**Why annotate manually?** An earlier OWLv2 experiment explored automatically identifying image rotation and facing direction, but did not produce sufficiently reliable results for this dataset. Owl Unit therefore records these decisions explicitly during annotation. **Left/right facing** and **image rotation** are independent metadata: turning an image upright does not change the owl's actual pose.

### Working principles

- **Preserve the source photographs.** Rotations are stored as metadata; image copies for training are generated separately.
- **Preserve the annotation coordinate system.** Landmark coordinates are recorded in both the original, EXIF-normalized image and manually oriented image.
- **Keep datasets distinct.** Each collection has its own annotation folder and training configuration.
- **Use a CLI for every implemented pipeline stage.** Individual Python scripts are implementation details.
- **Do not infer invisible landmarks.** Mark them missing/occluded.

---

## Repository layout

The layout below reflects the current project organisation, without requiring the source directories to be rearranged:

```text
Owl-Unit/
├── README.md                        # This document
├── pyproject.toml                    # Python package and owl-unit CLI entry point
├── owl_unit_setup.toml              # Annotation configuration
├── environment.yml                  # Conda environment record
├── requirements.txt                 # Python dependency record
├── run.py                           # Legacy script dispatcher (CLI preferred)
├── configs/
│   └── training/
│       ├── Siviglia.toml            # Dataset-specific training settings
│       └── Scotland.toml            # Created when that dataset is ready
├── src/
│   ├── annotations/
│   │   ├── napari_annotate.py       # Manual annotation GUI / CSV persistence
│   │   └── test_coordinates.py      # Coordinate transform checks
│   ├── cli/
│   │   └── cli.py                   # Registered CLI subcommands
│   ├── training/
│   │   ├── create_training_config.py
│   │   ├── prepare_dlc_dataset.py
│   │   └── train_dlc.py
│   └── evaluation/                  # Future evaluation tooling
├── data/
│   ├── raw_images/
│   │   ├── Siviglia/                # Original input photographs (local)
│   │   └── Scotland/                # Example additional collection
│   ├── annotations/
│   │   ├── Siviglia/
│   │   │   ├── owl_manifest.csv
│   │   │   └── owl_keypoints.csv
│   │   └── Scotland/
│   └── processed/
│       ├── upright_images/          # Optional oriented copies
│       └── dlc/                     # Generated DeepLabCut projects
│           └── Siviglia/
├── notebooks/                       # Exploratory notebooks
└── results/                         # Experimental outputs (local)
```

Directories for other datasets are examples, not necessarily already present. Generated directories and files appear when you run the corresponding workflow.

---

## Installation

### Prerequisites

- Python **3.11** (the development environment used for this project).
- Conda or another Python environment manager.
- A desktop session with a functional Qt display for napari (including WSLg if running in Windows Subsystem for Linux).
- DeepLabCut **3.x** for the PyTorch training workflow.
- An appropriate PyTorch installation; a CUDA-capable GPU is recommended for training, but not required for annotation.

From the repository root:

```bash
cd ~/Owl-Unit
conda activate owl-unit
python -m pip install -e .
```

The `-e .` installs the **current directory** as an editable package. Changes to Python source files are then picked up without reinstalling in most cases. Changing package metadata or entry points in `pyproject.toml` may require another installation.

If the environment has not yet been created, one option is:

```bash
conda create -n owl-unit python=3.11 -y
conda activate owl-unit
python -m pip install -e .
```

For the training stage, install and verify DeepLabCut separately, so that you can check its effect on the existing napari/PyTorch environment:

```bash
python -m pip install deeplabcut tables opencv-python
python -c "import deeplabcut; print('DeepLabCut:', deeplabcut.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

**Dependency caution:** DLC and napari have overlapping scientific, Qt and ML dependencies; pin versions in a reproducible environment when the pipeline is stable. If modifying an existing working environment, consider cloning it first.

Verify the CLI:

```bash
owl-unit --help
```

Expected subcommands in the current implementation: `annotate`, `setup-training`, `prepare`, and `train`.

---

## Quick start

Assuming the images are in `data/raw_images/Siviglia/`:

```bash
# 1. Annotate a small pilot (the first five images)
owl-unit annotate Siviglia --limit 5

# 2. Create the dataset-specific training configuration
owl-unit setup-training Siviglia

# 3. Convert saved labels into a DeepLabCut project
owl-unit prepare Siviglia

# 4. Generate DeepLabCut's training files
owl-unit train Siviglia --create-dataset

# 5. Train the model
owl-unit train Siviglia --train
```

**Do not start full training immediately after a tiny pilot.** The preparation script requires at least **three annotated images**, but that is a technical minimum, *not* a recommended amount of training data. For a realistic experiment, collect a varied sample covering different owls, poses, lighting conditions, facial scales, and occlusions.

---

## CLI command reference

| Command | Purpose | Principal outputs |
|---|---|---|
| `owl-unit annotate Siviglia` | Open the napari annotation interface for a dataset | `data/annotations/Siviglia/*.csv` |
| `owl-unit annotate Siviglia --limit 5` | Annotate/review a small subset | Same location; existing records remain |
| `owl-unit annotate --prompt` | Enter input and output directories interactively | Chosen output directory |
| `owl-unit annotate Siviglia --input-dir PATH --output-dir PATH` | Override dataset-derived paths | Chosen output directory |
| `owl-unit setup-training Siviglia` | Generate a dataset-specific training TOML | `configs/training/Siviglia.toml` |
| `owl-unit setup-training Siviglia --overwrite` | Regenerate an existing TOML | Replaces the previous TOML; manual edits are lost |
| `owl-unit prepare Siviglia` | Convert manual annotations and images to DLC format | `data/processed/dlc/Siviglia/` |
| `owl-unit prepare Siviglia --config PATH` | Use a specified training TOML | DLC project selected by TOML |
| `owl-unit train Siviglia --create-dataset` | Create DLC training/test files | DLC project training files |
| `owl-unit train Siviglia --train` | Train the DLC PyTorch network | DLC checkpoints and logs |
| `owl-unit train Siviglia --create-dataset --train` | Do both sequentially | Both kinds of outputs |
| `owl-unit train Siviglia --config PATH --train` | Train with a specified TOML | Project selected by TOML |

Help is available for each command:

```bash
owl-unit annotate --help
owl-unit setup-training --help
owl-unit prepare --help
owl-unit train --help
```

`evaluate`, `predict`, and `visualise` are **not yet implemented CLI subcommands**.

---

## Annotation workflow

Launch the napari interface:

```bash
owl-unit annotate Siviglia
```

The current annotation GUI supports:

1. **Load and browse photographs.** The image list and (where using the updated interface) thumbnail filmstrip help navigate a collection; previous annotations can be revisited.
2. **Select facing direction.** Available values are `Left`, `Right`, `Front`, and `Unclear`.
3. **Rotate into an upright orientation.** Rotate left or right in **90-degree increments**. This is non-destructive.
4. **Place nine points.** Click the location for each named landmark. An already placed point can be selected and corrected.
5. **Flag occluded points.** Use **Not visible / occluded** rather than guessing a coordinate.
6. **Save and continue.** Save after finishing the nine landmarks; the app advances to another pending image.
7. **Revisit saved records.** When a previously annotated image is loaded, the saved rotation, facing direction and landmarks are restored for review.

### Save behaviour

- The action **Save + next unlabelled** writes the annotations for the current image; unsaved edits are not persisted merely by clicking in the viewer.
- **Skip image** records the image as skipped. In the existing implementation, skipping an already annotated image (after confirmation) **removes its saved landmark records**, so use this carefully.
- Files are written as CSV, with a temporary-file/rename operation intended to reduce the chance of partial file writes.
- By default, original photographs are **not changed**. When the **Save oriented image copy on save** option is enabled, an oriented copy is written to the processed-image directory.
- On opening a collection, the tool starts at the first image whose status is `pending` (or falls back to the start when there is no pending image).

### Coordinate conventions

The interface stores coordinates in two systems:

| Coordinates | Interpretation |
|---|---|
| `x_original`, `y_original` | Position on the original photograph **after applying its EXIF orientation** |
| `x_oriented`, `y_oriented` | Position on the photograph **after the manual 0/90/180/270° rotation** |

Coordinates are in pixels; the top-left of the image is the origin, **x increases rightward** and **y increases downward**. Rotation is recorded as `rotation_ccw_deg`, measured counterclockwise. The DLC preparation script uses the **oriented** coordinates paired with the corresponding oriented image copy. Rotations are not the same as a horizontal mirror flip.

> **Convention:** Left/right landmark names refer to **image-left and image-right in the upright annotated view**, not necessarily the owl's anatomical left and right. Write a landmark-placement protocol and use it consistently across annotators and datasets.

---

## Landmark definitions

The project currently defines **nine** landmarks:

| Order | Name | Intended location |
|---:|---|---|
| 1 | `lower_left` | Lower-left part of the facial-disc boundary |
| 2 | `middle_left` | Midpoint on the left boundary |
| 3 | `upper_left` | Upper-left part of the boundary |
| 4 | `top_middle` | Uppermost central boundary point |
| 5 | `upper_right` | Upper-right part of the boundary |
| 6 | `middle_right` | Midpoint on the right boundary |
| 7 | `lower_right` | Lower-right part of the boundary |
| 8 | `bottom_middle` | Lowermost central boundary point |
| 9 | `beak_tip` | Visible tip of the beak |

These descriptions establish the *names*, not a finished biological annotation protocol. Decide precisely which edge of the facial disc each point represents—particularly where plumage makes the boundary ambiguous—before scaling up to the full dataset. **Do not silently rename or reorder landmarks after collecting labels**, since the saved CSV columns and the trained DLC model depend on them.

For a side-facing image, some landmarks may be hidden. Record them as missing instead of assigning inferred positions. Facial size estimated from 2D images is in **pixels** unless camera calibration or a known physical scale is available.

---

## Annotation file formats

Every dataset has its own annotation directory, e.g.:

```text
data/annotations/Siviglia/
├── owl_manifest.csv
└── owl_keypoints.csv
```

### `owl_manifest.csv`

One row per photograph, including the identity, annotation status, facing direction, rotation and (in the current implementation) wide-format coordinate columns for the nine points.

Important fields include:

| Field | Meaning |
|---|---|
| `dataset` | Dataset identifier, such as `Siviglia`, **if the dataset-field additions have been applied** |
| `image_id` | Image path relative to that dataset's image root |
| `image_path` | Stored image location |
| `status` | `pending`, `annotated`, or `skipped` |
| `facing` | `Left`, `Right`, `Front`, or `Unclear` |
| `rotation_ccw_deg` | Manual rotation: `0`, `90`, `180`, or `270` |
| `original_width`, `original_height` | Source image dimensions after EXIF correction |
| `oriented_width`, `oriented_height` | Dimensions after manual rotation |
| `visible_points`, `missing_points` | Landmark counts |
| `corrected_image_path` | Optional saved upright-copy path |
| `updated_at_utc` | Most recent save time in UTC |
| `<landmark>_x`, `<landmark>_y`, `<landmark>_visibility` | Wide-format fields for each landmark |

### `owl_keypoints.csv`

One row per image **per landmark**. Fields include:

```text
image_id,landmark,visibility,x_oriented,y_oriented,x_original,y_original
```

A `dataset` column may also be present if that enhancement has been applied. For missing landmarks, the coordinate fields remain empty and `visibility` is `missing`. For annotated visible landmarks, `visibility` is `visible` and the coordinates are numeric.

**Why both files?** The manifest is convenient for browsing, filtering, and understanding the state of the collection; the long-form keypoint table is convenient for machine-learning conversion and point-wise analysis.

> **Multiple-dataset safety:** The application identifies images by their relative `image_id`. Two collections may contain identical file names. Keep each collection's manifest/keypoints in its **own directory**, even if the optional `dataset` column is present.

---

## Dataset-specific configuration

Owl Unit uses **two different kinds of TOML file**:

| File | Purpose |
|---|---|
| `pyproject.toml` | Defines the installable Python package, dependencies, and `owl-unit` CLI entry point |
| `owl_unit_setup.toml` | Defines annotation options and default input/output paths |
| `configs/training/<DATASET>.toml` | Defines the dataset-specific DeepLabCut training experiment |

`pyproject.toml` is **not** interchangeable with an annotation or training config. A separate `setup.py` is generally not required when `pyproject.toml` already defines the build system and package metadata.

### Annotation TOML

Example excerpt from `owl_unit_setup.toml`:

```toml
[paths]
images_root = "data/raw_images/Siviglia"
output_dir = "data/annotations"

[images]
extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"]
recursive = true

[napari_annotation]
manifest_csv = "data/annotations/owl_manifest.csv"
keypoints_csv = "data/annotations/owl_keypoints.csv"
corrected_dir = "data/processed/upright_images"
save_corrected_images = false
landmarks = [
    "lower_left", "middle_left", "upper_left", "top_middle",
    "upper_right", "middle_right", "lower_right", "bottom_middle",
    "beak_tip",
]
```

Specifying a dataset with `owl-unit annotate Siviglia` overrides the default input and annotation-output directories; it uses:

```text
data/raw_images/Siviglia/
data/annotations/Siviglia/
```

### Training TOML

Generate it automatically **after** creating the dataset's annotation CSVs:

```bash
owl-unit setup-training Siviglia
```

Output: `configs/training/Siviglia.toml`. An example of the configurable sections:

```toml
[project]
name = "Owl Unit"
dataset = "Siviglia"

[paths]
images = "data/raw_images/Siviglia"
annotations = "data/annotations/Siviglia"
manifest = "data/annotations/Siviglia/owl_manifest.csv"
keypoints = "data/annotations/Siviglia/owl_keypoints.csv"
dlc_project = "data/processed/dlc/Siviglia"
results = "results/training/Siviglia"

[model]
framework = "deeplabcut"
engine = "pytorch"
backbone = "resnet_50"

[training]
epochs = 200
batch_size = 8
save_epochs = 10
train_fraction = 0.80
validation_fraction = 0.10
test_fraction = 0.10
random_seed = 42
device = "cuda"
```

These are **Owl Unit settings**, not DeepLabCut's native `config.yaml`. The preparation/training scripts map selected fields into the DLC project and its training APIs.

**Current implementation notes:**

- The generator records image counts and the number of `annotated` images **at the time the TOML is generated**. Those counts may become stale as more images are labelled; the preparation script reads the annotation CSVs again when it runs.
- The preparation code currently uses `train_fraction` for DLC's `TrainingFraction` setting. A custom **independent train/validation/test split is not yet implemented**; the presence of `validation_fraction` and `test_fraction` in the TOML does not implement it.
- The current training code reads `epochs`, `batch_size`, `save_epochs`, and `device` for the DLC call. `random_seed` is not yet applied to the internal DLC split by the supplied scripts.
- `--overwrite` replaces an existing TOML with generator defaults; copy experiment configs before changing them for reproducibility.

---

## DeepLabCut data preparation

Prepare a dataset after collecting and saving enough annotations:

```bash
owl-unit prepare Siviglia
```

The preparation stage (`src/training/prepare_dlc_dataset.py`) does the following:

1. Reads `configs/training/Siviglia.toml` and checks that its dataset name matches the requested dataset.
2. Reads the dataset's `owl_manifest.csv` and `owl_keypoints.csv`.
3. Selects rows with `status = annotated` and checks that at least three images are available.
4. Validates each image's presence, rotation (`0`, `90`, `180`, `270`), and numeric visible-point coordinates.
5. Applies EXIF orientation and then the saved counterclockwise manual rotation to make training-image copies.
6. Writes DLC-compatible **HDF5 and CSV landmark tables**; hidden landmarks are represented as `NaN`.
7. Creates or reuses a DLC project and updates its configured body parts and training fraction.
8. Writes `source_image_mapping.csv` to preserve the correspondence to each original photo, including its facing label and manual rotation.

The preparation script uses a short **synthetic reference video** solely to initialise a standard DLC project. It is not an owl training example.

The DLC project root is controlled by `paths.dlc_project` in the training TOML. Inside the resulting project, relevant outputs include:

```text
data/processed/dlc/Siviglia/
├── owl_unit_dlc_project.json     # Points to generated DLC config.yaml
├── source_image_mapping.csv      # Source-to-DLC image mapping
├── _owl_unit_reference.avi        # Project setup artifact
└── <DLC-created-project-folder>/
    ├── config.yaml
    ├── labeled-data/
    │   └── <reference-video-name>/
    │       ├── img_000000.png
    │       ├── img_000001.png
    │       ├── CollectedData_<scorer>.h5
    │       └── CollectedData_<scorer>.csv
    └── ...                       # DLC's additional project files
```

**Important:** The converter indexes generated images as `img_000000.png`, `img_000001.png`, etc. Re-running preparation after changing which images are annotated may change this mapping and rewrite generated files. **Keep the source CSVs immutable for a training run, back up the DLC project if needed, and rebuild/recheck DLC training files when annotations change.** Do not treat model results from different dataset snapshots as directly comparable without recording which data was used.

---

## Training

DeepLabCut training is split into two CLI stages.

### 1. Create the DLC training dataset

```bash
owl-unit train Siviglia --create-dataset
```

This runs DeepLabCut's `create_training_dataset()` using the PyTorch engine, network choice from the training TOML, and its existing project `config.yaml`. DLC creates its own training/test files.

### 2. Run the model training

```bash
owl-unit train Siviglia --train
```

This calls DLC's `train_network()` using the configured device, number of epochs, batch size, and checkpoint interval. **Checkpoint storage and console output are governed by DLC's generated project structure.** The `results` path in the TOML is not currently used by the supplied training wrapper to relocate DLC checkpoints.

For a first smoke test, use a modest configuration:

```toml
[training]
epochs = 5
batch_size = 4
save_epochs = 1
train_fraction = 0.80
device = "cuda"
```

On a system without working CUDA support, set `device = "cpu"` for the test. GPU availability can be checked with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Once a short training run is successful, configure a longer run **as a deliberately separate experiment**. The current wrapper does **not** explicitly manage the choice between starting from scratch and resuming a checkpoint; check the generated DLC configuration and checkpoint state before starting a full run in the same project.

### Data partitioning and scientific validity

A large number of images does not imply a large number of **independently annotated examples**. To evaluate generalisation, reserve images (preferably grouped by **individual owl** or capture session where known) for independent assessment; otherwise near-identical photographs can leak across training and evaluation. The current wrapper does not enforce a grouped holdout or a separate untouched test dataset.

---

## Adding a new dataset

To work with another collection named `Scotland`:

**1. Add photographs** to:

```text
data/raw_images/Scotland/
```

**2. Annotate and save a pilot:**

```bash
owl-unit annotate Scotland --limit 10
```

The new CSVs are created in:

```text
data/annotations/Scotland/
```

**3. Generate the training configuration:**

```bash
owl-unit setup-training Scotland
```

**4. Prepare and train:**

```bash
owl-unit prepare Scotland
owl-unit train Scotland --create-dataset
owl-unit train Scotland --train
```

The source image folders, CSV files, training TOMLs, and DLC project directories remain separate between Siviglia and Scotland. **Use identical landmark definitions and annotation conventions** if you plan to combine datasets later.

---

## Data management and Git

The repository is intended to share **code and reproducible configuration**, not automatically upload all photographs and generated models.

A typical `.gitignore` policy is:

```gitignore
# Python cache and build artifacts
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/

# Local environments
.venv/
env/
.env

# Data (keep private/local unless explicitly cleared for sharing)
data/raw_images/
data/annotations/
data/processed/

# Generated training outputs
results/
models/
*.pt
*.pth
*.ckpt

# Notebook caches
.ipynb_checkpoints/
```

The training TOML files under `configs/training/` are suitable for version control **after checking that they contain no sensitive or machine-specific paths**. Annotation CSVs may ultimately be valuable to version, but first verify their provenance, rights, and appropriate sharing policy. Keeping them out of Git does **not** back them up: use an appropriate separate backup.

Before committing:

```bash
git status
git add -n .                # Preview what would be staged
git diff --cached --stat    # Inspect staged changes, if any
```

After review, stage only intended source code and configuration:

```bash
git add README.md pyproject.toml owl_unit_setup.toml src/ configs/
git status
git commit -m "Document Owl Unit annotation and DeepLabCut workflow"
git push origin main
```

---

## Troubleshooting

### `owl-unit: command not found`

Activate the environment and perform an editable install:

```bash
conda activate owl-unit
cd ~/Owl-Unit
python -m pip install -e .
owl-unit --help
```

### `pip install -e .` fails because of `setup.py`

`setup.py` must contain **Python**, not TOML. Annotation settings belong in `owl_unit_setup.toml`, and installation metadata belongs in `pyproject.toml`. A minimal optional legacy file is:

```python
from setuptools import setup
setup()
```

A modern `pyproject.toml`-driven package need not include `setup.py` at all.

### TOML decode error

Check that the configuration contains TOML syntax (for example `[paths]` and `images_root = "..."`), not pasted Python code:

```bash
python -c "import tomllib; tomllib.load(open('owl_unit_setup.toml','rb')); print('TOML valid')"
```

### `Invalid step script: ...` using `run.py`

`run.py` is the older dispatcher. Prefer the installed CLI, `owl-unit annotate Siviglia`, which imports the annotation module directly and avoids manually specifying step-script paths.

### Qt `QWidget` error

Make sure `napari_annotate.py` imports the Qt widgets it uses, particularly `QWidget`, from `qtpy.QtWidgets` before defining the custom annotation widget.

### WSL / OpenGL warning (`libEGL`, `dri2`)

Some WSL graphics configurations emit EGL warnings even when the viewer can still open. If the viewer genuinely fails to display, test software rendering:

```bash
QT_OPENGL=software LIBGL_ALWAYS_SOFTWARE=1 owl-unit annotate Siviglia --limit 5
```

This is a diagnostic workaround, not necessarily the best long-term graphics configuration.

### Training configuration cannot be generated

`owl-unit setup-training DATASET` expects:

```text
data/raw_images/DATASET/
data/annotations/DATASET/owl_manifest.csv
data/annotations/DATASET/owl_keypoints.csv
```

If annotations were previously saved directly under `data/annotations/`, **back them up and carefully migrate them** to the dataset-specific folder before generating the TOML. Do not overwrite newer files.

### `prepare` complains about missing or invalid annotations

Verify that images were saved as `annotated`, all nine landmark names agree with the training TOML, source files still exist, and visible points are within image bounds in the **oriented** coordinate system. Missing/occluded points may have blank coordinates. Preparation currently requires at least three annotated images.

### CUDA was requested but is unavailable

The training wrapper checks `torch.cuda.is_available()` when the configured device starts with `cuda`. Install a compatible PyTorch/CUDA combination or temporarily select `device = "cpu"` in the training TOML.

---

## Limitations and next steps

**Implemented now:**

- Manual napari annotation with rotation, pose labels, nine named landmarks, and CSV persistence.
- Dataset-specific CLI selection and training-TOML generation.
- Conversion of oriented images and landmarks to a DLC-format labelled project.
- DeepLabCut PyTorch training dataset creation and network training.

**Recommended future work:**

1. **Formalise and illustrate the nine-landmark annotation protocol**; perform inter-annotator checks on a small common subset.
2. **Build a true independent evaluation split**, preferably grouped by individual owl/session and stratified by pose when practical.
3. **Add `owl-unit evaluate DATASET`**, reporting Euclidean pixel error per point, median error, normalised error and errors by facing direction.
4. **Add `owl-unit predict DATASET`** to generate labelled previews for previously unseen images.
5. **Record experiment metadata** (software versions, data snapshot, seed, config, model checkpoints, metrics) to support repeatable comparisons.
6. **Develop facial-disc measurements**, with appropriate caution about perspective and pixel-to-physical-unit conversion.

---

## External software

- [napari](https://napari.org/) — interactive image viewer and point-annotation layers.
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) — markerless pose and keypoint estimation.
- [PyTorch](https://pytorch.org/) — deep learning framework used by the DLC PyTorch backend.
- [Python packaging guide](https://packaging.python.org/) — `pyproject.toml` and editable installations.

---

**Project status:** Active development. The dataset-specific annotation-to-training path has been set up, but scientific accuracy and generalisation remain to be demonstrated through held-out evaluation. The project may change as annotation standards, dataset coverage, and model comparisons are refined.
