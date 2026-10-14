#!/usr/bin/env python3

"""
Owl Unit - Generate dataset-specific DeepLabCut training configuration.

Examples
--------
python src/training/create_training_config.py Siviglia
python src/training/create_training_config.py Scotland
"""

import argparse
import csv
from pathlib import Path


LANDMARKS = [
    "lower_left",
    "middle_left",
    "upper_left",
    "top_middle",
    "upper_right",
    "middle_right",
    "lower_right",
    "bottom_middle",
    "beak_tip",
]


def get_project_root():
    return Path(__file__).resolve().parents[2]


def count_images(directory):

    extensions = {
        ".jpg", ".jpeg", ".png",
        ".tif", ".tiff", ".webp"
    }

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def count_annotations(manifest_path):

    if not manifest_path.exists():
        return 0

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        return sum(
            row.get("status", "").strip().lower() == "annotated"
            for row in reader
        )


def generate_config(dataset, root, overwrite=False):

    # --------------------------------------------------------
    # DATASET PATHS
    # --------------------------------------------------------

    images_dir = root / "data" / "raw_images" / dataset

    annotations_dir = (
        root / "data" / "annotations" / dataset
    )

    manifest_path = (
        annotations_dir / "owl_manifest.csv"
    )

    keypoints_path = (
        annotations_dir / "owl_keypoints.csv"
    )

    config_dir = root / "configs" / "training"

    config_path = config_dir / f"{dataset}.toml"

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not images_dir.is_dir():

        raise FileNotFoundError(
            f"Image directory not found: {images_dir}"
        )

    if not manifest_path.is_file():

        raise FileNotFoundError(
            f"Annotation manifest not found: {manifest_path}"
        )

    if not keypoints_path.is_file():

        raise FileNotFoundError(
            f"Keypoints CSV not found: {keypoints_path}"
        )

    if config_path.exists() and not overwrite:

        raise FileExistsError(
            f"Configuration already exists: {config_path}\n"
            "Use --overwrite to replace it."
        )

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    n_images = count_images(images_dir)

    n_annotated = count_annotations(manifest_path)

    # --------------------------------------------------------
    # GENERATE TOML
    # --------------------------------------------------------

    config_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    landmarks_toml = ",\n".join(
        f'    "{name}"' for name in LANDMARKS
    )

    content = f"""\
# ============================================================
# OWL UNIT - DEEPLABCUT TRAINING CONFIGURATION
# Dataset: {dataset}
# ============================================================

[project]
name = "Owl Unit"
dataset = "{dataset}"

[paths]
images = "data/raw_images/{dataset}"
annotations = "data/annotations/{dataset}"
manifest = "data/annotations/{dataset}/owl_manifest.csv"
keypoints = "data/annotations/{dataset}/owl_keypoints.csv"

dlc_project = "data/processed/dlc/{dataset}"
results = "results/training/{dataset}"

[dataset]
total_images = {n_images}
annotated_images = {n_annotated}

[model]
framework = "deeplabcut"
engine = "pytorch"
backbone = "resnet_50"

[training]
epochs = 200
batch_size = 8
train_fraction = 0.80
validation_fraction = 0.10
test_fraction = 0.10
random_seed = 42
device = "cuda"

[landmarks]
names = [
{landmarks_toml}
]
"""

    config_path.write_text(
        content,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("OWL UNIT - TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Dataset:          {dataset}")
    print(f"Total images:     {n_images}")
    print(f"Annotated images: {n_annotated}")
    print(f"Configuration:    {config_path}")
    print("=" * 60)

    return config_path


def main():

    parser = argparse.ArgumentParser(
        description="Generate Owl Unit training configuration"
    )

    parser.add_argument(
        "dataset",
        help="Dataset name, e.g. Siviglia"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing configuration"
    )

    args = parser.parse_args()

    dataset = args.dataset

    if (
        dataset in (".", "..")
        or Path(dataset).name != dataset
        or "/" in dataset
        or "\\" in dataset
    ):
        parser.error("Dataset must be a single folder name.")

    generate_config(
        dataset=dataset,
        root=get_project_root(),
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()