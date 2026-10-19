#!/usr/bin/env python3

"""
Owl Unit - Generate dataset-specific DeepLabCut training configurations.

This script:
    - Identifies a dataset.
    - Counts available owl images.
    - Reads existing napari annotations.
    - Calculates facing-direction statistics.
    - Selects annotated images based on facing direction.
    - Loads landmark definitions from owl_unit_setup.toml.
    - Generates a dataset-specific training TOML.

Examples
--------
owl-unit setup-training Siviglia

owl-unit setup-training Siviglia --facing left_right

owl-unit setup-training Siviglia --facing left --overwrite

owl-unit setup-training Siviglia --facing right --overwrite

owl-unit setup-training Siviglia \
    --facing left_right \
    --normalize-facing \
    --target-facing left \
    --overwrite
"""

import argparse
import csv
import tomllib

from pathlib import Path
from collections import Counter


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

VALID_FACING = {
    "all",
    "left",
    "right",
    "front",
    "left_right",
}

FACING_GROUPS = {
    "all": {"left", "right", "front", "unclear"},
    "left": {"left"},
    "right": {"right"},
    "front": {"front"},
    "left_right": {"left", "right"},
}


# ============================================================
# PROJECT ROOT
# ============================================================

def get_project_root():
    """
    Return the Owl Unit repository root.

    Expected structure:

    Owl-Unit/
        src/
            training/
                create_training_config.py
    """

    return Path(__file__).resolve().parents[2]


# ============================================================
# DATASET NAME VALIDATION
# ============================================================

def validate_dataset_name(dataset):
    """
    Ensure that the dataset is a single directory name.
    """

    if (
        not dataset
        or dataset in (".", "..")
        or Path(dataset).name != dataset
        or "/" in dataset
        or "\\" in dataset
    ):
        raise ValueError(
            "Dataset must be a single folder name, "
            "for example 'Siviglia'."
        )


# ============================================================
# LOAD PROJECT CONFIGURATION
# ============================================================

def load_project_config(root):
    """
    Load the main Owl Unit annotation configuration.

    Landmark names are read from:

        owl_unit_setup.toml

    This avoids maintaining multiple independent
    landmark definitions.
    """

    config_path = root / "owl_unit_setup.toml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Main configuration not found: {config_path}"
        )

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    landmarks = config.get(
        "napari_annotation", {}
    ).get("landmarks", [])

    if not landmarks:
        raise ValueError(
            "No landmarks defined in owl_unit_setup.toml"
        )

    if len(landmarks) != 9:
        raise ValueError(
            f"Expected 9 landmarks, found {len(landmarks)}."
        )

    if len(set(landmarks)) != len(landmarks):
        raise ValueError(
            "Duplicate landmark names found."
        )

    return config, landmarks


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(directory):
    """
    Count supported images recursively.
    """

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# LOAD ANNOTATION MANIFEST
# ============================================================

def load_manifest(manifest_path):
    """
    Load owl_manifest.csv.

    Expected columns include:

        image_id
        status
        facing
    """

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Annotation manifest not found: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "image_id",
            "status",
            "facing",
        }

        columns = set(reader.fieldnames or [])

        missing = required - columns

        if missing:
            raise ValueError(
                f"Manifest missing columns: {sorted(missing)}"
            )

        rows = list(reader)

    return rows


# ============================================================
# ANNOTATION STATISTICS
# ============================================================

def get_annotation_statistics(rows):
    """
    Calculate overall annotation statistics.

    Only images marked 'annotated' are counted
    in the facing-direction distribution.
    """

    status_counts = Counter(
        row.get("status", "").strip().lower()
        for row in rows
    )

    annotated_rows = [
        row
        for row in rows
        if row.get("status", "").strip().lower()
        == "annotated"
    ]

    facing_counts = Counter(
        row.get("facing", "").strip().lower()
        for row in annotated_rows
    )

    return {
        "total_manifest": len(rows),
        "annotated": len(annotated_rows),
        "pending": status_counts.get("pending", 0),
        "skipped": status_counts.get("skipped", 0),
        "left": facing_counts.get("left", 0),
        "right": facing_counts.get("right", 0),
        "front": facing_counts.get("front", 0),
        "unclear": facing_counts.get("unclear", 0),
    }


# ============================================================
# FACING SELECTION
# ============================================================

def select_annotations(
    rows,
    facing="left_right",
    exclude_unclear=True,
):
    """
    Select annotated photographs based on facing direction.

    Parameters
    ----------
    rows : list[dict]
        Rows loaded from owl_manifest.csv.

    facing : str
        One of:
            all
            left
            right
            front
            left_right

    exclude_unclear : bool
        Whether to exclude images classified as Unclear.

    Returns
    -------
    list[dict]
        Selected annotated rows.
    """

    if facing not in VALID_FACING:
        raise ValueError(
            f"Invalid facing '{facing}'. "
            f"Choose from {sorted(VALID_FACING)}"
        )

    allowed = set(FACING_GROUPS[facing])

    if exclude_unclear:
        allowed.discard("unclear")

    selected = []

    for row in rows:

        status = row.get(
            "status", ""
        ).strip().lower()

        direction = row.get(
            "facing", ""
        ).strip().lower()

        if status != "annotated":
            continue

        if direction not in allowed:
            continue

        selected.append(row)

    return selected


# ============================================================
# TOML UTILITIES
# ============================================================

def toml_string_array(values):
    """
    Convert a list of strings into a formatted TOML array.
    """

    lines = [
        f'    "{value}"'
        for value in values
    ]

    return ",\n".join(lines)


# ============================================================
# GENERATE TRAINING CONFIGURATION
# ============================================================

def generate_config(
    dataset,
    root,
    overwrite=False,
    facing="left_right",
    normalize_facing=False,
    target_facing="left",
    exclude_unclear=True,
):
    """
    Generate a dataset-specific DeepLabCut training TOML.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. Siviglia.

    root : Path
        Owl Unit repository root.

    overwrite : bool
        Whether to overwrite an existing configuration.

    facing : str
        Facing directions to include in training.

    normalize_facing : bool
        Whether to standardise images by horizontal flipping.

    target_facing : str
        Target facing direction after normalisation.

    exclude_unclear : bool
        Exclude photographs with uncertain facing direction.
    """

    root = Path(root).expanduser().resolve()

    validate_dataset_name(dataset)

    if facing not in VALID_FACING:
        raise ValueError(
            f"Invalid facing: {facing}"
        )

    if target_facing not in {"left", "right"}:
        raise ValueError(
            "target_facing must be 'left' or 'right'"
        )

    if normalize_facing and facing not in {
        "left",
        "right",
        "left_right",
    }:
        raise ValueError(
            "Facing normalisation currently supports "
            "left/right photographs only."
        )

    # ========================================================
    # DATASET PATHS
    # ========================================================

    images_dir = (
        root
        / "data"
        / "raw_images"
        / dataset
    )

    annotations_dir = (
        root
        / "data"
        / "annotations"
        / dataset
    )

    manifest_path = (
        annotations_dir
        / "owl_manifest.csv"
    )

    keypoints_path = (
        annotations_dir
        / "owl_keypoints.csv"
    )

    config_dir = (
        root
        / "configs"
        / "training"
    )

    config_path = (
        config_dir
        / f"{dataset}.toml"
    )

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}"
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    if not keypoints_path.is_file():
        raise FileNotFoundError(
            f"Keypoints file not found: {keypoints_path}"
        )

    if config_path.exists() and not overwrite:
        raise FileExistsError(
            f"Configuration already exists: {config_path}\n"
            "Use --overwrite to replace it."
        )

    # ========================================================
    # LOAD LANDMARKS
    # ========================================================

    _, landmarks = load_project_config(root)

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    n_images = count_images(images_dir)

    manifest_rows = load_manifest(
        manifest_path
    )

    stats = get_annotation_statistics(
        manifest_rows
    )

    selected_rows = select_annotations(
        manifest_rows,
        facing=facing,
        exclude_unclear=exclude_unclear,
    )

    n_selected = len(selected_rows)

    if n_selected == 0:
        raise ValueError(
            f"No annotated images match facing='{facing}'."
        )

    # ========================================================
    # GENERATE TOML
    # ========================================================

    config_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    landmarks_toml = toml_string_array(
        landmarks
    )

    content = f"""\
# ============================================================
# OWL UNIT - DEEPLABCUT TRAINING CONFIGURATION
# ============================================================
#
# Dataset: {dataset}
#
# Generated automatically by:
#
# owl-unit setup-training {dataset}
#
# NOTE:
# Selection settings must be applied by prepare_dlc_dataset.py.
#
# ============================================================


# ============================================================
# PROJECT
# ============================================================

[project]

name = "Owl Unit"
dataset = "{dataset}"


# ============================================================
# DATA PATHS
# ============================================================

[paths]

images = "data/raw_images/{dataset}"
annotations = "data/annotations/{dataset}"

manifest = "data/annotations/{dataset}/owl_manifest.csv"
keypoints = "data/annotations/{dataset}/owl_keypoints.csv"

dlc_project = "data/processed/dlc/{dataset}"
results = "results/training/{dataset}"


# ============================================================
# DATASET STATISTICS
# ============================================================

[dataset]

total_images = {n_images}
annotated_images = {stats["annotated"]}

selected_images = {n_selected}

pending_images = {stats["pending"]}
skipped_images = {stats["skipped"]}

left_images = {stats["left"]}
right_images = {stats["right"]}
front_images = {stats["front"]}
unclear_images = {stats["unclear"]}


# ============================================================
# TRAINING IMAGE SELECTION
# ============================================================

[selection]

# Options:
#   all
#   left
#   right
#   front
#   left_right

facing = "{facing}"

# Exclude photographs with unclear direction
exclude_unclear = {str(exclude_unclear).lower()}

# Standardise facing direction by horizontally flipping images
normalize_facing = {str(normalize_facing).lower()}

# Direction to standardise towards
target_facing = "{target_facing}"


# ============================================================
# DEEPLABCUT MODEL
# ============================================================

[model]

framework = "deeplabcut"
engine = "pytorch"

backbone = "resnet_50"


# ============================================================
# TRAINING PARAMETERS
# ============================================================

[training]

epochs = 200
batch_size = 8

train_fraction = 0.80
validation_fraction = 0.10
test_fraction = 0.10

random_seed = 42

device = "cuda"


# ============================================================
# FACIAL LANDMARKS
# ============================================================

[landmarks]

names = [
{landmarks_toml}
]
"""

    config_path.write_text(
        content,
        encoding="utf-8"
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("OWL UNIT - DEEPLABCUT TRAINING CONFIGURATION")
    print("=" * 65)

    print(f"Dataset:          {dataset}")
    print(f"Total images:     {n_images}")
    print(f"Annotated images: {stats['annotated']}")
    print(f"Selected images:  {n_selected}")

    print("\nFacing distribution:")

    print(f"  Left:    {stats['left']}")
    print(f"  Right:   {stats['right']}")
    print(f"  Front:   {stats['front']}")
    print(f"  Unclear: {stats['unclear']}")

    print("\nTraining selection:")

    print(f"  Facing:            {facing}")
    print(f"  Exclude unclear:   {exclude_unclear}")
    print(f"  Normalize facing:  {normalize_facing}")
    print(f"  Target facing:     {target_facing}")

    print(f"\nLandmarks: {len(landmarks)}")

    print(f"\nConfiguration saved to:\n  {config_path}")

    print("=" * 65 + "\n")

    return config_path


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Owl Unit DeepLabCut "
            "training configurations"
        )
    )

    parser.add_argument(
        "dataset",
        help="Dataset name, e.g. Siviglia"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing training configuration"
    )

    parser.add_argument(
        "--facing",
        choices=sorted(VALID_FACING),
        default="left_right",
        help="Facing directions to include in training"
    )

    parser.add_argument(
        "--normalize-facing",
        action="store_true",
        help="Standardise owl facing direction by mirroring"
    )

    parser.add_argument(
        "--target-facing",
        choices=["left", "right"],
        default="left",
        help="Target direction for facing normalisation"
    )

    parser.add_argument(
        "--include-unclear",
        action="store_true",
        help="Include images marked Unclear when facing=all"
    )

    args = parser.parse_args()

    generate_config(
        dataset=args.dataset,
        root=get_project_root(),
        overwrite=args.overwrite,
        facing=args.facing,
        normalize_facing=args.normalize_facing,
        target_facing=args.target_facing,
        exclude_unclear=not args.include_unclear,
    )


if __name__ == "__main__":
    main()