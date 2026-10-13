#!/usr/bin/env python3

"""Owl Unit command-line interface."""

import argparse
import tomllib
from pathlib import Path

import napari

from annotations.napari_annotate import launch


# ============================================================
# PATH UTILITIES
# ============================================================

def get_project_root():
    """Find the Owl Unit project directory."""

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise FileNotFoundError("Could not locate Owl Unit project root.")


def resolve_path(value, root):
    """Resolve relative paths against the project root."""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = root / path

    return path.resolve()


# ============================================================
# MAIN CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        prog="owl-unit",
        description="Owl Unit - Facial Landmark Annotation Pipeline"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # ========================================================
    # ANNOTATION
    # ========================================================

    annotate = subparsers.add_parser(
        "annotate",
        help="Launch napari for manual landmark annotation"
    )

    annotate.add_argument(
        "--dataset",
        nargs="?",
        default=None,
        help="Dataset name, e.g. Siviglia"
    )

    annotate.add_argument(
        "--config",
        type=Path,
        default=Path("owl_unit_setup.toml"),
        help="Path to TOML configuration file"
    )

    annotate.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override the input image directory"
    )

    annotate.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the annotation output directory"
    )

    annotate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to load"
    )

    annotate.add_argument(
        "--prompt",
        action="store_true",
        help="Interactively select input and output directories"
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    root = get_project_root()

    config_path = resolve_path(args.config, root)

    if not config_path.is_file():
        parser.error(f"Configuration not found: {config_path}")

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    # ========================================================
    # DATASET SELECTION
    # ========================================================

    if args.command == "annotate":

        if args.dataset:

            dataset_name = args.dataset

            if (
                dataset_name in (".", "..")
                or Path(dataset_name).name != dataset_name
                or "/" in dataset_name
                or "\\" in dataset_name
            ):
                parser.error(
                    "Dataset must be a single folder name."
                )

            input_dir = root / "data" / "raw_images" / dataset_name
            output_dir = root / "data" / "annotations" / dataset_name

        else:

            input_dir = config["paths"]["images_root"]

            output_dir = config["paths"].get(
                "output_dir",
                "data/annotations"
            )

        # ====================================================
        # COMMAND-LINE OVERRIDES
        # ====================================================

        if args.input_dir is not None:
            input_dir = args.input_dir

        if args.output_dir is not None:
            output_dir = args.output_dir

        # ====================================================
        # INTERACTIVE PROMPT
        # ====================================================

        if args.prompt:

            entered = input(
                f"Image directory [{input_dir}]: "
            ).strip()

            if entered:
                input_dir = entered

            entered = input(
                f"Output directory [{output_dir}]: "
            ).strip()

            if entered:
                output_dir = entered

        # ====================================================
        # RESOLVE AND VALIDATE PATHS
        # ====================================================

        input_dir = resolve_path(input_dir, root)
        output_dir = resolve_path(output_dir, root)

        if not input_dir.is_dir():
            parser.error(
                f"Image directory does not exist: {input_dir}"
            )

        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be greater than zero")

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        dataset_name = input_dir.name

        # ====================================================
        # SUMMARY
        # ====================================================

        print("\n" + "=" * 60)
        print("OWL UNIT - NAPARI ANNOTATION")
        print("=" * 60)

        print(f"Dataset:       {dataset_name}")
        print(f"Configuration: {config_path}")
        print(f"Input:         {input_dir}")
        print(f"Output:        {output_dir}")
        print(f"Image limit:   {args.limit or 'All'}")

        print("=" * 60 + "\n")

        # ====================================================
        # LAUNCH NAPARI
        # ====================================================

        launch(
            config_path=config_path,
            limit=args.limit,
            input_dir=input_dir,
            output_dir=output_dir
        )

        napari.run()


if __name__ == "__main__":
    main()