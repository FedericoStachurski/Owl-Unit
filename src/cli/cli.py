"""Owl Unit CLI: annotation, dataset setup, preparation, and DeepLabCut training."""

import argparse
import tomllib
from pathlib import Path


def get_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Cannot locate Owl-Unit/pyproject.toml")


def resolve_path(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def validate_dataset(parser, name):
    if not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
        parser.error("Dataset must be one folder name, e.g. Siviglia")


def main():
    parser = argparse.ArgumentParser(prog="owl-unit", description="Owl facial landmark annotation and DeepLabCut training")
    sub = parser.add_subparsers(dest="command", required=True)

    annotate = sub.add_parser("annotate", help="Open manual annotation interface in napari")
    annotate.add_argument("dataset", nargs="?", help="Dataset folder, e.g. Siviglia")
    annotate.add_argument("--config", type=Path, default=Path("owl_unit_setup.toml"))
    annotate.add_argument("--input-dir", type=Path)
    annotate.add_argument("--output-dir", type=Path)
    annotate.add_argument("--limit", type=int)
    annotate.add_argument("--prompt", action="store_true")

    setup_training = sub.add_parser("setup-training", help="Generate dataset-specific training TOML")
    setup_training.add_argument("dataset")
    setup_training.add_argument("--overwrite", action="store_true")

    prepare = sub.add_parser("prepare", help="Convert saved napari points to DeepLabCut format")
    prepare.add_argument("dataset")
    prepare.add_argument("--config", type=Path, help="Training TOML (default: configs/training/DATASET.toml)")

    train = sub.add_parser("train", help="Create DLC dataset and/or train DeepLabCut")
    train.add_argument("dataset")
    train.add_argument("--config", type=Path, help="Training TOML (default: configs/training/DATASET.toml)")
    train.add_argument("--create-dataset", action="store_true", help="Build DLC training/test files")
    train.add_argument("--train", action="store_true", help="Fit network using existing DLC training files")

    args = parser.parse_args()
    root = get_project_root()
    if getattr(args, "dataset", None) is not None:
        validate_dataset(parser, args.dataset)

    if args.command == "setup-training":
        from training.create_training_config import generate_config
        generate_config(dataset=args.dataset, root=root, overwrite=args.overwrite)
        return

    if args.command == "prepare":
        from training.prepare_dlc_dataset import prepare as prepare_dataset
        config_path = resolve_path(args.config, root) if args.config else None
        prepare_dataset(args.dataset, config_path=config_path)
        return

    if args.command == "train":
        from training.train_dlc import run as train_dlc
        if not (args.create_dataset or args.train):
            parser.error("train needs --create-dataset and/or --train")
        config_path = resolve_path(args.config, root) if args.config else None
        train_dlc(args.dataset, config_path=config_path,
                  create_dataset=args.create_dataset, train=args.train)
        return

    # Only annotation needs napari, so don't import it for setup/preparation/training.
    config_path = resolve_path(args.config, root)
    if not config_path.is_file():
        parser.error(f"Configuration not found: {config_path}")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    if args.dataset:
        input_dir = root / "data" / "raw_images" / args.dataset
        output_dir = root / "data" / "annotations" / args.dataset
    else:
        input_dir = config["paths"]["images_root"]
        output_dir = config["paths"].get("output_dir", "data/annotations")

    if args.input_dir is not None:
        input_dir = args.input_dir
    if args.output_dir is not None:
        output_dir = args.output_dir
    if args.prompt:
        input_dir = input(f"Image directory [{input_dir}]: ").strip() or input_dir
        output_dir = input(f"Output directory [{output_dir}]: ").strip() or output_dir

    input_dir = resolve_path(input_dir, root)
    output_dir = resolve_path(output_dir, root)
    if not input_dir.is_dir():
        parser.error(f"Image folder not found: {input_dir}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OWL UNIT] Dataset: {input_dir.name}\n[OWL UNIT] Images: {input_dir}\n"
          f"[OWL UNIT] Annotation output: {output_dir}\n[OWL UNIT] Limit: {args.limit or 'All'}")
    from annotations.napari_annotate import launch
    import napari
    launch(config_path=config_path, limit=args.limit,
           input_dir=input_dir, output_dir=output_dir)
    napari.run()


if __name__ == "__main__":
    main()