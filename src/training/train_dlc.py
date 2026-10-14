#!/usr/bin/env python3
"""Build DLC's train/test files and train PyTorch keypoint model.

Run after prepare_dlc_dataset.py:
    python src/training/train_dlc.py Siviglia --create-dataset
    python src/training/train_dlc.py Siviglia --train
"""
import argparse
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(dataset, config_path=None, create_dataset=False, train=False):
    if not dataset or dataset in ('.', '..') or '/' in dataset or '\\' in dataset:
        raise ValueError('Dataset must be a single folder name')
    cfg_file = Path(config_path or ROOT / 'configs' / 'training' / f'{dataset}.toml').resolve()
    with cfg_file.open('rb') as f:
        cfg = tomllib.load(f)
    if cfg['project']['dataset'] != dataset:
        raise ValueError('Dataset mismatch')
    folder = Path(cfg['paths']['dlc_project']).expanduser()
    if not folder.is_absolute():
        folder = ROOT / folder
    record = folder / 'owl_unit_dlc_project.json'
    if not record.is_file():
        raise FileNotFoundError('Run prepare_dlc_dataset.py first')
    config_yaml = Path(json.loads(record.read_text())['config_yaml'])
    if not config_yaml.is_file():
        raise FileNotFoundError(config_yaml)

    import deeplabcut
    engine = getattr(deeplabcut.Engine, 'PYTORCH', None)
    if engine is None:
        raise RuntimeError('Installed DeepLabCut does not expose Engine.PYTORCH; install DLC 3.x')

    if create_dataset:
        deeplabcut.create_training_dataset(
            str(config_yaml), num_shuffles=1,
            net_type=cfg.get('model', {}).get('backbone', 'resnet_50'),
            engine=engine,
            userfeedback=False,
        )
        print('[CREATED] DLC training dataset')

    if train:
        train_cfg = cfg.get('training', {})
        device = str(train_cfg.get('device', 'cpu'))
        if device.startswith('cuda'):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('TOML requested CUDA but torch.cuda.is_available() is False')
        deeplabcut.train_network(
            str(config_yaml), shuffle=1, device=device,
            epochs=int(train_cfg.get('epochs', 200)),
            batch_size=int(train_cfg.get('batch_size', 8)),
            save_epochs=int(train_cfg.get('save_epochs', 10)),
        )
        print('[TRAINED] DLC training completed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset')
    p.add_argument('--config', type=Path)
    p.add_argument('--create-dataset', action='store_true')
    p.add_argument('--train', action='store_true')
    args = p.parse_args()
    if not (args.create_dataset or args.train):
        p.error('Choose --create-dataset and/or --train')
    run(args.dataset, args.config, args.create_dataset, args.train)


if __name__ == '__main__':
    main()
