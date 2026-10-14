#!/usr/bin/env python3
"""Convert Owl Unit napari annotations into a DLC 3 single-animal project.

Run: python src/training/prepare_dlc_dataset.py Siviglia
"""
import argparse
import csv
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required annotation file missing: {path}")
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def make_reference_video(path):
    """DLC's project creator requires a video; make a small, synthetic placeholder.

    This is ONLY for registering the still-image dataset as a standard DLC project.
    It is not used as training data.
    """
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 1, (64, 64))
    if not writer.isOpened():
        raise RuntimeError('Unable to write temporary reference video with OpenCV (MJPG AVI).')
    try:
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError('Unable to create reference video')


def prepare(dataset, config_path=None):
    if not dataset or dataset in ('.','..') or '/' in dataset or '\\' in dataset:
        raise ValueError('Dataset must be a single folder name')
    config_path = Path(config_path or ROOT / 'configs' / 'training' / f'{dataset}.toml').expanduser().resolve()
    with config_path.open('rb') as f:
        conf = tomllib.load(f)
    if conf['project']['dataset'] != dataset:
        raise ValueError('Dataset argument and TOML dataset disagree')

    def relpath(value):
        p = Path(value).expanduser()
        return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()

    images_root = relpath(conf['paths']['images'])
    manifest_rows = read_csv(relpath(conf['paths']['manifest']))
    keypoint_rows = read_csv(relpath(conf['paths']['keypoints']))
    landmarks = conf['landmarks']['names']
    if len(landmarks) != 9 or len(set(landmarks)) != 9:
        raise ValueError('Expected nine unique landmarks')
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)
    if any('dataset' in r and r['dataset'] not in ('', dataset) for r in manifest_rows):
        raise ValueError('Manifest contains rows from another dataset')
    if any('dataset' in r and r['dataset'] not in ('', dataset) for r in keypoint_rows):
        raise ValueError('Keypoints contains rows from another dataset')

    selected = [r for r in manifest_rows if r.get('status', '').strip().lower() == 'annotated']
    if len(selected) < 3:
        raise ValueError(f'Only {len(selected)} annotated images. Need at least 3 to prepare a training dataset.')

    point_map = {}
    for r in keypoint_rows:
        key = (r['image_id'], r['landmark'])
        if key in point_map:
            raise ValueError(f'Duplicate landmark row: {key}')
        point_map[key] = r

    out_root = relpath(conf['paths']['dlc_project'])
    out_root.mkdir(parents=True, exist_ok=True)
    config_record = out_root / 'owl_unit_dlc_project.json'
    if config_record.exists():
        project_yaml = Path(json.loads(config_record.read_text())['config_yaml'])
        if not project_yaml.exists():
            raise FileNotFoundError(f'Previously created DLC project missing: {project_yaml}')
    else:
        import deeplabcut
        ref_video = out_root / '_owl_unit_reference.avi'
        make_reference_video(ref_video)
        project_yaml = deeplabcut.create_new_project(
            project=f'OwlUnit_{dataset}', experimenter='OwlUnit', videos=[str(ref_video)],
            working_directory=str(out_root), copy_videos=True,
        )
        if project_yaml == 'nothingcreated':
            raise RuntimeError('DeepLabCut did not create project')
        project_yaml = Path(project_yaml).resolve()
        config_record.write_text(json.dumps({'config_yaml': str(project_yaml)}, indent=2))

    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(str(project_yaml))
    cfg['bodyparts'] = list(landmarks)
    cfg['skeleton'] = []
    cfg['TrainingFraction'] = [float(conf.get('training', {}).get('train_fraction', 0.8))]
    cfg['default_net_type'] = str(conf.get('model', {}).get('backbone', 'resnet_50'))
    cfg['default_augmenter'] = 'default'
    auxiliaryfunctions.write_config(str(project_yaml), cfg)

    video_keys = list(cfg['video_sets'])
    if len(video_keys) != 1:
        raise RuntimeError('Expected one registered DLC reference video')
    folder_name = Path(video_keys[0]).stem
    labels_dir = Path(cfg['project_path']) / 'labeled-data' / folder_name
    labels_dir.mkdir(parents=True, exist_ok=True)

    scorer = cfg['scorer']
    cols = pd.MultiIndex.from_product([[scorer], landmarks, ['x','y']], names=['scorer','bodyparts','coords'])
    data, index, mapping = [], [], []
    for j, row in enumerate(selected):
        image_id = row['image_id']
        source = (images_root / image_id).resolve()
        if not source.is_file() or not source.is_relative_to(images_root):
            raise FileNotFoundError(f'Missing/invalid image: {source}')
        rotation = int(row.get('rotation_ccw_deg') or 0)
        if rotation not in (0,90,180,270):
            raise ValueError(f'Invalid rotation {rotation} for {image_id}')
        with Image.open(source) as im:
            original = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        rotated = np.ascontiguousarray(np.rot90(original, rotation // 90))
        height, width = rotated.shape[:2]
        # Distinct filenames are essential when source datasets include subdirectories.
        target_name = f'img_{j:06d}.png'
        Image.fromarray(rotated).save(labels_dir / target_name)
        values = []
        for landmark in landmarks:
            p = point_map.get((image_id, landmark))
            if p is None or p.get('visibility', '').lower() != 'visible':
                values.extend([np.nan, np.nan])
                continue
            x, y = float(p['x_oriented']), float(p['y_oriented'])
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError(f'Out-of-bounds landmark: {image_id} {landmark} ({x},{y}) vs {width}x{height}')
            values.extend([x,y])
        if not any(np.isfinite(v) for v in values):
            raise ValueError(f'No visible keypoints: {image_id}')
        data.append(values)
        index.append(f'labeled-data/{folder_name}/{target_name}')
        mapping.append({'image_id': image_id, 'dlc_image': index[-1], 'facing': row.get('facing',''), 'rotation_ccw_deg': rotation})

    df = pd.DataFrame(data, index=index, columns=cols, dtype=float)
    h5 = labels_dir / f'CollectedData_{scorer}.h5'
    df.to_hdf(h5, key='df_with_missing', mode='w')
    df.to_csv(labels_dir / f'CollectedData_{scorer}.csv')
    pd.DataFrame(mapping).to_csv(out_root / 'source_image_mapping.csv', index=False)
    print(f'[PREPARED] {len(df)} images, {len(landmarks)} keypoints -> {h5}')
    print(f'[DLC CONFIG] {project_yaml}')
    return project_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--config', type=Path, help='Dataset training TOML (defaults to configs/training/<dataset>.toml)')
    args = parser.parse_args()
    prepare(args.dataset, args.config)


if __name__ == '__main__':
    main()
