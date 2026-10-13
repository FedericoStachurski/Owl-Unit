#!/usr/bin/env python3
"""Owl Unit: manual 9-point owl-face annotation in napari.

Run from ~/Owl-Unit:
    python run.py napari
    python run.py napari --limit 5

Or from a Jupyter notebook after `%gui qt`, using importlib as
demonstrated in the supplied notebook.
"""
from __future__ import annotations

import argparse
import csv
import os
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FIELDS = [
    "dataset", "image_id", "image_path", "status", "facing", "rotation_ccw_deg",
    "original_width", "original_height", "oriented_width", "oriented_height",
    "visible_points", "missing_points", "corrected_image_path", "updated_at_utc",
]
KEYPOINT_FIELDS = [
    "image_id", "landmark", "visibility", "x_oriented", "y_oriented",
    "x_original", "y_original",
]
SUPPORTED_DIRECTIONS = ("Left", "Right", "Front", "Unclear")


def image_to_rotated(y: float, x: float, original_h: int, original_w: int, k: int):
    """Map EXIF-normalized original (y,x) to np.rot90(..., k) pixel coordinates."""
    k %= 4
    if k == 0:
        return y, x
    if k == 1:
        return original_w - 1 - x, y
    if k == 2:
        return original_h - 1 - y, original_w - 1 - x
    return x, original_h - 1 - y


def rotated_to_image(y: float, x: float, original_h: int, original_w: int, k: int):
    """Inverse of image_to_rotated."""
    k %= 4
    if k == 0:
        return y, x
    if k == 1:
        return x, original_w - 1 - y
    if k == 2:
        return original_h - 1 - y, original_w - 1 - x
    return original_h - 1 - x, y


def load_config(config_path: Path | None = None):
    path = Path(config_path) if config_path else ROOT / "setup.toml"
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    section = config.get("napari_annotation", {})
    landmarks = section.get("landmarks", [])
    if not landmarks or len(landmarks) != 9 or len(set(landmarks)) != 9:
        raise ValueError("[napari_annotation].landmarks must contain 9 unique names (8 face points + beak).")

    def abs_path(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else (path.parent / p).resolve()

    return {
        "images_root": abs_path(config["paths"]["images_root"]),
        "extensions": {ext.casefold() for ext in config.get("images", {}).get("extensions", [".jpg", ".jpeg", ".png"])},
        "recursive": bool(config.get("images", {}).get("recursive", True)),
        "manifest_path": abs_path(section.get("manifest_csv", "data/annotations/owl_manifest.csv")),
        "keypoints_path": abs_path(section.get("keypoints_csv", "data/annotations/owl_keypoints.csv")),
        "corrected_dir": abs_path(section.get("corrected_dir", "data/processed/upright_images")),
        "save_corrected": bool(section.get("save_corrected_images", False)),
        "landmarks": landmarks,
        "config_path": path,
        "repo_root": path.parent,
    }


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_write_csv(path: Path, columns: list[str], rows: list[dict]):
    """One-file atomic save (minimise risk from an interrupted GUI)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class AnnotationStorage:
    def __init__(self, settings: dict, limit: int | None = None):
        self.config = settings
        root = settings["images_root"]
        if not root.exists():
            raise FileNotFoundError(f"Owl photographs not found at {root}")
        paths = root.rglob("*") if settings["recursive"] else root.glob("*")
        all_images = sorted((p for p in paths if p.is_file() and p.suffix.casefold() in settings["extensions"]), key=lambda p: str(p).casefold())
        if limit is not None:
            if limit < 1:
                raise ValueError("--limit must be >= 1")
            all_images = all_images[:limit]
        if not all_images:
            raise RuntimeError(f"No images found in {root}")
        self.files = all_images
        self.ids = [p.relative_to(root).as_posix() for p in self.files]
        # The wide manifest also holds oriented coordinates for all nine points.
        self.manifest_fields = MANIFEST_FIELDS + [
            f"{name}_{suffix}"
            for name in settings["landmarks"]
            for suffix in ("x", "y", "visibility")
        ]
        self.manifest = {row["image_id"]: row for row in read_csv_rows(settings["manifest_path"])}
        self.keypoints = read_csv_rows(settings["keypoints_path"])
        # Do not erase rows previously annotated outside a --limit pilot.
        for image_id in self.ids:
            if image_id not in self.manifest:
                self.manifest[image_id] = self.new_manifest_row(image_id)
        self.persist_manifest()

    def new_manifest_row(self, image_id: str):
        try:
            path_rel = (self.config["images_root"] / image_id).relative_to(self.config["repo_root"]).as_posix()
        except ValueError:
            path_rel = str(self.config["images_root"] / image_id)
        row = {
            "image_id": image_id, "image_path": path_rel, "status": "pending",
            "facing": "", "rotation_ccw_deg": "0", "original_width": "",
            "original_height": "", "oriented_width": "", "oriented_height": "",
            "visible_points": "0", "missing_points": "0", "corrected_image_path": "",
            "updated_at_utc": "",
        }
        for name in self.config["landmarks"]:
            for suffix in ("x", "y", "visibility"):
                row[f"{name}_{suffix}"] = ""
        return row

    def persist_manifest(self):
        atomic_write_csv(self.config["manifest_path"], self.manifest_fields, [self.manifest[k] for k in sorted(self.manifest)])

    def get_keypoints(self, image_id):
        return {r["landmark"]: r for r in self.keypoints if r["image_id"] == image_id}

    def save(self, image_id: str, entry: dict, keypoints_for_image: list[dict] | None):
        if keypoints_for_image is not None:
            self.keypoints = [r for r in self.keypoints if r["image_id"] != image_id]
            self.keypoints.extend(keypoints_for_image)
            # Write points FIRST: a completed manifest should not point to unsaved points.
            atomic_write_csv(self.config["keypoints_path"], KEYPOINT_FIELDS, self.keypoints)
        self.manifest[image_id] = entry
        self.persist_manifest()


def launch(config_path=None, limit=None, input_dir=None, output_dir=None,):
    """Start the napari UI. In notebooks, call after `%gui qt`. Returns (viewer, widget)."""
    import napari
    from qtpy.QtCore import Qt, QSize, QTimer
    from qtpy.QtGui import QIcon, QPixmap, QImage, QColor, QPainter, QPen, QFont
    from qtpy.QtWidgets import (
        QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
        QPushButton, QVBoxLayout, QWidget, QScrollArea, QFrame, QSizePolicy,
    )

    settings = load_config(
        Path(config_path) if config_path else None
    )

    if input_dir is not None:
        settings["images_root"] = Path(input_dir).expanduser().resolve()

    if output_dir is not None:

        output_dir = Path(output_dir).expanduser().resolve()

        settings["manifest_path"] = output_dir / "owl_manifest.csv"
        settings["keypoints_path"] = output_dir / "owl_keypoints.csv"

    settings["dataset"] = settings["images_root"].name

    settings["corrected_dir"] = (
        settings["repo_root"]
        / "data"
        / "processed"
        / "upright_images"
        / settings["dataset"]
    )

    storage = AnnotationStorage(settings, limit)
    viewer = napari.Viewer(title="Owl Unit — Manual face annotations")

    class OwlAnnotationWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.store = storage
            self.conf = settings
            self.files = storage.files
            self.ids = storage.ids
            self.landmarks = settings["landmarks"]
            self.current_idx = 0
            self.current_k = 0
            self.original = np.zeros((64, 64, 3), dtype=np.uint8)
            self.points = {name: None for name in self.landmarks}  # None = not yet answered; 'missing' = occluded
            self.active_idx = 0
            self.dirty = False
            self.refreshing = False

            outer = QVBoxLayout(self)
            self.counter = QLabel()
            self.counter.setWordWrap(True)
            outer.addWidget(self.counter)
            self.file_label = QLabel()
            self.file_label.setWordWrap(True)
            outer.addWidget(self.file_label)

            outer.addWidget(QLabel("Facing direction (independent of image rotation):"))
            self.facing_box = QComboBox()
            self.facing_box.addItems(["— Select facing —", *SUPPORTED_DIRECTIONS])
            self.facing_box.currentTextChanged.connect(self.on_facing_changed)
            outer.addWidget(self.facing_box)

            turn_row = QHBoxLayout()
            self.turn_ccw = QPushButton("↺ Rotate left 90°")
            self.turn_cw = QPushButton("↻ Rotate right 90°")
            self.turn_ccw.clicked.connect(lambda: self.rotate(+1))
            self.turn_cw.clicked.connect(lambda: self.rotate(-1))
            turn_row.addWidget(self.turn_ccw)
            turn_row.addWidget(self.turn_cw)
            outer.addLayout(turn_row)
            self.rotation_label = QLabel("Rotation: 0° CCW")
            outer.addWidget(self.rotation_label)

            outer.addWidget(QLabel("Nine landmarks (select any item to place/correct it):"))
            self.landmark_list = QListWidget()
            self.landmark_list.currentRowChanged.connect(self.select_landmark)
            outer.addWidget(self.landmark_list)
            self.instruction = QLabel()
            self.instruction.setWordWrap(True)
            outer.addWidget(self.instruction)

            actions = QHBoxLayout()
            self.missing_button = QPushButton("Not visible / occluded")
            self.clear_button = QPushButton("Clear selected point")
            self.missing_button.clicked.connect(self.mark_missing)
            self.clear_button.clicked.connect(self.clear_point)
            actions.addWidget(self.missing_button)
            actions.addWidget(self.clear_button)
            outer.addLayout(actions)

            self.corrected_box = QCheckBox("Save oriented image copy on save")
            self.corrected_box.setChecked(settings["save_corrected"])
            outer.addWidget(self.corrected_box)
            self.save_button = QPushButton("💾 Save + next unlabelled")
            self.save_button.clicked.connect(self.save_next)
            outer.addWidget(self.save_button)
            skip_row = QHBoxLayout()
            self.previous_button = QPushButton("← Previous")
            self.next_button = QPushButton("Next →")
            self.skip_button = QPushButton("Skip image")
            self.previous_button.clicked.connect(lambda: self.navigate(-1))
            self.next_button.clicked.connect(lambda: self.navigate(+1))
            self.skip_button.clicked.connect(self.skip_image)
            skip_row.addWidget(self.previous_button)
            skip_row.addWidget(self.next_button)
            skip_row.addWidget(self.skip_button)
            outer.addLayout(skip_row)
            self.help_label = QLabel("Click on the owl image to place the selected point.\n"
                "The next unmarked point is selected automatically.\n"
                "Scroll to zoom; drag with Space held to pan.\n"
                "Rotation does NOT change whether you select Left or Right.\n"
                "Changes are saved when you press Save + next.")
            self.help_label.setWordWrap(True)
            outer.addWidget(self.help_label)

            self.thumb_buttons = []
            self.thumb_loaded = set()
            self.build_filmstrip()

            self.image_layer = viewer.add_image(
                self.original, rgb=True, name="Owl image (CLICK HERE)")
            self.points_layer = viewer.add_points(
                np.empty((0, 2), dtype=float), ndim=2, name="Facial landmarks",
                features=pd.DataFrame({"label": pd.Series(dtype=str)}),
                text={"string": "{label}", "size": 8, "color": "yellow", "translation": [7, -7]},
                size=32, symbol="disc", face_color="red", border_color="white",
            )
            self.points_layer.editable = False
            self.image_layer.mouse_drag_callbacks.append(self.on_image_click)
            self.load_index(self.first_pending_index())
            QTimer.singleShot(0, self.populate_visible_thumbnails)

        def build_filmstrip(self):
            """Thumbnail navigation dock along the bottom of the napari viewer."""
            self.filmstrip = QWidget()
            layout = QVBoxLayout(self.filmstrip)
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(3)
            legend = QLabel("IMAGE BROWSER     ● Annotated     ○ Pending     ✕ Skipped     (click to revisit)")
            layout.addWidget(legend)
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.scroll.setFixedHeight(147)
            self.scroll.setFrameShape(QFrame.Shape.NoFrame)
            inner = QWidget()
            row = QHBoxLayout(inner)
            row.setContentsMargins(3, 3, 3, 3)
            row.setSpacing(6)
            for idx, image_id in enumerate(self.ids):
                btn = QPushButton()
                btn.setFixedSize(120, 111)
                btn.setIconSize(QSize(112, 77))
                btn.setToolTip(image_id)
                btn.clicked.connect(lambda checked=False, idx=idx: self.jump_to(idx))
                row.addWidget(btn)
                self.thumb_buttons.append(btn)
                self.set_thumb_status(idx)
            row.addStretch()
            self.scroll.setWidget(inner)
            self.scroll.horizontalScrollBar().valueChanged.connect(lambda _: self.populate_visible_thumbnails())
            layout.addWidget(self.scroll)
            viewer.window.add_dock_widget(self.filmstrip, name="Owl images / annotation progress", area="bottom")

        def set_thumb_status(self, idx):
            status = self.store.manifest[self.ids[idx]].get("status", "pending")
            marker = {"annotated": "●", "skipped": "✕", "pending": "○"}.get(status, "○")
            color = {"annotated": "#4fca91", "skipped": "#e8a46e", "pending": "#b9c0c8"}.get(status, "#b9c0c8")
            current = idx == self.current_idx
            border = "2px solid #ffda75" if current else "1px solid #56606d"
            points = self.store.get_keypoints(self.ids[idx])
            visible = [name for name, row in points.items() if row.get("visibility") == "visible"]
            self.thumb_buttons[idx].setToolTip(
                f"{self.ids[idx]}\nStatus: {status}\nFacing: {self.store.manifest[self.ids[idx]].get('facing', '')}"
                f"\nSaved landmarks ({len(visible)}): {', '.join(visible)}"
            )
            self.thumb_buttons[idx].setText(f"{idx+1} {marker} {status[:4]}")
            self.thumb_buttons[idx].setStyleSheet(
                f"QPushButton {{color:{color}; border:{border}; text-align:bottom; padding-bottom:2px; background:#252a31;}}"
            )

        def thumbnail_pixmap(self, idx):
            """Generate a small upright thumbnail with saved keypoint markers."""
            image_id = self.ids[idx]
            with Image.open(self.files[idx]) as file:
                im = ImageOps.exif_transpose(file).convert("RGB")
            original_w, original_h = im.size
            k = (int(self.store.manifest[image_id].get("rotation_ccw_deg") or 0) // 90) % 4
            if k:
                im = im.rotate(90 * k, expand=True)
            full_w, full_h = im.size
            im.thumbnail((110, 75))
            pix = QPixmap.fromImage(QImage(im.tobytes(), im.width, im.height, im.width * 3, QImage.Format.Format_RGB888).copy())
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            sx, sy = im.width / full_w, im.height / full_h
            saved = self.store.get_keypoints(image_id)
            for name, entry in saved.items():
                if entry.get("visibility") != "visible":
                    continue
                try:
                    x, y = float(entry["x_oriented"]) * sx, float(entry["y_oriented"]) * sy
                except (ValueError, KeyError, TypeError):
                    continue
                painter.setPen(QPen(QColor("white"), 1))
                painter.setBrush(QColor("#f04f52"))
                painter.drawEllipse(int(x) - 2, int(y) - 2, 5, 5)
            painter.end()
            return pix

        def populate_visible_thumbnails(self):
            """Load thumbnails only around the visible part of the scrollbar."""
            if not self.thumb_buttons:
                return
            left = self.scroll.horizontalScrollBar().value()
            width = max(1, self.scroll.viewport().width())
            first = max(0, left // 126 - 2)
            last = min(len(self.ids), (left + width) // 126 + 4)
            for idx in range(first, last):
                if idx not in self.thumb_loaded:
                    try:
                        self.thumb_buttons[idx].setIcon(QIcon(self.thumbnail_pixmap(idx)))
                    except (OSError, ValueError) as exc:
                        self.thumb_buttons[idx].setToolTip(f"{self.ids[idx]}\nThumbnail error: {exc}")
                    self.thumb_loaded.add(idx)

        def refresh_thumbnail(self, idx):
            self.thumb_loaded.discard(idx)
            self.set_thumb_status(idx)
            self.populate_visible_thumbnails()

        def jump_to(self, idx):
            if idx == self.current_idx:
                return
            if self.maybe_discard_edits():
                self.load_index(idx)

        def center_current_thumbnail(self):
            btn = self.thumb_buttons[self.current_idx]
            self.scroll.ensureWidgetVisible(btn, 15, 0)
            for idx in range(len(self.thumb_buttons)):
                self.set_thumb_status(idx)
            self.populate_visible_thumbnails()

        def first_pending_index(self):
            for i, image_id in enumerate(self.ids):
                if self.store.manifest[image_id].get("status", "pending") == "pending":
                    return i
            return 0

        def load_index(self, i):
            self.current_idx = i
            path = self.files[i]
            image_id = self.ids[i]
            with Image.open(path) as file:
                # EXIF orientation is applied explicitly, before manual rotations.
                self.original = np.asarray(ImageOps.exif_transpose(file).convert("RGB"))
            h, w = self.original.shape[:2]
            info = self.store.manifest[image_id]
            self.current_k = (int(info.get("rotation_ccw_deg") or 0) // 90) % 4
            self.points = {name: None for name in self.landmarks}
            saved = self.store.get_keypoints(image_id)
            for name in self.landmarks:
                entry = saved.get(name)
                if not entry:
                    continue
                if entry.get("visibility") == "missing":
                    self.points[name] = "missing"
                elif entry.get("visibility") == "visible":
                    # Store in ORIGINAL (EXIF-normalized) coordinates internally.
                    self.points[name] = (float(entry["y_original"]), float(entry["x_original"]))

            direction = info.get("facing", "")
            self.facing_box.blockSignals(True)
            self.facing_box.setCurrentText(direction if direction in SUPPORTED_DIRECTIONS else "— Select facing —")
            self.facing_box.blockSignals(False)
            self.image_layer.data = np.ascontiguousarray(np.rot90(self.original, self.current_k))
            self.image_layer.name = "Owl image (CLICK HERE)"
            self.active_idx = next((j for j, name in enumerate(self.landmarks) if self.points[name] is None), 0)
            self.dirty = False
            self.render()
            viewer.reset_view()
            viewer.layers.selection.active = self.image_layer
            self.center_current_thumbnail()

        def on_image_click(self, layer, event):
            if str(getattr(event, "button", "1")).lower() not in ("1", "left"):
                return
            if any("space" in str(mod).casefold() for mod in getattr(event, "modifiers", [])):
                return
            y, x = np.asarray(layer.world_to_data(event.position), dtype=float)[-2:]
            h, w = layer.data.shape[:2]
            if not (0 <= y < h and 0 <= x < w):
                return
            original_h, original_w = self.original.shape[:2]
            orig_y, orig_x = rotated_to_image(float(y), float(x), original_h, original_w, self.current_k)
            name = self.landmarks[self.active_idx]
            self.points[name] = (orig_y, orig_x)
            self.dirty = True
            self.advance_after_mark()
            self.render()

        def advance_after_mark(self):
            n = len(self.landmarks)
            next_idx = next((i for i in range(self.active_idx + 1, n) if self.points[self.landmarks[i]] is None), None)
            if next_idx is None:
                next_idx = next((i for i in range(0, n) if self.points[self.landmarks[i]] is None), self.active_idx)
            self.active_idx = next_idx

        def select_landmark(self, row):
            if not self.refreshing and row >= 0:
                self.active_idx = row
                self.instruction.setText(f"Click to place: {self.landmarks[row]}")
                viewer.layers.selection.active = self.image_layer

        def mark_missing(self):
            self.points[self.landmarks[self.active_idx]] = "missing"
            self.dirty = True
            self.advance_after_mark()
            self.render()

        def clear_point(self):
            self.points[self.landmarks[self.active_idx]] = None
            self.dirty = True
            self.render()

        def rotate(self, change: int):
            self.current_k = (self.current_k + change) % 4
            self.image_layer.data = np.ascontiguousarray(np.rot90(self.original, self.current_k))
            # Stored points are in unrotated coordinates, so they rotate with the image.
            self.dirty = True
            self.render()
            viewer.reset_view()
            viewer.layers.selection.active = self.image_layer

        def on_facing_changed(self, _):
            self.dirty = True
            self.render()

        def render(self):
            original_h, original_w = self.original.shape[:2]
            coords, names = [], []
            for name in self.landmarks:
                val = self.points[name]
                if isinstance(val, tuple):
                    coords.append(image_to_rotated(*val, original_h, original_w, self.current_k))
                    names.append(name)
            self.points_layer.data = np.asarray(coords, dtype=float).reshape(-1, 2)
            self.points_layer.features = pd.DataFrame({"label": names})
            self.refreshing = True
            self.landmark_list.clear()
            for name in self.landmarks:
                status = self.points[name]
                prefix = "✓" if isinstance(status, tuple) else "×" if status == "missing" else "○"
                self.landmark_list.addItem(f"{prefix}  {name}")
            self.landmark_list.setCurrentRow(self.active_idx)
            self.refreshing = False
            count_vis = sum(isinstance(v, tuple) for v in self.points.values())
            count_missing = sum(v == "missing" for v in self.points.values() if isinstance(v, str))
            done = sum(self.store.manifest[image_id].get("status") == "annotated" for image_id in self.ids)
            skipped = sum(self.store.manifest[image_id].get("status") == "skipped" for image_id in self.ids)
            self.counter.setText(f"Image {self.current_idx + 1}/{len(self.ids)}   |   "
                f"Saved: {done} annotated, {skipped} skipped   |   "
                f"This image: {count_vis} points + {count_missing} missing / {len(self.landmarks)}")
            self.file_label.setText(f"{self.ids[self.current_idx]}  {'[unsaved edits]' if self.dirty else ''}")
            self.rotation_label.setText(f"Rotation: {(90 * self.current_k) % 360}° CCW (non-destructive)")
            self.instruction.setText(f"Current point: {self.landmarks[self.active_idx]}")
            viewer.layers.selection.active = self.image_layer

        def maybe_discard_edits(self):
            if not self.dirty:
                return True
            return QMessageBox.question(
                self, "Unsaved annotations", "Discard unsaved changes to this image?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes

        def navigate(self, delta: int):
            i = self.current_idx + delta
            if not (0 <= i < len(self.ids)):
                return
            if self.maybe_discard_edits():
                self.load_index(i)

        def write_corrected(self, image_id):
            rel = Path(image_id).with_suffix(".png")
            target = self.conf["corrected_dir"] / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            # EXIF-normalized + explicitly rotated, so no hidden EXIF transform remains.
            arr = np.ascontiguousarray(np.rot90(self.original, self.current_k))
            Image.fromarray(arr).save(target)
            try:
                return target.relative_to(self.conf["repo_root"]).as_posix()
            except ValueError:
                return str(target)

        def base_entry(self):
            image_id = self.ids[self.current_idx]
            entry = dict(self.store.manifest[image_id])
            h, w = self.original.shape[:2]
            oriented_h, oriented_w = np.rot90(self.original, self.current_k).shape[:2]
            direction = self.facing_box.currentText()
            return entry | {
                "facing": direction if direction in SUPPORTED_DIRECTIONS else "",
                "rotation_ccw_deg": str(90 * self.current_k),
                "original_width": str(w), "original_height": str(h),
                "oriented_width": str(oriented_w), "oriented_height": str(oriented_h),
                "visible_points": str(sum(isinstance(v, tuple) for v in self.points.values())),
                "missing_points": str(sum(v == "missing" for v in self.points.values() if isinstance(v, str))),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

        def save_next(self):
            direction = self.facing_box.currentText()
            if direction not in SUPPORTED_DIRECTIONS:
                QMessageBox.warning(self, "Facing needed", "Choose Left, Right, Front, or Unclear before saving.")
                return
            unmarked = [name for name, val in self.points.items() if val is None]
            if unmarked:
                QMessageBox.warning(self, "Unmarked landmarks", "Click or mark Not visible for every landmark.\n" + ", ".join(unmarked))
                return
            if not any(isinstance(v, tuple) for v in self.points.values()):
                QMessageBox.warning(self, "No visible points", "Place at least one point, or Skip this image.")
                return
            image_id = self.ids[self.current_idx]
            h, w = self.original.shape[:2]
            rows = []
            for name in self.landmarks:
                val = self.points[name]
                if val == "missing":
                    rows.append({"image_id": image_id, "landmark": name, "visibility": "missing",
                        "x_oriented": "", "y_oriented": "", "x_original": "", "y_original": ""})
                else:
                    orig_y, orig_x = val
                    y, x = image_to_rotated(orig_y, orig_x, h, w, self.current_k)
                    rows.append({"image_id": image_id, "landmark": name, "visibility": "visible",
                        "x_oriented": round(x, 3), "y_oriented": round(y, 3),
                        "x_original": round(orig_x, 3), "y_original": round(orig_y, 3)})
            try:
                entry = self.base_entry()
                entry["status"] = "annotated"
                for point in rows:
                    name = point["landmark"]
                    entry[f"{name}_x"] = point["x_oriented"]
                    entry[f"{name}_y"] = point["y_oriented"]
                    entry[f"{name}_visibility"] = point["visibility"]
                entry["corrected_image_path"] = self.write_corrected(image_id) if self.corrected_box.isChecked() else ""
                self.store.save(image_id, entry, rows)
                self.refresh_thumbnail(self.current_idx)
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return
            self.dirty = False
            target = next((i for i in list(range(self.current_idx + 1, len(self.ids))) + list(range(0, self.current_idx))
                 if self.store.manifest[self.ids[i]].get("status") == "pending"), None)
            if target is None:
                self.render()
                QMessageBox.information(self, "Finished", "No pending images in this selection. Use Previous/Next to review or edit saved images.")
            else:
                self.load_index(target)

        def skip_image(self):
            image_id = self.ids[self.current_idx]
            if self.store.manifest[image_id].get("status") == "annotated":
                answer = QMessageBox.question(self, "Skip annotated image?",
                    "This will mark the image skipped and REMOVE its saved keypoints. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    return
            try:
                entry = self.base_entry()
                entry["status"] = "skipped"
                entry["corrected_image_path"] = ""
                entry["visible_points"] = "0"
                entry["missing_points"] = "0"
                for name in self.landmarks:
                    for suffix in ("x", "y", "visibility"):
                        entry[f"{name}_{suffix}"] = ""
                self.store.save(image_id, entry, [])
                self.refresh_thumbnail(self.current_idx)
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return
            self.dirty = False
            target = next((i for i in list(range(self.current_idx + 1, len(self.ids))) + list(range(0, self.current_idx))
                if self.store.manifest[self.ids[i]].get("status") == "pending"), None)
            if target is None:
                self.render()
                QMessageBox.information(self, "Finished", "There are no remaining pending images.")
            else:
                self.load_index(target)

    widget = OwlAnnotationWidget()
    viewer.window.add_dock_widget(widget, name="Owl Unit annotation", area="right")
    return viewer, widget


def main():
    p = argparse.ArgumentParser(description="Annotate owl facial landmarks in napari")
    p.add_argument("--limit", type=int, default=None, help="Restrict pilot to first N images")
    p.add_argument("--config", type=Path, default=None)

    args = p.parse_args()
    import napari
    launch(config_path=args.config, limit=args.limit)
    napari.run()


if __name__ == "__main__":
    main()
