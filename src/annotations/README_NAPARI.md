# Owl Unit — Manual Owl Facial Annotation in napari

This setup does **not** use OWLv2. You manually classify orientation and mark facial points while inspecting each image.

## Requirements

Work from the Linux/WSL terminal inside `~/Owl-Unit`:

```bash
conda activate owl-unit
python -m pip install "napari[pyqt6, optional]" pandas pillow
python run.py napari --limit 5
```

If you prefer Jupyter, open `notebooks/01_manual_napari_annotation.ipynb` instead. It uses `%gui qt` to display the full napari window outside the notebook. Your WSL setup must support GUI windows (e.g. WSLg).

## What you see

The main viewer displays one owl image at a time. The **Owl Unit annotation** panel on the right lets you:

1. Choose **Left / Right / Front / Unclear** (how the *owl's face points* in the upright image; not the current rotation angle).
2. Rotate **left or right in 90° steps** until the owl appears upright. The original image file is never modified. Any already-clicked points rotate with the picture.
3. Mark **8 face landmarks + beak tip**, one click at a time. The currently selected point is highlighted in the numbered list and the list advances after every click. To correct a point, select its name and click the right location. You can zoom by scrolling and pan using Space+drag.
4. Use **Not visible / occluded** for landmarks that cannot be identified in profile or obstructed images; **Clear selected** to remove an incorrect annotation.
5. Click **Save + next unlabelled** to save the face direction, rotation, 9 landmark slots and coordinates. The app moves to the next pending photograph. **Next** and **Previous** let you review or edit earlier images. **Skip image** marks an unusable image.

Do a **five-image pilot** before the full dataset. Once satisfied run `python run.py napari` to work through all 669 pictures. Completed and skipped photos are retained in the manifest, and the app resumes at the first pending image.

## Landmark definitions — decide before bulk labelling

The supplied `setup.toml` has placeholder names `face_01` to `face_08`, plus `beak_tip`. Precisely define these landmarks and the order before annotating the whole dataset (for example, a clockwise scheme with a fixed starting point). Avoid changing the scheme after saving annotations; this would make old and new rows inconsistent.

The buttons **Left / Right / Front / Unclear** describe the direction of the owl's gaze/head relative to the *upright displayed image*. They do not mean rotate the picture or refer to the owl's own anatomical left/right. For ambiguous images choose Unclear.

## Outputs

- `data/annotations/owl_manifest.csv` — **one row per photograph**, initially `pending`, updated when Save or Skip is pressed. Includes image ID/path, facing direction, rotation (CCW degrees), image dimensions, **all 9 landmarks in wide `face_01_x`, `face_01_y`, `face_01_visibility` columns**, annotation counts, and optional oriented-copy path.
- `data/annotations/owl_keypoints.csv` — **nine rows per annotated photo**. Each row is one landmark with `visibility` (`visible` / `missing`) and coordinates.
  - `x_oriented`, `y_oriented` apply to the image shown in napari after rotation.
  - `x_original`, `y_original` apply to the image after standard `PIL.ImageOps.exif_transpose` (before our manual rotation). If using an image library that ignores EXIF, normalize it the same way when training.
  - All coordinates are pixel positions, with the origin at the top-left. `x` is horizontal, `y` vertical.

The original `data/raw_images/Siviglia/` is NEVER modified. By default, no rotated image copy is saved; the manifest's `rotation_ccw_deg` lets you reconstruct it. If you check **Save oriented image copy on save**, the app writes PNG copies in `data/processed/upright_images/`. Those files can take substantial disk space. You can also set `save_corrected_images = true` in `setup.toml`.

The app updates the two CSVs with atomic replacements to reduce the chance of partial files from an interrupted write. Save frequently. **Leaving an image with Previous / Next warns about unsaved edits**; simply closing napari does not autosave.

## `.gitignore`

Add the following entries to the project's `.gitignore`, without ignoring your annotation CSVs:

```gitignore
data/raw_images/
data/processed/upright_images/
```

Commit the configuration, annotator script and CSVs once the pilot annotations look correct. Keeping the images ignored prevents unnecessarily uploading the photographs.

## Troubleshooting

- `ModuleNotFoundError: napari` → use `python -m pip install "napari[pyqt6, optional]"` within the active `owl-unit` environment.
- Qt says it cannot open a display under WSL → check `echo $DISPLAY` and `echo $WAYLAND_DISPLAY`; run from a WSLg-capable desktop session.
- Clicking the photo does nothing → select **Owl image (CLICK HERE)** in the layers list (rather than the points layer). The widget selects this layer for you when navigating.
- The landmarks disappear when you rotate → they should be transformed from original coordinates automatically; ensure you are running the supplied script, not an older annotator.

## Git note

If your repository already has a customised `setup.toml` or `run.py`, do not overwrite them blindly. Copy just the `[napari_annotation]` section and `napari = "scripts/04_napari_annotate.py"` under its existing `[steps]` section, then copy the Python script and notebook.
