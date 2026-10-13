import importlib.util
from pathlib import Path
import numpy as np
import tempfile
import csv
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / 'scripts/04_napari_annotate.py'
spec = importlib.util.spec_from_file_location('annotator', SRC)
annotator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotator)

def test_roundtrip():
    h,w=240,400
    arr=np.arange(h*w).reshape(h,w)
    for k in range(4):
        rot=np.rot90(arr,k)
        for y,x in [(0,0),(0,w-1),(h-1,0),(h-1,w-1),(50,201),(22.5,33.25)]:
            yy,xx=annotator.image_to_rotated(y,x,h,w,k)
            assert 0<=yy<rot.shape[0] and 0<=xx<rot.shape[1]
            y0,x0=annotator.rotated_to_image(yy,xx,h,w,k)
            assert np.allclose((y0,x0),(y,x))
            if float(y).is_integer() and float(x).is_integer():
                assert rot[int(yy),int(xx)]==arr[int(y),int(x)]

def test_manifest_and_resume():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        folder = root / "data/raw_images/Siviglia"
        folder.mkdir(parents=True)
        Image.new("RGB", (35, 25)).save(folder / "owl_A.jpg")
        Image.new("RGB", (40, 30)).save(folder / "owl_B.jpg")
        settings = {
            "images_root": folder, "extensions": {".jpg"}, "recursive": True,
            "manifest_path": root / "data/annotations/owl_manifest.csv",
            "keypoints_path": root / "data/annotations/owl_keypoints.csv",
            "corrected_dir": root / "data/processed/upright_images",
            "repo_root": root, "landmarks": [f"face_{j:02d}" for j in range(1,9)] + ["beak_tip"],
        }
        db = annotator.AnnotationStorage(settings, limit=1)
        assert len(db.manifest) == 1
        assert db.ids == ["owl_A.jpg"]
        entry = db.manifest["owl_A.jpg"] | {"status": "annotated", "facing": "Right", "rotation_ccw_deg": "90"}
        points = [{"image_id": "owl_A.jpg", "landmark": "beak_tip", "visibility": "visible", "x_oriented": 12, "y_oriented": 13, "x_original": 13, "y_original": 14}]
        db.save("owl_A.jpg", entry, points)
        db2 = annotator.AnnotationStorage(settings)
        assert len(db2.manifest) == 2
        assert db2.manifest["owl_A.jpg"]["status"] == "annotated"
        assert db2.manifest["owl_B.jpg"]["status"] == "pending"
        assert db2.get_keypoints("owl_A.jpg")["beak_tip"]["x_oriented"] == "12"

if __name__=='__main__':
    test_roundtrip()
    test_manifest_and_resume()
    print('OK: orientation coordinate roundtrips and CSV manifest persistence')
