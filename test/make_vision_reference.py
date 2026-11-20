"""
Reference outputs from the laptop demo (eye_quality_gate_demo.py) for the web app's parity test.

For every saved burst in captures/ (8 frames per eye) it records what the Python code produces:
alignment shifts, glare/opacity masks, opacity fraction, the de-glared composite, the 224x224
classifier input and the model scores. test_vision.mjs runs web/vision.js + the ONNX model on the
same frames and compares.

Also writes a resize/quality reference on a few dataset crops.

Run (inside .venv):
    python web/test/make_vision_reference.py
Writes web/test/_out/ (git-ignored: derived from personal eye photos).
"""

import csv
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import eye_quality_gate_demo as demo  # noqa: E402

OUT = ROOT / "web/test/_out"


class CapturingClassifier(demo.EyeClassifier):
    """Keeps the exact crop handed to the model."""

    def predict(self, crop_bgr):
        self.last_crop = crop_bgr.copy()
        return super().predict(crop_bgr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clf = CapturingClassifier()
    rows = {r["folder"]: r for r in csv.DictReader(open(ROOT / "captures/captures.csv"))}
    bursts = []
    for folder in sorted(glob.glob(str(ROOT / "captures/*"))):
        name = os.path.basename(folder)
        frames = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
        if name not in rows or len(frames) < 8 or not rows[name]["iris_to_crop_ratio"]:
            continue
        crops = [cv2.imread(f) for f in frames]
        ratio = float(rows[name]["iris_to_crop_ratio"])
        grays = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops]
        _, shifts = demo._align_gray_frames(grays)
        res = demo.glare_separation(crops, [ratio] * 8, [1.0] * 8)
        pred = demo.classify_eye(clf, res, [ratio] * 8, [1.0] * 8)
        rgb_input = cv2.cvtColor(clf.last_crop, cv2.COLOR_BGR2RGB)
        rgb_input.astype(np.uint8).tofile(OUT / f"{name}_input.bin")
        cv2.cvtColor(res["median_color"], cv2.COLOR_BGR2RGB).tofile(OUT / f"{name}_median.bin")
        q = demo.quality_check(crops[0])
        bursts.append({
            "name": name, "frames": [str(Path(f).relative_to(ROOT)).replace("\\", "/") for f in frames],
            "ratio": ratio, "shifts": [[float(a), float(b)] for a, b in shifts],
            "opacity_fraction": float(res["opacity_fraction"]),
            "glare_pixels": int((res["glare_transient_mask"] > 0).sum()),
            "opacity_pixels": int((res["stable_opacity_mask"] > 0).sum()),
            "quality_frame0": {"passed": bool(q[0]), "blur": float(q[1]), "brightness": float(q[2])},
            "cataract_score": pred["cataract_score"], "mature_score": pred["mature_score"], "output": pred["output"],
        })

    # resize + quality on dataset crops at their original (varied) sizes
    resize_cases = []
    for f in sorted((ROOT / "processed_images/test/normal").glob("*.png"))[:4] + \
            sorted((ROOT / "processed_images/test/cataract").glob("*.png"))[:4]:
        img = cv2.imread(str(f))
        area = cv2.resize(img, (180, 180), interpolation=cv2.INTER_AREA)
        linear = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        q = demo.quality_check(img)
        key = f.stem.replace(" ", "_")
        cv2.cvtColor(area, cv2.COLOR_BGR2RGB).tofile(OUT / f"resize_{f.parent.name}_{key}_area.bin")
        cv2.cvtColor(linear, cv2.COLOR_BGR2RGB).tofile(OUT / f"resize_{f.parent.name}_{key}_linear.bin")
        resize_cases.append({"file": str(f.relative_to(ROOT)).replace("\\", "/"), "key": f"{f.parent.name}_{key}",
                             "quality": {"passed": bool(q[0]), "blur": float(q[1]), "brightness": float(q[2])}})

    (OUT / "reference.json").write_text(json.dumps({
        "bursts": bursts, "resize_cases": resize_cases,
        "thresholds": {"screen_threshold": float(clf.heads["screen_threshold"]),
                       "severity_threshold": float(clf.heads["severity_threshold"])}}, indent=1))
    print(f"{len(bursts)} bursts and {len(resize_cases)} resize cases written to {OUT}")


if __name__ == "__main__":
    main()
