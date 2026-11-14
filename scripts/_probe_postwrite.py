#!/usr/bin/env python3
"""Temporary probe: post-write validation of the regenerated notebooks.

- NB1 (prosody_emotion_train.ipynb): structure, Step 8 markers, byte-identical
  embedded prosody_emotion.py / train_prosody_emotion.py payloads.
- NB2 (runpod_emotion_streaming_prosody.ipynb) and the source
  runpod_emotion_streaming.ipynb: Step 12 embedded PROSODY_MODULE byte-identical
  to the repo file, AUTO_EMOTION_MODULE byte-identical, no stale v1 weights.
- Functional smoke of the NB2-embedded prosody module in a temp dir (import,
  extract_features, rule predict, v2 weight roundtrip, v1 rejection).
"""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB1 = ROOT / "prosody_emotion_train.ipynb"
NB2 = ROOT / "runpod_emotion_streaming_prosody.ipynb"
SRC = ROOT / "runpod_emotion_streaming.ipynb"

failures = []


def check(name, ok, info=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" - {info}" if info else ""))
    if not ok:
        failures.append(name)


pe_repo = (ROOT / "IMTalker" / "prosody_emotion.py").read_text(encoding="utf-8").replace("\r\n", "\n")
tr_repo = (ROOT / "scripts" / "train_prosody_emotion.py").read_text(encoding="utf-8").replace("\r\n", "\n")
ae_repo = (ROOT / "IMTalker" / "auto_emotion.py").read_text(encoding="utf-8").replace("\r\n", "\n")

# --- NB1 ---------------------------------------------------------------------
nb1 = json.loads(NB1.read_text(encoding="utf-8"))
ids1 = [c["id"] for c in nb1["cells"]]
check("NB1 has 19 cells with sequential ids",
      ids1 == [f"prosody-train-{i:02d}" for i in range(19)], f"{len(ids1)} cells")
cell3 = "".join(nb1["cells"][3]["source"])
check("NB1 embeds byte-identical prosody_emotion.py", pe_repo in cell3)
check("NB1 embeds byte-identical train_prosody_emotion.py", tr_repo in cell3)
step8 = "".join(nb1["cells"][17]["source"])
for needle in ("TEST_AUDIO_DIR", "SEGMENT_S", "classify_wav", "max|z|", "OOD_Z",
               "feat_mean", "chunks:"):
    check(f"NB1 Step 8 contains {needle!r}", needle in step8)
check("NB1 Step 8 markdown documents segmentation + OOD",
      "segmented" in "".join(nb1["cells"][16]["source"])
      and "Out-of-distribution" in "".join(nb1["cells"][16]["source"]))


# --- NB2 + source: Step 12 embedded payloads ----------------------------------
def step12_src(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell = [c for c in nb["cells"] if c.get("id") == "emotion-stream-33"]
    return "".join(cell[0]["source"]), len(nb["cells"])


def extract_payload(src, marker):
    m = re.search(marker + r" = r'''(.*?)'''", src, re.DOTALL)
    return m.group(1) if m else None


for label, path in (("NB2", NB2), ("source NB", SRC)):
    src12, n_cells = step12_src(path)
    pe_emb = extract_payload(src12, "PROSODY_MODULE")
    ae_emb = extract_payload(src12, "AUTO_EMOTION_MODULE")
    check(f"{label}: Step 12 PROSODY_MODULE byte-identical to repo file",
          pe_emb == pe_repo,
          f"{len(pe_emb or '')} vs {len(pe_repo)} chars")
    check(f"{label}: Step 12 AUTO_EMOTION_MODULE byte-identical to repo file",
          ae_emb == ae_repo)
    m = re.search(r'PROSODY_WEIGHTS_B64 = "([^"]*)"', src12)
    check(f"{label}: no stale v1 weights embedded", m is not None and m.group(1) == "",
          f"{len(m.group(1)) if m else -1} b64 chars")

nb2 = json.loads(NB2.read_text(encoding="utf-8"))
check("NB2 has source+1 cells", len(nb2["cells"]) == len(json.loads(SRC.read_text(encoding='utf-8'))["cells"]) + 1,
      f"{len(nb2['cells'])} cells")

# --- functional smoke of the NB2-embedded prosody module ----------------------
import numpy as np  # noqa: E402

src12, _ = step12_src(NB2)
pe_emb = extract_payload(src12, "PROSODY_MODULE")
tmp = Path(tempfile.mkdtemp(prefix="prosody_postwrite_"))
(tmp / "prosody_emotion.py").write_text(pe_emb, encoding="utf-8")
sys.path.insert(0, str(tmp))
import prosody_emotion as pe  # noqa: E402

check("embedded module is format v3", pe.WEIGHTS_FORMAT_VERSION == 3,
      f"v{pe.WEIGHTS_FORMAT_VERSION}, z-clip {pe.Z_CLIP}, target rms {pe.TARGET_RMS}")
sr = pe.INPUT_SR
t = np.arange(int(2.5 * sr)) / sr
x = (0.2 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
f1 = pe.extract_features(x)
f2 = pe.extract_features(x * 4.0)
check("embedded module: finite features", np.isfinite(f1).all() and f1.shape == (pe.FEATURE_DIM,))
check("embedded module: gain-invariant features",
      np.allclose(f1, f2, rtol=5e-2, atol=5e-3),
      f"max diff {np.abs(f1 - f2).max():.4f}")
clf = pe.ProsodyEmotionClassifier(weights_path=tmp / "nope.npz", verbose=False)
check("embedded module: rule fallback when no weights", clf.backend == "rules", clf.backend)
eid, label, conf = clf.predict(x)
check("embedded module: rule predict contract", 0 <= eid <= 5 and 0.0 <= conf <= 1.0,
      f"{eid} {label} {conf:.2f}")

wpath = tmp / "w.npz"
rng = np.random.default_rng(0)
W = rng.normal(0, 0.1, (pe.NUM_EMOTIONS, pe.FEATURE_DIM))
b = np.zeros(pe.NUM_EMOTIONS)
mu = f1.copy()
sd = np.full(pe.FEATURE_DIM, 1.5, dtype=np.float64)
pe.save_weights(wpath, W, b, mu, sd)
model = pe.load_weights(wpath)
check("embedded module: v2 weights roundtrip", model.W.shape == W.shape)
v1path = tmp / "v1.npz"
np.savez(v1path, W=W, b=b, feat_mean=mu, feat_std=sd,
         feature_names=np.array(pe.FEATURE_NAMES), temperature=np.float32(1.0),
         format_version=np.int32(1))
try:
    pe.load_weights(v1path)
    check("embedded module: v1 weights rejected", False)
except ValueError as exc:
    check("embedded module: v1 weights rejected", "format v1" in str(exc))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print(f"[ok] post-write validation passed ({tmp})")
