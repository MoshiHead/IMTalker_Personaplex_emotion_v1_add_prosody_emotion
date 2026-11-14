#!/usr/bin/env python3
"""Generate the two prosody-emotion notebooks.

Outputs (written to the repo root):

1. ``prosody_emotion_train.ipynb`` — dataset download (RAVDESS + optional
   CREMA-D) + training of the ``prosody-v1`` softmax weights. Clones the
   user's repo and self-heals (writes embedded byte-identical copies of
   ``IMTalker/prosody_emotion.py`` and ``scripts/train_prosody_emotion.py``)
   when the repo does not contain them yet.

2. ``runpod_emotion_streaming_prosody.ipynb`` — a copy of the battle-tested
   ``runpod_emotion_streaming.ipynb`` with surgical changes only: new-repo
   clone with fallback, ``PROSODY_WEIGHTS_FILE`` hand-off from notebook 1,
   external-weights priority in Step 12, and a backend print in the SER smoke
   test.

Usage:
    python scripts/make_prosody_notebooks.py            # build + validate
    python scripts/make_prosody_notebooks.py --dryrun   # also execute NB1's
                                                        # core cells end-to-end
                                                        # on synthetic data
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "runpod_emotion_streaming.ipynb"
NB1_PATH = ROOT / "prosody_emotion_train.ipynb"
NB2_PATH = ROOT / "runpod_emotion_streaming_prosody.ipynb"

NEW_REPO = "https://github.com/MoshiHead/IMTalker_Personaplex_emotion_v1_add_prosody_emotion.git"
PREV_REPO = "https://github.com/MoshiHead/IMTalker_Personaplex_emotion_v1_auto_emotion_v1.git"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_payloads():
    """Read the two repo files embedded into NB1 as self-heal payloads."""
    pe = (ROOT / "IMTalker" / "prosody_emotion.py").read_text(encoding="utf-8")
    tr = (ROOT / "scripts" / "train_prosody_emotion.py").read_text(encoding="utf-8")
    for name, payload in (("prosody_emotion.py", pe), ("train_prosody_emotion.py", tr)):
        assert "'''" not in payload, f"{name} contains a triple-quote - cannot embed safely"
        assert not payload.endswith("\\"), f"{name} ends with a backslash - cannot embed safely"
    return pe, tr


def _src(text):
    """Notebook source field: list of lines, each keeping its newline."""
    return text.splitlines(keepends=True)


def md_cell(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _src(text)}


def code_cell(cell_id, text):
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "execution_count": None, "outputs": [], "source": _src(text)}


def make_nb(cells, metadata=None):
    meta = metadata or {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def replace_once(text, old, new, label):
    n = text.count(old)
    assert n == 1, f"{label}: expected exactly 1 occurrence of the anchor, found {n}"
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# Notebook 1 — prosody_emotion_train.ipynb (markdown cells)
# ---------------------------------------------------------------------------

NB1_MD_00 = """\
# Prosody Emotion Training — RAVDESS (+ CREMA-D) → `prosody_emotion.npz`

Trains the built-in **`prosody-v1`** speech-emotion backend from scratch. No
pretrained models, no GPU, no torch — pure numpy, and the exact same feature
code runs at inference time inside the live avatar (train/serve parity by
construction).

**Pipeline**
1. Clone the project repo (or reuse the existing checkout) and make sure the
   prosody files are present — self-healing: embedded byte-identical copies
   are written when the repo does not contain them yet.
2. Download + extract **RAVDESS** speech audio (Zenodo, ~215 MB, 1,440 clips,
   24 actors — CC BY-NC-SA, non-commercial research use; cite Livingstone &
   Russo 2018).
3. Optionally download **CREMA-D** audio (git-LFS, ~600 MB, 7,442 clips,
   91 actors). Adds speaker diversity; skipped automatically if the download
   fails.
4. Extract the 56-dim prosodic features (optional 5x augmentation) and train a
   6-class softmax regression in numpy — speaker-grouped k-fold CV, then a
   final refit on 100% of the data.
5. Write the ~7 KB weights to `IMTalker/checkpoints/prosody_emotion.npz` and
   copy them to `/workspace/prosody_emotion.npz` for the streaming notebook.
6. *(Optional)* Batch-test any audio folder against the trained weights with
   confidence scores (Step 8).

**Runtime** on a RunPod CPU: ~25–40 min total (downloads ~5–10 min, feature
extraction ~15–25 min with augmentation, the training itself takes seconds).
A CPU-only pod is enough.

**Expected accuracy:** ~55–70% speaker-independent CV (prosody-only features
on acted speech). That is the honest range for hand-crafted prosodic features —
the win over the built-in rule-based fallback is consistency, not magic.

> **Companion notebook:** `runpod_emotion_streaming_prosody.ipynb` picks the
> trained weights up from `PROSODY_WEIGHTS_FILE` (`/workspace/prosody_emotion.npz`)
> automatically when run on the same pod. If you train elsewhere, upload the
> small `.npz` to the pod at that path (Jupyter file upload) before its Step 12.
"""

NB1_CODE_01 = """\
# --- Parameters (edit me) -----------------------------------------------------
REPO_URL = "https://github.com/MoshiHead/IMTalker_Personaplex_emotion_v1_add_prosody_emotion.git"
PROJECT_ROOT = "/workspace/speech2avatar"       # repo checkout (shared with the streaming notebook)
DATA_DIR = "/workspace/prosody_data"            # dataset downloads (kept outside the repo)
WEIGHTS_OUT = "/workspace/prosody_emotion.npz"  # hand-off path the streaming notebook reads
USE_CREMAD = True   # +7,442 clips / 91 speakers via git-lfs; auto-falls back to RAVDESS-only
AUGMENT = True      # 5x data: gain {0.7, 1.3} x tape-speed {0.9, 1.1}
FOLDS = 5           # speaker-grouped cross-validation folds
SEED = 0
"""

NB1_MD_02 = """\
## Step 1 — Clone / verify the project repo

Clones the prosody-emotion repo into `PROJECT_ROOT` (skipped when a checkout is
already present, e.g. shared with the streaming notebook). Then verifies the
two files this notebook needs:

- `IMTalker/prosody_emotion.py` — 56-dim feature extractor + classifier
- `scripts/train_prosody_emotion.py` — the training pipeline

If the repo does not contain them yet (fresh repo), **embedded copies** —
byte-identical to the source of truth — are written, with a warning. Push your
local workspace to the repo to make the embedded path unnecessary.
"""

NB1_MD_04 = """\
## Step 2 — Python dependencies

Only tiny wheels: `numpy` (features + training) and `soundfile` (WAV loading;
`scipy` is an automatic fallback loader when absent). No torch, no
transformers, no GPU.
"""

NB1_MD_06 = """\
## Step 3 — Download RAVDESS (required)

The speech portion of the Ryerson Audio-Visual Database of Emotional Speech
and Song: `Audio_Speech_Actors_01-24.zip` (~215 MB, 1,440 clips, 24 actors,
48 kHz 16-bit WAV) from Zenodo record 1188976.

Filename convention: `03-01-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav`
with emotion codes 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
07 disgust, 08 surprised. Training maps **neutral+calm → 0**, happy → 1,
sad → 2, angry → 3, **surprised → 4**, fearful → 5 (disgust is skipped: the
avatar has no disgust expression).

License: CC BY-NC-SA 4.0 — non-commercial research use; cite Livingstone &
Russo (2018), PLoS ONE 13(5): e0196391.

Idempotent: re-running the cell skips an already-extracted dataset.
"""

NB1_MD_08 = """\
## Step 4 — Download CREMA-D (optional, recommended)

7,442 clips from 91 actors (diverse ages/accents) — the speaker diversity that
RAVDESS (24 actors) lacks. Clips with `XX` (unspecified) intensity are skipped;
the rest map onto the same 6 classes.

The wavs live in **git-LFS** inside the CREMA-D GitHub repo (a plain clone only
yields 130-byte pointer files), so this cell installs `git-lfs` (via apt when
missing) and does a sparse blobless clone + `git lfs pull` of `AudioWAV/` only
(~600 MB). If anything fails — e.g. the repo's public LFS bandwidth quota —
the cell prints a warning and training continues **RAVDESS-only**. Set
`USE_CREMAD = False` to skip deliberately.

Idempotent: re-running skips a completed download.
"""

NB1_MD_10 = """\
## Step 5 — Dataset audit

Counts the usable clips per avatar emotion class for each dataset, using the
trainer's own scanners — so what you see is exactly what training will use.
"""

NB1_MD_12 = """\
## Step 6 — Train

Runs `scripts/train_prosody_emotion.py`:

- 56-dim prosodic features per clip (F0 stats/contour, voiced-time RMS stats,
  pauses / speech rate, spectral flux / centroid, voiced ZCR, coarse formant
  proxies) — the identical code path the live avatar uses at inference.
- **Noise augmentation (always on)**: every training clip also appears once
  with added white/pink noise at a random 6–30 dB SNR. RAVDESS/CREMA-D are
  studio recordings; without this the model learns "bright/noisy spectrum =
  angry" and collapses *every real-world recording to angry*. This is the
  channel-robustness fix.
- **Your own audio** (`EXTRA_AUDIO_DIR`): files named `<emotion>-*.wav`
  (angry/fear/happy/sad/neutral/surprise) are added to training ×10. They form
  their own CV speaker group, so the report shows an honest held-out accuracy
  on *your* audio. Set to `""` for pure dataset training.
- Standardize → numpy softmax regression (full-batch L-BFGS).
- **Speaker-grouped k-fold CV** (no speaker appears in both train and test) →
  an honest generalization estimate, then a final refit on 100% of the data.
- Writes `prosody_emotion.npz` (~7 KB: weights, bias, mu, sd, meta).

Expected duration: ~15–30 min with CREMA-D (CPU-bound feature extraction),
a few minutes RAVDESS-only.
"""

NB1_MD_14 = """\
## Step 7 — Validate + hand off

Loads the trained weights through `ProsodyEmotionClassifier` (the inference
entry point used by the avatar), classifies one demo clip per emotion, measures
per-utterance latency, and copies the weights to `WEIGHTS_OUT` — the exact
path the streaming notebook's `PROSODY_WEIGHTS_FILE` parameter points to.
"""

NB1_MD_16 = """\
## Done — next steps

1. Run `runpod_emotion_streaming_prosody.ipynb` on the **same pod** (shared
   `/workspace`): its Step 12 copies `/workspace/prosody_emotion.npz` into
   `IMTalker/checkpoints/` and the SER smoke test prints
   `prosody backend = softmax`.
2. Trained on another machine? Upload `prosody_emotion.npz` (~7 KB) to the pod
   at `/workspace/prosody_emotion.npz` (Jupyter file upload) before Step 12.
3. No weights file at all → the avatar still runs on the built-in rule-based
   prosody fallback; retraining is optional.
4. **Step 8** above batch-tests any audio folder (`TEST_AUDIO_DIR`) against
   the trained weights with confidence scores — re-runnable at any time.

**Rollback:** delete `IMTalker/checkpoints/prosody_emotion.npz` (rule fallback)
or set `SER_MODEL` to any HF backend — every previous SER model remains
available.

**Retraining tips:** more data helps most (add IEMOCAP/MSP via the same
filename→label pattern); `AUGMENT = False` for a quick iteration; `FOLDS`
trades CV runtime vs. estimate stability.

**Citations:** RAVDESS — Livingstone & Russo (2018), PLoS ONE 13(5): e0196391
(CC BY-NC-SA 4.0). CREMA-D — Cao et al. (2014), IEEE Trans. Affective
Computing 5(4) (see the CREMA-D repo for its license terms).
"""

# ---------------------------------------------------------------------------
# Notebook 1 — code cells
# ---------------------------------------------------------------------------

def nb1_code_03(pe_payload, tr_payload):
    """Clone + verify + self-heal cell, with the two embedded payloads."""
    return (
        "import subprocess\n"
        "from pathlib import Path\n"
        "\n"
        'REQUIRED_FILES = ["IMTalker/prosody_emotion.py", "scripts/train_prosody_emotion.py"]\n'
        "\n"
        "# Embedded byte-identical copies, written only when the repo lacks them.\n"
        "EMBEDDED = {\n"
        '    "IMTalker/prosody_emotion.py": r\'\'\'' + pe_payload + "''',\n"
        '    "scripts/train_prosody_emotion.py": r\'\'\'' + tr_payload + "''',\n"
        "}\n"
        "\n"
        "\n"
        "def have_all(root):\n"
        "    return all((Path(root) / rel).is_file() for rel in REQUIRED_FILES)\n"
        "\n"
        "\n"
        "if have_all(PROJECT_ROOT):\n"
        '    print(f"[skip] repo already present with prosody files: {PROJECT_ROOT}")\n'
        "else:\n"
        "    if not Path(PROJECT_ROOT).exists():\n"
        '        print(f"[clone] {REPO_URL} -> {PROJECT_ROOT}")\n'
        '        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, PROJECT_ROOT], check=True)\n'
        "    if have_all(PROJECT_ROOT):\n"
        '        print(f"[ok] cloned repo contains the prosody files: {PROJECT_ROOT}")\n'
        "    else:\n"
        '        print("[warn] the repo checkout is missing the prosody files - writing embedded copies")\n'
        '        print("       (push your local workspace to the repo to make this unnecessary)")\n'
        "        for rel, payload in EMBEDDED.items():\n"
        "            dest = Path(PROJECT_ROOT) / rel\n"
        "            dest.parent.mkdir(parents=True, exist_ok=True)\n"
        '            dest.write_text(payload, encoding="utf-8")\n'
        '            print(f"[written] {dest} ({len(payload)} chars, embedded copy)")\n'
        "\n"
        'assert have_all(PROJECT_ROOT), f"prosody files missing after setup: {REQUIRED_FILES}"\n'
        'print(f"[ok] project ready: {PROJECT_ROOT}")\n'
    )


NB1_CODE_05 = """\
import importlib
import subprocess
import sys


def ensure(module, pip_name=None):
    try:
        importlib.import_module(module)
        print(f"[ok] {module}")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or module], check=True)
        print(f"[installed] {pip_name or module}")


ensure("numpy")
ensure("soundfile")
try:
    importlib.import_module("scipy")
    print("[ok] scipy (optional fallback loader available)")
except ImportError:
    print("[info] scipy not installed - soundfile will be used (that is fine)")
"""

NB1_CODE_07 = """\
import time
import urllib.request
import zipfile
from pathlib import Path

DATA_ROOT = Path(DATA_DIR)
RAVDESS_URLS = [
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
    "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content",
]
RAVDESS_ZIP = DATA_ROOT / "Audio_Speech_Actors_01-24.zip"
RAVDESS_DIR = DATA_ROOT / "ravdess"


def download(urls, dest, min_bytes=1_000_000, retries=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        url = urls[(attempt - 1) % len(urls)]
        try:
            print(f"[download] {url} (attempt {attempt}/{retries})")
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                total = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
            if total < min_bytes:
                raise IOError(f"downloaded only {total} bytes")
            print(f"[ok] {dest.name}: {total / 1e6:.1f} MB in {time.time() - t0:.0f}s")
            return
        except Exception as exc:
            print(f"[warn] attempt {attempt} failed: {exc!r}")
            last = exc
            if dest.exists():
                dest.unlink()
    raise last


if (RAVDESS_DIR / ".done").is_file():
    print(f"[skip] RAVDESS already extracted: {RAVDESS_DIR}")
else:
    download(RAVDESS_URLS, RAVDESS_ZIP, min_bytes=200_000_000)
    with zipfile.ZipFile(RAVDESS_ZIP) as zf:
        bad = zf.testzip()
        assert bad is None, f"corrupt member in {RAVDESS_ZIP}: {bad}"
        print(f"[extract] {RAVDESS_ZIP.name} -> {RAVDESS_DIR} (~1 min)")
        RAVDESS_DIR.mkdir(parents=True, exist_ok=True)
        zf.extractall(RAVDESS_DIR)
    wavs = sorted(RAVDESS_DIR.rglob("*.wav"))
    assert len(wavs) == 1440, f"expected 1440 RAVDESS clips, found {len(wavs)}"
    (RAVDESS_DIR / ".done").write_text("ok", encoding="utf-8")

print(f"[ok] RAVDESS ready: {len(list(RAVDESS_DIR.rglob('*.wav')))} clips in {RAVDESS_DIR}")
"""

NB1_CODE_09 = """\
import shutil
import subprocess
from pathlib import Path

CREMAD_DIR = DATA_ROOT / "cremad"
CREMAD_AUDIO_DIR = CREMAD_DIR / "AudioWAV"
cremad_ok = False


def _run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr[-400:]}")
    return proc


if not USE_CREMAD:
    print("[skip] USE_CREMAD = False")
elif (CREMAD_DIR / ".done").is_file():
    cremad_ok = True
    print(f"[skip] CREMA-D already downloaded: {CREMAD_AUDIO_DIR}")
else:
    try:
        try:
            _run(["git", "lfs", "version"])
        except Exception:
            print("[setup] git-lfs missing - installing via apt-get (RunPod pods run as root)")
            try:
                _run(["apt-get", "update", "-qq"], timeout=300)
            except Exception as exc:
                print(f"[warn] apt-get update failed ({exc!r}); trying install anyway")
            _run(["apt-get", "install", "-y", "git-lfs"], timeout=600)
        _run(["git", "lfs", "install"])
        if CREMAD_DIR.exists():
            shutil.rmtree(CREMAD_DIR)
        _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
              "https://github.com/CheyneyComputerScience/CREMA-D.git", str(CREMAD_DIR)], timeout=600)
        _run(["git", "-C", str(CREMAD_DIR), "sparse-checkout", "set", "AudioWAV"], timeout=600)
        print("[download] git lfs pull (AudioWAV, ~600 MB) - this takes a few minutes ...")
        _run(["git", "-C", str(CREMAD_DIR), "lfs", "pull"], timeout=5400)
        wavs = sorted(CREMAD_AUDIO_DIR.glob("*.wav"))
        assert len(wavs) >= 7000, f"expected ~7442 CREMA-D clips, found {len(wavs)}"
        small = [w for w in wavs[:50] if w.stat().st_size < 2048]
        assert not small, "files look like git-LFS pointers (lfs pull did not fetch content)"
        (CREMAD_DIR / ".done").write_text("ok", encoding="utf-8")
        cremad_ok = True
        print(f"[ok] CREMA-D ready: {len(wavs)} clips in {CREMAD_AUDIO_DIR}")
    except Exception as exc:
        print(f"[warn] CREMA-D download failed: {exc!r}")
        print("[warn] continuing with RAVDESS only (1,440 clips, 24 speakers) -")
        print("       the model still trains; extra speaker diversity would improve it")
"""

NB1_CODE_11 = """\
import importlib.util
from collections import Counter
from pathlib import Path

# Import the trainer module by path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "train_prosody_emotion", Path(PROJECT_ROOT) / "scripts" / "train_prosody_emotion.py")
trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer)

NAMES = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "surprised", 5: "fearful"}


def audit(label, files):
    # scanner rows are (wav_path, emotion_id, speaker_id)
    counts = Counter(fid for _, fid, _spk in files)
    total = max(1, sum(counts.values()))
    print(f"{label}: {total} clips")
    for fid in range(6):
        n = counts.get(fid, 0)
        print(f"  id={fid} {NAMES[fid]:<9} {n:>5} clips ({100 * n / total:.1f}%)")


ravdess_files = trainer.scan_ravdess(str(RAVDESS_DIR))
audit("RAVDESS (speech, neutral+calm merged)", ravdess_files)
cremad_files = []
if cremad_ok:
    cremad_files = trainer.scan_cremad(str(CREMAD_AUDIO_DIR))
    audit("CREMA-D (XX intensity skipped)", cremad_files)
total = len(ravdess_files) + len(cremad_files)
assert total > 0, "no training clips found - check the download cells above"
print(f"[ok] total training clips: {total}")
"""

NB1_CODE_13 = """\
import subprocess
import sys
import time
from pathlib import Path

WEIGHTS_PATH = Path(PROJECT_ROOT) / "IMTalker" / "checkpoints" / "prosody_emotion.npz"
WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Your own labeled audio (files named <emotion>-*.wav) joins the training set
# x10 - adapts the model to your voice/room. Set to "" for pure dataset training.
EXTRA_AUDIO_DIR = "/workspace/audio"

cmd = [
    sys.executable, str(Path(PROJECT_ROOT) / "scripts" / "train_prosody_emotion.py"),
    "--ravdess_dir", str(RAVDESS_DIR),
    "--out", str(WEIGHTS_PATH),
    "--folds", str(FOLDS), "--seed", str(SEED),
]
if cremad_ok:
    cmd += ["--cremad_dir", str(CREMAD_AUDIO_DIR)]
if AUGMENT:
    cmd += ["--augment"]
if EXTRA_AUDIO_DIR:
    if Path(EXTRA_AUDIO_DIR).is_dir():
        cmd += ["--extra_dir", EXTRA_AUDIO_DIR]
    else:
        print(f"[skip] EXTRA_AUDIO_DIR does not exist: {EXTRA_AUDIO_DIR}")
print("[run]", " ".join(cmd))
t0 = time.time()
proc = subprocess.run(cmd)
assert proc.returncode == 0, "training failed - see the output above"
print(f"[ok] training finished in {(time.time() - t0) / 60:.1f} min")
assert WEIGHTS_PATH.is_file() and WEIGHTS_PATH.stat().st_size > 1000
print(f"[ok] weights: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size:,} bytes)")
"""

NB1_CODE_15 = """\
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(PROJECT_ROOT) / "IMTalker"))
import prosody_emotion as pe

clf = pe.ProsodyEmotionClassifier(weights_path=str(WEIGHTS_PATH))
assert clf.backend == "softmax", f"expected the trained softmax backend, got {clf.backend!r}"
print(f"[ok] loaded trained classifier: backend={clf.backend}")

# One demo clip per emotion (CREMA-D preferred for speaker diversity; RAVDESS
# fills in). Single clips are a sanity check, not an accuracy estimate.
demo = {}
for path, fid, _spk in (trainer.scan_cremad(str(CREMAD_AUDIO_DIR)) if cremad_ok else []):
    demo.setdefault(fid, path)
for path, fid, _spk in trainer.scan_ravdess(str(RAVDESS_DIR)):
    demo.setdefault(fid, path)

n_correct = 0
for fid in sorted(demo):
    wav = trainer.load_wav(demo[fid])  # resampled to 24 kHz, the classifier's input rate
    y, label, conf = clf.predict(wav)
    ok = "OK  " if y == fid else "MISS"
    n_correct += (y == fid)
    print(f"  [{ok}] expected id={fid} {NAMES[fid]:<9} -> predicted id={y} "
          f"{label:<10} conf={conf:.2f}  ({Path(demo[fid]).name})")
print(f"[info] demo sanity check: {n_correct}/{len(demo)} correct (single clips)")

# Latency: the live avatar runs this on a daemon CPU thread per utterance.
wav = trainer.load_wav(demo[sorted(demo)[0]])
for _ in range(3):
    clf.predict(wav)  # warm-up
t0 = time.perf_counter()
reps = 20
for _ in range(reps):
    clf.predict(wav)
ms = (time.perf_counter() - t0) / reps * 1000
print(f"[ok] latency: {ms:.1f} ms/utterance on this CPU")

# Hand off: the streaming notebook picks the weights up from WEIGHTS_OUT.
shutil.copyfile(WEIGHTS_PATH, WEIGHTS_OUT)
print(f"[ok] copied -> {WEIGHTS_OUT}")
print("[next] run runpod_emotion_streaming_prosody.ipynb on this pod - it finds")
print("       the weights at PROSODY_WEIGHTS_FILE automatically. If this notebook")
print("       ran elsewhere, upload the .npz to the pod at /workspace/prosody_emotion.npz.")
"""


NB1_MD_17 = """\
## Step 8 — Test on your own audio folder (optional, re-runnable)

Point `TEST_AUDIO_DIR` at any folder of audio clips and re-run the cell below
as often as you like — no retraining needed. The trained `.npz` is loaded once
and every file is classified with its **confidence score** plus the top-3
probability breakdown.

Recursively scans `.wav .flac .ogg .mp3 .m4a` (mp3/m4a support depends on the
soundfile build; unreadable files are reported, not fatal). Silent or
too-short clips are skipped with a note. The cell reloads the trainer helpers
and the trained weights if they are not in memory, so it also works after a
kernel restart.

Two robustness details (mirroring the live avatar):

- **Long files are segmented** into ~4 s chunks — the model was trained on
  1–10 s utterances, and the live avatar classifies VAD segments of the same
  size. Per-chunk votes are shown and the final prediction averages the
  chunk probabilities.
- **Out-of-distribution warnings**: if a file's features sit far outside the
  RAVDESS training distribution (max |z| > 5, e.g. another language, shouting,
  singing), the cell prints which features are off-scale. Predictions there
  are clipped to the |z| ≤ 5 boundary instead of saturating — treat their
  confidence as unreliable.
- **Edge silence is trimmed** (VAD-style) before classification — studio
  training clips have none, and leftover silence corrupts energy/pause
  features. Trimmed seconds show as `6.4s speech (of 8.1s)`.
"""

NB1_CODE_18 = """\
# --- Test the trained classifier on your own audio folder ---------------------
TEST_AUDIO_DIR = "/workspace/audio"  # <-- edit me, then re-run this cell
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
SEGMENT_S = 4.0    # classify long files in utterance-sized chunks (like the live VAD)
MAX_FILE_S = 6.0   # files longer than this are segmented and their votes averaged
OOD_Z = 5.0        # |z| beyond this = feature far outside the training distribution

import time
from collections import Counter
from pathlib import Path

import numpy as np

# Re-runnable standalone (e.g. after a kernel restart): reload helpers/weights.
if "WEIGHTS_PATH" not in dir():
    WEIGHTS_PATH = Path(PROJECT_ROOT) / "IMTalker" / "checkpoints" / "prosody_emotion.npz"
if "trainer" not in dir():
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "train_prosody_emotion", Path(PROJECT_ROOT) / "scripts" / "train_prosody_emotion.py")
    trainer = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(trainer)
if "pe" not in dir():
    import sys
    sys.path.insert(0, str(Path(PROJECT_ROOT) / "IMTalker"))
    import prosody_emotion as pe
if "clf" not in dir() or getattr(clf, "backend", None) != "softmax":
    clf = pe.ProsodyEmotionClassifier(weights_path=str(WEIGHTS_PATH))
    assert clf.backend == "softmax", "trained weights not loaded - run the training cell first"

# files that were included in training via --extra_dir (metadata in the npz) -
# predictions on them are training-set results, not generalisation evidence
_trained_on = set()
if WEIGHTS_PATH.is_file():
    with np.load(WEIGHTS_PATH) as _w:
        if "extra_files" in _w.files:
            _trained_on = {str(x) for x in _w["extra_files"]}


def classify_wav(wav):
    # Classify a waveform: edge silence is trimmed first (VAD-style, mirroring
    # the live avatar), files longer than MAX_FILE_S are split into SEGMENT_S
    # chunks and aggregated by mean probability. Also returns the max |z|
    # score and the most out-of-distribution features seen, for diagnostics.
    wav = pe.trim_edges(wav)
    if len(wav) < pe.FRAME_LEN * 2:
        return [], Counter(), 0.0, []
    seg = int(SEGMENT_S * pe.INPUT_SR)
    if len(wav) <= int(MAX_FILE_S * pe.INPUT_SR):
        chunks = [wav]
    else:
        chunks = [wav[i:i + seg] for i in range(0, len(wav), seg)]
        if len(chunks) > 1 and len(chunks[-1]) < seg // 4:
            chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
            chunks.pop()
    mu, sd = clf.model.feat_mean, clf.model.feat_std
    probs_list, votes, max_z, ood = [], Counter(), 0.0, []
    for c in chunks:
        feats = pe.extract_features(c)
        if not feats.any():
            continue
        z = np.nan_to_num((feats - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        az = np.abs(z)
        if float(az.max()) > max_z:
            max_z = float(az.max())
            ood = [(pe.FEATURE_NAMES[i], float(z[i])) for i in np.argsort(-az)[:3]]
        p = clf.model.predict_proba(feats)  # clips z at +/-pe.Z_CLIP internally
        probs_list.append(p)
        votes[int(np.argmax(p))] += 1
    return probs_list, votes, max_z, ood


test_dir = Path(TEST_AUDIO_DIR)
if not test_dir.is_dir():
    print(f"[skip] TEST_AUDIO_DIR does not exist: {test_dir}")
    print("       upload audio there (or edit the path above) and re-run this cell")
else:
    files = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"[warn] no audio files ({', '.join(sorted(AUDIO_EXTS))}) found under {test_dir}")
    else:
        print(f"[test] classifying {len(files)} file(s) from {test_dir} "
              f"(weights: {clf.weights_path})\\n")
        print(f"{'file':<50} {'predicted':<14} {'conf':>5}  top-3  [details]")
        print("-" * 110)
        counts = Counter()
        n_skip = n_err = n_ood = n_extra = 0
        t0 = time.perf_counter()
        for p in files:
            try:
                wav = trainer.load_wav(p)  # mono float32 @ 24 kHz
                dur_raw = len(wav) / pe.INPUT_SR
                wav = pe.trim_edges(wav)   # idempotent; classify_wav re-checks
                probs_list, votes, max_z, ood = classify_wav(wav)
                if not probs_list:
                    n_skip += 1
                    print(f"{p.name[:50]:<50} {'skipped':<14} {'--':>5}  silent or too short")
                    continue
                probs = np.mean(probs_list, axis=0)
                order = np.argsort(probs)[::-1]
                y = int(order[0])
                label = pe.EMOTION_NAMES[y]
                conf = float(probs[y])
                counts[label] += 1
                top3 = ", ".join(f"{pe.EMOTION_NAMES[i]} {probs[i]:.2f}" for i in order[:3])
                dur = len(wav) / pe.INPUT_SR
                detail = (f"{dur:.1f}s speech (of {dur_raw:.1f}s)" if dur_raw - dur > 0.2
                          else f"{dur:.1f}s")
                if len(probs_list) > 1:
                    tally = ", ".join(f"{pe.EMOTION_NAMES[k]}x{v}" for k, v in votes.most_common())
                    detail += f", {len(probs_list)} chunks: {tally}"
                if p.name in _trained_on:
                    n_extra += 1
                    detail += ", in training"
                print(f"{p.name[:50]:<50} {label + f' ({y})':<14} {conf:5.2f}  {top3}  [{detail}]")
                if max_z > OOD_Z:
                    n_ood += 1
                    worst = ", ".join(f"{n} z={v:+.1f}" for n, v in ood if abs(v) > OOD_Z)
                    print(f"{'':<52}[warn] out-of-distribution (max|z|={max_z:.1f}): {worst}")
            except Exception as exc:
                n_err += 1
                print(f"{p.name[:50]:<50} ERROR: {exc}")
        dt = time.perf_counter() - t0
        print("-" * 110)
        summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"
        print(f"[summary] {sum(counts.values())}/{len(files)} classified: {summary}")
        if n_skip or n_err:
            print(f"[info] {n_skip} skipped (silent/short), {n_err} unreadable")
        if n_ood:
            print(f"[warn] {n_ood} file(s) were far outside the RAVDESS training distribution")
            print(f"       (max|z| > {OOD_Z:.0f}); their confidence is unreliable - predictions")
            print(f"       there are clipped to the |z| <= {pe.Z_CLIP:.0f} boundary, not saturated.")
        if n_extra:
            print(f"[info] {n_extra} file(s) marked 'in training' were part of --extra_dir")
            print(f"       training - high accuracy on them is expected, not generalisation")
        if counts:
            print(f"[info] avg {dt / sum(counts.values()) * 1000:.1f} ms/file")
"""


def build_nb1(pe_payload, tr_payload):
    cells = [
        md_cell("prosody-train-00", NB1_MD_00),
        code_cell("prosody-train-01", NB1_CODE_01),
        md_cell("prosody-train-02", NB1_MD_02),
        code_cell("prosody-train-03", nb1_code_03(pe_payload, tr_payload)),
        md_cell("prosody-train-04", NB1_MD_04),
        code_cell("prosody-train-05", NB1_CODE_05),
        md_cell("prosody-train-06", NB1_MD_06),
        code_cell("prosody-train-07", NB1_CODE_07),
        md_cell("prosody-train-08", NB1_MD_08),
        code_cell("prosody-train-09", NB1_CODE_09),
        md_cell("prosody-train-10", NB1_MD_10),
        code_cell("prosody-train-11", NB1_CODE_11),
        md_cell("prosody-train-12", NB1_MD_12),
        code_cell("prosody-train-13", NB1_CODE_13),
        md_cell("prosody-train-14", NB1_MD_14),
        code_cell("prosody-train-15", NB1_CODE_15),
        md_cell("prosody-train-16", NB1_MD_17),
        code_cell("prosody-train-17", NB1_CODE_18),
        md_cell("prosody-train-18", NB1_MD_16),
    ]
    return make_nb(cells)


# ---------------------------------------------------------------------------
# Notebook 2 — runpod_emotion_streaming_prosody.ipynb (surgical edits)
# ---------------------------------------------------------------------------

NB2_MD_PREPEND = """\
# Prosody-emotion streaming variant

This notebook is `runpod_emotion_streaming.ipynb` wired to the **prosody SER
backend** (`prosody-v1`) — pure-numpy, CPU-only, zero model download at runtime.

**Two-notebook workflow**
1. Run `prosody_emotion_train.ipynb` first (same pod): it downloads RAVDESS
   (+ CREMA-D), trains the 6-class softmax, and writes
   `/workspace/prosody_emotion.npz`.
2. Run this notebook: Step 12 picks the trained weights up from
   `PROSODY_WEIGHTS_FILE` automatically. If you trained elsewhere, upload the
   ~7 KB `.npz` to that path (Jupyter file upload) before Step 12.
3. Without the weights file the pipeline still works: it falls back to the
   built-in rule-based prosody classifier (retraining is optional).

---
"""

NB2_CODE_CLONE = """\
# Clone the prosody-emotion repo. When it does not (yet) contain the pipeline
# code (fresh repo), fall back to the previous full-pipeline repo - Step 12
# writes the prosody files (auto_emotion.py / prosody_emotion.py) on top
# either way, so both paths converge to the same runtime.
import os
import shutil
import subprocess

NEW_REPO = "https://github.com/MoshiHead/IMTalker_Personaplex_emotion_v1_add_prosody_emotion.git"
PREV_REPO = "https://github.com/MoshiHead/IMTalker_Personaplex_emotion_v1_auto_emotion_v1.git"

if os.path.isdir("speech2avatar"):
    print("[skip] speech2avatar already exists")
else:
    subprocess.run(["git", "clone", NEW_REPO, "speech2avatar"], check=True)
    if not os.path.isfile("speech2avatar/IMTalker/start_winner_live.sh"):
        print("[warn] new repo does not contain the pipeline yet - cloning the previous repo instead")
        shutil.rmtree("speech2avatar")
        subprocess.run(["git", "clone", PREV_REPO, "speech2avatar"], check=True)
print("[ok] repo ready")
"""

NB2_WEIGHTS_PARAM = """\

# Optional external prosody weights, written by prosody_emotion_train.ipynb.
# When this file exists it takes priority over the (empty) embedded base64 in
# Step 12; otherwise the rule-based fallback classifier is used. Upload the
# ~7 KB .npz to this path if you trained on another machine.
PROSODY_WEIGHTS_FILE = "/workspace/prosody_emotion.npz"
"""

NB2_OLD_WEIGHTS_BLOCK = """\
pe_weights_path = Path(IMTALKER_DIR) / "checkpoints" / "prosody_emotion.npz"
if PROSODY_WEIGHTS_B64:
"""

NB2_NEW_WEIGHTS_BLOCK = """\
pe_weights_path = Path(IMTALKER_DIR) / "checkpoints" / "prosody_emotion.npz"
# Weight priority: external trained file (from prosody_emotion_train.ipynb) >
# base64 embedded in this cell > rule-based fallback in prosody_emotion.py.
_ext_weights = Path(PROSODY_WEIGHTS_FILE) if "PROSODY_WEIGHTS_FILE" in dir() else None
if _ext_weights is not None and _ext_weights.is_file():
    pe_weights_path.parent.mkdir(parents=True, exist_ok=True)
    pe_weights_path.write_bytes(_ext_weights.read_bytes())
    print(f"[written] {pe_weights_path} <- external trained weights ({_ext_weights.stat().st_size} bytes)")
elif PROSODY_WEIGHTS_B64:
"""

NB2_OLD_SMOKE_LINE = "clf = SpeechEmotionClassifier(model_name=sys.argv[2], device=\"cpu\", half=False, disgust_map=0)\n"
NB2_NEW_SMOKE_LINE = (
    NB2_OLD_SMOKE_LINE
    + "if getattr(clf, \"_prosody\", None) is not None:\n"
    + "    print(f\"smoke: prosody backend = {clf._prosody.backend}\")\n"
)

NB2_EDITED_IDS = {"prosody-stream-00", "emotion-stream-01", "emotion-stream-03",
                  "emotion-stream-33", "emotion-stream-35"}


def build_nb2():
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    by_id = {c.get("id"): c for c in cells}
    for cid in ("emotion-stream-01", "emotion-stream-03", "emotion-stream-33", "emotion-stream-35"):
        assert cid in by_id, f"expected cell id {cid} in {SRC_NB.name}"

    # 1. clone cell: new repo with fallback
    by_id["emotion-stream-01"]["source"] = _src(NB2_CODE_CLONE)

    # 2. parameters: new GIT_REPO_URL + PROSODY_WEIGHTS_FILE
    params = "".join(by_id["emotion-stream-03"]["source"])
    params = replace_once(
        params,
        'GIT_REPO_URL = "https://github.com/nash-raf/speech2avatar.git"',
        f'GIT_REPO_URL = "{NEW_REPO}"',
        "GIT_REPO_URL",
    )
    params = replace_once(params, 'SER_DEVICE = "cpu"\n',
                          'SER_DEVICE = "cpu"\n' + NB2_WEIGHTS_PARAM, "PROSODY_WEIGHTS_FILE param")
    by_id["emotion-stream-03"]["source"] = _src(params)

    # 3. Step 12: external weights file takes priority
    step12 = "".join(by_id["emotion-stream-33"]["source"])
    step12 = replace_once(step12, NB2_OLD_WEIGHTS_BLOCK, NB2_NEW_WEIGHTS_BLOCK, "Step 12 weights")
    by_id["emotion-stream-33"]["source"] = _src(step12)

    # 4. Step 13 smoke test: report the prosody backend
    step13 = "".join(by_id["emotion-stream-35"]["source"])
    step13 = replace_once(step13, NB2_OLD_SMOKE_LINE, NB2_NEW_SMOKE_LINE, "Step 13 smoke backend")
    by_id["emotion-stream-35"]["source"] = _src(step13)

    nb["cells"] = [md_cell("prosody-stream-00", NB2_MD_PREPEND)] + cells
    return nb


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _compile_cell(source, cid, tolerate_magics):
    if tolerate_magics:
        lines = [ln for ln in source.splitlines(keepends=True)
                 if not ln.lstrip().startswith(("!", "%"))]
        source = "".join(lines)
    compile(source, f"<cell {cid}>", "exec")


def validate_nb1(nb, pe_payload, tr_payload):
    json.dumps(nb)  # JSON-serializable
    ids = [c["id"] for c in nb["cells"]]
    assert ids == [f"prosody-train-{i:02d}" for i in range(19)], f"unexpected cell ids: {ids}"
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code":
            for ln in src.splitlines():
                assert not ln.lstrip().startswith(("!", "%")), \
                    f"cell {c['id']} uses an IPython magic - NB1 must stay plain python"
            _compile_cell(src, c["id"], tolerate_magics=False)
    cell3 = "".join(nb["cells"][3]["source"])
    assert nb["cells"][3]["id"] == "prosody-train-03"
    assert pe_payload in cell3, "embedded prosody_emotion.py payload missing/altered"
    assert tr_payload in cell3, "embedded train_prosody_emotion.py payload missing/altered"
    for needle in ("REPO_URL", "RAVDESS_URLS", "zenodo.org/records/1188976", "git lfs pull",
                   "speaker-grouped", "WEIGHTS_OUT", "prosody_emotion.npz", "TEST_AUDIO_DIR",
                   "SEGMENT_S", "classify_wav", "max|z|", "EXTRA_AUDIO_DIR", "extra_files",
                   "trim_edges"):
        assert any(needle in "".join(c["source"]) for c in nb["cells"]), f"missing: {needle}"
    print(f"[ok] NB1 validated: {len(nb['cells'])} cells, all code cells compile, embeds byte-identical")


def validate_nb2(nb):
    json.dumps(nb)
    orig = json.loads(SRC_NB.read_text(encoding="utf-8"))
    orig_by_id = {c.get("id"): c for c in orig["cells"]}
    assert len(nb["cells"]) == len(orig["cells"]) + 1, "NB2 must be orig + 1 prepended cell"
    n_unchanged = 0
    for c in nb["cells"]:
        cid = c.get("id")
        if cid in NB2_EDITED_IDS:
            continue
        assert cid in orig_by_id, f"unexpected new cell id: {cid}"
        assert c["source"] == orig_by_id[cid]["source"], f"cell {cid} drifted from the source notebook"
        n_unchanged += 1
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            _compile_cell("".join(c["source"]), c.get("id"), tolerate_magics=True)
    full = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in (NEW_REPO, PREV_REPO, "PROSODY_WEIGHTS_FILE", "_ext_weights",
                   "prosody backend = ", 'SER_MODEL = "prosody-v1"', 'SER_DEVICE = "cpu"'):
        assert needle in full, f"missing: {needle}"
    print(f"[ok] NB2 validated: {len(nb['cells'])} cells ({n_unchanged} byte-identical to source, "
          f"{len(NB2_EDITED_IDS)} edited), all code cells compile")


# ---------------------------------------------------------------------------
# dry run — execute NB1's core (non-network) cells end-to-end on synthetic data
# ---------------------------------------------------------------------------

def dryrun_nb1(nb1, pe_payload, tr_payload):
    import numpy as np

    print("\n[dryrun] executing NB1 core cells against a synthetic dataset ...")
    tmp = Path(tempfile.mkdtemp(prefix="prosody_nb1_dryrun_"))
    proj = tmp / "proj"
    proj.mkdir()
    (proj / "README.md").write_text("dummy checkout without prosody files\n", encoding="utf-8")
    cells = {c["id"]: "".join(c["source"]) for c in nb1["cells"] if c["cell_type"] == "code"}

    ns = {}
    params = (cells["prosody-train-01"]
              .replace("/workspace/speech2avatar", proj.as_posix())
              .replace("/workspace/prosody_data", (tmp / "data").as_posix())
              .replace("/workspace/prosody_emotion.npz", (tmp / "prosody_emotion.npz").as_posix()))
    exec(compile(params, "<cell-01>", "exec"), ns)

    exec(compile(cells["prosody-train-03"], "<cell-03>", "exec"), ns)  # self-heal write path
    assert (proj / "IMTalker" / "prosody_emotion.py").read_text(encoding="utf-8") == pe_payload
    assert (proj / "scripts" / "train_prosody_emotion.py").read_text(encoding="utf-8") == tr_payload
    print("[dryrun] self-heal wrote byte-identical embedded files")

    exec(compile(cells["prosody-train-05"], "<cell-05>", "exec"), ns)  # deps

    # Synthetic RAVDESS: 2 actors x 8 emotion codes (incl. disgust, which the
    # mapper skips) — enough for the scanner, audit, and a 2-fold training run.
    try:
        import soundfile as sf
        write_wav = lambda p, x, sr: sf.write(str(p), x, sr)
    except ImportError:
        from scipy.io import wavfile as _wavfile
        write_wav = lambda p, x, sr: _wavfile.write(str(p), sr, (x * 32767).astype("int16"))
    rav = Path(ns["DATA_DIR"]) / "ravdess"
    sr = 16000
    t = np.arange(int(sr * 1.2)) / sr
    for actor in (1, 2):
        adir = rav / f"Actor_{actor:02d}"
        adir.mkdir(parents=True)
        for emo in range(1, 9):
            f0 = 90 + emo * 25 + actor * 8
            x = (0.2 * np.sin(2 * np.pi * f0 * t)
                 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")
            write_wav(adir / f"03-01-{emo:02d}-01-01-01-{actor:02d}.wav", x, sr)

    ns["RAVDESS_DIR"] = rav
    ns["cremad_ok"] = False
    exec(compile(cells["prosody-train-11"], "<cell-11>", "exec"), ns)  # audit

    # probe folder: two emotion-named files (join training via --extra_dir) plus
    # unnamed robustness probes (gain pair, 20 s file)
    probe_dir = tmp / "probe_audio"
    probe_dir.mkdir()
    t3 = np.arange(int(sr * 3.0)) / sr
    base = (0.2 * np.sin(2 * np.pi * 140 * t3)
            * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t3))).astype("float32")
    t20 = np.arange(int(sr * 20.0)) / sr
    long20 = (0.2 * np.sin(2 * np.pi * 175 * t20)
              * (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t20))).astype("float32")
    write_wav(probe_dir / "happy-probe.wav", 0.25 * np.sin(2 * np.pi * 240 * t3), sr)
    write_wav(probe_dir / "sad-probe.wav", 0.25 * np.sin(2 * np.pi * 95 * t3), sr)
    write_wav(probe_dir / "probe_quiet.wav", base, sr)
    write_wav(probe_dir / "probe_loud.wav", base * 3.0, sr)
    write_wav(probe_dir / "probe_long20s.wav", long20, sr)
    write_wav(probe_dir / "probe_padded.wav",
              np.concatenate([np.zeros(int(0.8 * sr), np.float32), base,
                              np.zeros(int(1.6 * sr), np.float32)]), sr)

    # Train via the CLI on the synthetic data (fast, honest end-to-end run,
    # exercising --extra_dir and the default-on noise augmentation).
    weights = proj / "IMTalker" / "checkpoints" / "prosody_emotion.npz"
    weights.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(proj / "scripts" / "train_prosody_emotion.py"),
         "--ravdess_dir", str(rav), "--extra_dir", str(probe_dir),
         "--out", str(weights), "--folds", "2", "--seed", "0"])
    assert proc.returncode == 0, "dryrun training failed"
    assert weights.is_file() and weights.stat().st_size > 1000

    ns["WEIGHTS_PATH"] = weights
    exec(compile(cells["prosody-train-15"], "<cell-15>", "exec"), ns)  # validate + hand off
    assert Path(ns["WEIGHTS_OUT"]).is_file(), "hand-off copy missing"

    # Step 8 — batch-test cell against the synthetic dataset folder
    test_src = cells["prosody-train-17"].replace('"/workspace/audio"', f'"{rav.as_posix()}"')
    exec(compile(test_src, "<cell-17>", "exec"), ns)
    print("[dryrun] Step 8 batch-test cell executed against the synthetic folder")

    # Step 8 robustness: a 20 s file must be segmented into chunks, the
    # emotion-named probe files must be flagged 'in training', and the same
    # utterance 9.5 dB louder must predict the same emotion (the
    # loudness-collapse regression guard).
    import contextlib
    import io

    test_src2 = cells["prosody-train-17"].replace('"/workspace/audio"', f'"{probe_dir.as_posix()}"')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(test_src2, "<cell-17b>", "exec"), ns)
    out = buf.getvalue()
    assert "chunks:" in out, f"20 s file was not segmented by the Step 8 cell:\n{out}"
    assert "in training" in out, f"--extra_dir files not flagged as in-training:\n{out}"
    assert "(of " in out, f"padded file was not reported as trimmed:\n{out}"

    def _pred(path):
        wav = ns["trainer"].load_wav(path)
        probs_list, votes, max_z, ood = ns["classify_wav"](wav)
        return int(np.argmax(np.mean(probs_list, axis=0)))

    y_quiet = _pred(probe_dir / "probe_quiet.wav")
    y_loud = _pred(probe_dir / "probe_loud.wav")
    assert y_quiet == y_loud, f"gain invariance broken: quiet {y_quiet} vs loud {y_loud}"
    y_padded = _pred(probe_dir / "probe_padded.wav")
    assert y_padded == y_quiet, f"edge-silence invariance broken: padded {y_padded} vs clean {y_quiet}"
    print("[dryrun] Step 8 robustness: 20 s file segmented; gain- and edge-silence-invariant predictions")

    print(f"[dryrun] OK - synthetic end-to-end run passed ({tmp})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dryrun", action="store_true",
                    help="also execute NB1's core cells end-to-end on synthetic data")
    args = ap.parse_args()

    assert SRC_NB.is_file(), f"source notebook missing: {SRC_NB}"
    pe_payload, tr_payload = load_payloads()

    nb1 = build_nb1(pe_payload, tr_payload)
    validate_nb1(nb1, pe_payload, tr_payload)
    NB1_PATH.write_text(json.dumps(nb1, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[written] {NB1_PATH.name}")

    nb2 = build_nb2()
    validate_nb2(nb2)
    NB2_PATH.write_text(json.dumps(nb2, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[written] {NB2_PATH.name}")

    if args.dryrun:
        dryrun_nb1(nb1, pe_payload, tr_payload)

    print("\n[done] both notebooks generated and validated")


if __name__ == "__main__":
    main()

