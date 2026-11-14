#!/usr/bin/env python3
"""Functional tests for IMTalker/prosody_emotion.py and its wiring into
auto_emotion.py (PROSODY_SER_V1). Pure numpy - no torch, no funasr, no GPU.

Covers: feature extraction (determinism, finiteness, F0 accuracy, degenerate
inputs), the rule-based fallback, the trained-softmax save/load roundtrip,
the SpeechEmotionClassifier prosody routing, end-to-end AutoEmotionManager
plumbing with ser_model="prosody-v1", and the per-utterance latency budget.
"""

import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "IMTalker"))

import prosody_emotion as pe  # noqa: E402
from auto_emotion import AutoEmotionManager, SpeechEmotionClassifier  # noqa: E402

SR = 24_000
FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def synth(f0_mean=220.0, f0_mod_hz=3.0, f0_mod_depth=30.0, amp=0.2,
          syll_hz=4.0, seconds=3.0, seed=0):
    """Synthetic voiced utterance: harmonic stack, vibrato, syllable pulsing."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = f0_mean + f0_mod_depth * np.sin(2 * np.pi * f0_mod_hz * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros_like(t)
    for h, a in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        sig += a * np.sin(h * phase)
    env = amp * np.clip(0.55 + 0.45 * np.sin(2 * np.pi * syll_hz * t), 0.05, 1.0)
    sig = env * sig + 0.002 * rng.standard_normal(t.shape[0])
    return sig.astype(np.float32)


# --- 1. feature layout --------------------------------------------------------
check("FEATURE_DIM matches FEATURE_NAMES", pe.FEATURE_DIM == len(pe.FEATURE_NAMES))
check("feature names are unique", len(set(pe.FEATURE_NAMES)) == len(pe.FEATURE_NAMES))

# --- 2. determinism + finiteness ----------------------------------------------
sig = synth()
f1 = pe.extract_features(sig)
f2 = pe.extract_features(sig)
check("feature extraction is deterministic", np.array_equal(f1, f2))
check("features are finite on voiced speech", np.isfinite(f1).all())
check("feature vector shape", f1.shape == (pe.FEATURE_DIM,), f"{f1.shape}")

for name, x in [
    ("white noise", 0.1 * np.random.default_rng(1).standard_normal(SR * 2).astype(np.float32)),
    ("silence", np.zeros(SR * 2, dtype=np.float32)),
    ("clipped loud", np.clip(synth(amp=2.0), -1, 1)),
    ("very short (10 ms)", np.zeros(int(0.01 * SR), dtype=np.float32)),
    ("constant DC", np.full(SR, 0.5, dtype=np.float32)),
]:
    f = pe.extract_features(x)
    check(f"finite features on {name}", np.isfinite(f).all())

# --- 3. degenerate inputs -> zero vector -> neutral/0.0 ------------------------
check("silence -> all-zero features", not pe.extract_features(np.zeros(SR, np.float32)).any())
check("short input -> all-zero features",
      not pe.extract_features(np.zeros(int(0.01 * SR), np.float32)).any())
clf = pe.ProsodyEmotionClassifier(verbose=False)
check("silence -> neutral with 0 confidence",
      clf.predict(np.zeros(SR, np.float32)) == (0, "neutral", 0.0))

# --- 4. F0 accuracy ------------------------------------------------------------
for target in (120.0, 220.0, 300.0):
    f = pe.extract_features(synth(f0_mean=target, f0_mod_depth=0.0))
    got = float(f[pe.FEATURE_NAMES.index("f0_mean")])
    check(f"F0 estimate {target:.0f} Hz within 10%", abs(got - target) / target < 0.10,
          f"got {got:.1f} Hz")

# --- 5. rule-based fallback: valid outputs + sensible ordering -----------------
check("fallback backend when no npz exists", clf.backend == "rules", clf.backend)
excited = synth(f0_mean=260.0, f0_mod_depth=70.0, amp=0.30, syll_hz=5.0, seed=2)
calm = synth(f0_mean=95.0, f0_mod_depth=6.0, amp=0.05, syll_hz=2.0, seed=3)
eid_e, lab_e, conf_e = clf.predict(excited)
eid_c, lab_c, conf_c = clf.predict(calm)
check("rule prediction ids in range", 0 <= eid_e <= 5 and 0 <= eid_c <= 5,
      f"{eid_e}, {eid_c}")
check("rule confidences in (0, 1]", 0.0 < conf_e <= 1.0 and 0.0 < conf_c <= 1.0,
      f"{conf_e:.3f}, {conf_c:.3f}")
probs_e = clf.model.predict_proba(pe.extract_features(excited))
probs_c = clf.model.predict_proba(pe.extract_features(calm))
high_arousal = (1, 3, 4, 5)   # happy / angry / surprise / fear (all high-arousal)
low_arousal = (0, 2)          # neutral / sad
check("excited signal scores higher on high-arousal emotions",
      probs_e[list(high_arousal)].sum() > probs_c[list(high_arousal)].sum(),
      f"{probs_e[list(high_arousal)].sum():.2f} vs {probs_c[list(high_arousal)].sum():.2f}")
check("calm signal scores higher on low-arousal emotions",
      probs_c[list(low_arousal)].sum() > probs_e[list(low_arousal)].sum(),
      f"{probs_c[list(low_arousal)].sum():.2f} vs {probs_e[list(low_arousal)].sum():.2f}")
print(f"       rule labels: excited -> {lab_e} (p={conf_e:.2f}), calm -> {lab_c} (p={conf_c:.2f})")


# --- 6. trained-softmax save/load roundtrip ------------------------------------
# Linearly separable synthetic dataset: two Gaussian blobs per class region.
rng = np.random.default_rng(0)
n_per, D = 30, pe.FEATURE_DIM
X, y = [], []
centres = rng.normal(0, 2.5, size=(pe.NUM_EMOTIONS, D))
for c in range(pe.NUM_EMOTIONS):
    X.append(centres[c] + 0.5 * rng.standard_normal((n_per, D)))
    y += [c] * n_per
X = np.concatenate(X).astype(np.float64)
y = np.array(y)
mu, sd = X.mean(0), X.std(0) + 1e-6
Z = (X - mu) / sd
W = np.zeros((pe.NUM_EMOTIONS, D))
b = np.zeros(pe.NUM_EMOTIONS)
Y = np.eye(pe.NUM_EMOTIONS)[y]
lr = 0.5
for it in range(400):
    logits = Z @ W.T + b
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    G = (P - Y).T @ Z / len(Z) + 1e-4 * W
    W -= lr * G
    b -= lr * (P - Y).mean(0)

tmp_dir = Path(tempfile.mkdtemp(prefix="prosody_weights_"))
wpath = tmp_dir / "prosody_emotion.npz"
pe.save_weights(wpath, W, b, mu, sd, temperature=1.0)
check("weight file written", wpath.is_file() and wpath.stat().st_size < 100_000,
      f"{wpath.stat().st_size} bytes")
clf2 = pe.ProsodyEmotionClassifier(weights_path=wpath, verbose=False)
check("trained backend selected when npz is given", clf2.backend == "softmax", clf2.backend)
model = pe.load_weights(wpath)
Zt = (X - mu) / sd
preds = np.array([np.argmax(model.predict_proba(row)) for row in Zt])
train_acc = float((preds == y).mean())
check("trained softmax separates the synthetic blobs", train_acc > 0.95, f"{train_acc:.2%}")

# corrupted layout -> ValueError
bad = tmp_dir / "bad.npz"
np.savez(bad, W=W, b=b, feat_mean=mu, feat_std=sd,
         feature_names=np.array(["wrong"] * D), temperature=np.float32(1.0),
         format_version=np.int32(pe.WEIGHTS_FORMAT_VERSION))
try:
    pe.load_weights(bad)
    check("layout-mismatched npz raises ValueError", False)
except ValueError:
    check("layout-mismatched npz raises ValueError", True)

# legacy format-v1 weight file (pre loudness normalisation) -> clear retrain
# error, and the classifier degrades to the rule fallback instead of crashing
v1 = tmp_dir / "v1.npz"
np.savez(v1, W=W, b=b, feat_mean=mu, feat_std=sd,
         feature_names=np.array(pe.FEATURE_NAMES), temperature=np.float32(1.0),
         format_version=np.int32(1))
try:
    pe.load_weights(v1)
    check("format v1 weights are rejected with a retrain hint", False)
except ValueError as exc:
    check("format v1 weights are rejected with a retrain hint", "format v1" in str(exc),
          str(exc)[:80])
clf_v1 = pe.ProsodyEmotionClassifier(weights_path=v1, verbose=False)
check("v1 weights degrade to the rule fallback (no crash)",
      clf_v1.backend == "rules", clf_v1.backend)
v2 = tmp_dir / "v2.npz"
np.savez(v2, W=W, b=b, feat_mean=mu, feat_std=sd,
         feature_names=np.array(pe.FEATURE_NAMES), temperature=np.float32(1.0),
         format_version=np.int32(2))
try:
    pe.load_weights(v2)
    check("format v2 weights are also rejected (v3 = edge-trimmed)", False)
except ValueError as exc:
    check("format v2 weights are also rejected (v3 = edge-trimmed)",
          "format v2" in str(exc), str(exc)[:60])

# --- 6d. VAD edge trim (the edge-silence regression guard) ---------------------
# Studio training clips are tightly trimmed; real recordings carry 0.5-2 s of
# edge silence that corrupted duration/rms_db/slope features and flipped
# predictions (reproduced experimentally with exactly this padding).
sig_trim = synth(f0_mean=180.0, f0_mod_depth=40.0, amp=0.3, syll_hz=4.0, seed=6)
padded = np.concatenate([np.zeros(int(0.6 * SR), np.float32), sig_trim,
                         np.zeros(int(1.8 * SR), np.float32)])
trimmed = pe.trim_edges(padded)
check("edge trim removes padding (within 0.1 s)",
      abs(len(trimmed) - len(sig_trim)) <= int(0.1 * SR),
      f"{len(sig_trim)} vs {len(trimmed)} samples")
check("edge trim is idempotent", len(pe.trim_edges(trimmed)) == len(trimmed))
check("all-silent input trims to empty", pe.trim_edges(np.zeros(SR)).size == 0)
lab_clean = pe.EMOTION_NAMES[int(np.argmax(pe._RULE_MODEL.predict_proba(
    pe.extract_features(sig_trim))))]
lab_pad = pe.EMOTION_NAMES[int(np.argmax(pe._RULE_MODEL.predict_proba(
    pe.extract_features(padded))))]
check("edge silence no longer flips the rule prediction", lab_clean == lab_pad,
      f"{lab_clean} vs {lab_pad}")
dur_clean = pe.extract_features(sig_trim)[pe.FEATURE_NAMES.index("duration")]
dur_pad = pe.extract_features(padded)[pe.FEATURE_NAMES.index("duration")]
check("duration feature is silence-invariant after trim",
      abs(dur_clean - dur_pad) < 0.1, f"{dur_clean:.2f} vs {dur_pad:.2f}")

# --- 6b. v2 loudness normalisation (the all-angry regression guard) ------------
# The same utterance at a different recording level must yield (nearly) the
# same features and the same prediction; before v2 a level change pushed the
# level-dependent features far out of distribution and collapsed every input
# to one class.
sig_quiet = synth(f0_mean=180.0, f0_mod_depth=40.0, amp=0.05, syll_hz=4.0, seed=5)
f_quiet = pe.extract_features(sig_quiet)
f_loud = pe.extract_features(sig_quiet * 6.0)   # +15.6 dB, same utterance
check("loudness normalisation makes features gain-invariant",
      np.allclose(f_quiet, f_loud, rtol=5e-2, atol=5e-3),
      f"max abs diff {np.abs(f_quiet - f_loud).max():.4f}")
lab_q = pe.EMOTION_NAMES[int(np.argmax(pe._RULE_MODEL.predict_proba(f_quiet)))]
lab_l = pe.EMOTION_NAMES[int(np.argmax(pe._RULE_MODEL.predict_proba(f_loud)))]
check("gain change no longer flips the rule prediction", lab_q == lab_l,
      f"{lab_q} vs {lab_l}")

# --- 6c. z-score clipping winsorises absurd features ---------------------------
p5 = model.predict_proba((mu + 5.5 * sd).astype(np.float32))
p50 = model.predict_proba((mu + 50.0 * sd).astype(np.float32))
check("z-clipping winsorises absurd features to the clip boundary",
      np.allclose(p5, p50, atol=1e-5), f"max diff {np.abs(p5 - p50).max():.2e}")
check("z-clipped probs stay valid on absurd input",
      np.isfinite(p50).all() and abs(float(p50.sum()) - 1.0) < 1e-4,
      f"sum {p50.sum():.4f}")

# --- 7. SpeechEmotionClassifier routing -----------------------------------------
ser = SpeechEmotionClassifier("prosody-v1", device="cpu", half=False, disgust_map=0)
check("prosody backend routed inside SpeechEmotionClassifier", ser._prosody is not None)
check("prosody backend needs no torch model", ser.model is None and ser.half is False)
eid, label, conf = ser.predict(excited)
check("SER predict contract (id, label, conf)",
      0 <= eid <= 5 and label in pe.EMOTION_NAMES.values() and 0.0 <= conf <= 1.0,
      f"{eid}, {label}, {conf:.3f}")
check("emotion2vec names still route to the HF backend (not prosody)",
      SpeechEmotionClassifier._is_prosody(
          types.SimpleNamespace(model_name="emotion2vec/emotion2vec_plus_base")) is False)

# --- 8. end-to-end: AutoEmotionManager with the prosody backend ------------------
os.environ["IMTALKER_EMOTION_LOG_DIR"] = str(tmp_dir)
args = types.SimpleNamespace(
    emotion_id=0, ser_model="prosody-v1", ser_device="cpu", ser_half=False,
    ser_confidence_threshold=0.0, ser_disgust_map=0,
    vad_rms_threshold=0.02, vad_silence_sec=0.7,
    vad_min_speech_sec=0.5, vad_max_utterance_sec=12.0,
)
mgr = AutoEmotionManager(args)
mgr.start()  # loads the prosody classifier + starts the worker thread
check("manager started with the prosody classifier",
      isinstance(mgr._classifier, SpeechEmotionClassifier)
      and mgr._classifier._prosody is not None)

pcm16 = (np.concatenate([np.zeros(int(0.2 * SR), np.float32), excited,
                         np.zeros(int(1.0 * SR), np.float32)]) * 32768).astype(np.int16)
chunk = int(0.02 * SR)
feed_times = []
for i in range(0, len(pcm16), chunk):
    t0 = time.perf_counter()
    mgr.feed(pcm16[i:i + chunk], SR)
    feed_times.append(time.perf_counter() - t0)
check("feed() hot path stays in the microsecond range",
      np.mean(feed_times) < 5e-3, f"mean {np.mean(feed_times) * 1e3:.3f} ms")

deadline = time.time() + 10.0
while mgr.utterances_classified < 1 and time.time() < deadline:
    time.sleep(0.05)
check("prosody worker classified exactly one utterance", mgr.utterances_classified == 1,
      f"{mgr.utterances_classified}")
check("no feed errors in the end-to-end run", mgr.feed_errors == 0)
check("published emotion id is valid", 0 <= mgr.current_emotion_id() <= 5,
      f"{mgr.current_emotion_id()}")
mgr.stop()
os.environ.pop("IMTALKER_EMOTION_LOG_DIR", None)

# --- 9. latency budget ------------------------------------------------------------
clf3 = pe.ProsodyEmotionClassifier(verbose=False)
for _ in range(3):
    clf3.predict(excited)
t0 = time.perf_counter()
reps = 30
for _ in range(reps):
    clf3.predict(excited)
avg_ms = (time.perf_counter() - t0) / reps * 1e3
# Loose CI bound (slow laptops); a server CPU lands at ~3-8 ms.
check("prosody predict latency within budget", avg_ms < 150.0, f"{avg_ms:.1f} ms/utterance")

import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL PROSODY SER TESTS PASSED")

