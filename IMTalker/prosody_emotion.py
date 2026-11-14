"""Lightweight prosody-based speech emotion detection (PROSODY_SER_V1).

Replaces the emotion2vec+/funasr SER backend with a from-scratch, pure-numpy
pipeline that needs no pretrained model, no GPU and no downloads:

  utterance (24 kHz float32 mono, from auto_emotion.EnergyVAD)
    -> 25 ms frames / 20 ms hop (Hann windowed)
    -> per-frame contours: RMS(dB), F0 (autocorrelation), ZCR, spectral
       centroid / rolloff(85%) / flatness / flux, 13 MFCCs (precomputed mel
       filterbank + DCT)
    -> utterance-level statistics per contour (mean/std/min/max/range/slope)
       plus voiced-ratio, syllable rate, pause ratio, jitter, shimmer,
       duration and MFCC1-4 mean/std  ->  ~56-dim feature vector
    -> z-score normalise -> 6-class softmax linear model (one matrix
       multiply) -> (emotion_id, label, confidence)

Cost: ~2-5 ms CPU per utterance, zero VRAM, zero torch.

Classifier contract (identical to auto_emotion.SpeechEmotionClassifier.predict
so the manager / server patches / notebook smoke test work unchanged):

    predict(pcm_24k) -> (emotion_id: int 0-5, label: str, confidence: float)

Two backends share the same feature vector and softmax machinery:
  * "softmax" - weights trained from scratch on RAVDESS (+ optionally
    CREMA-D) by scripts/train_prosody_emotion.py and stored in a ~10 KB
    .npz (weights, bias, feature mean/std, feature names, temperature).
  * "rules"   - hand-crafted z-space weight matrix + population priors,
    used automatically when no .npz is found. Works for any speaker at
    moderate accuracy; the trained model replaces it transparently.

Weight-file search order: explicit path -> $PROSODY_EMOTION_WEIGHTS ->
<IMTalker>/checkpoints/prosody_emotion.npz -> <IMTalker>/prosody_emotion.npz
-> rule-based fallback (with a one-time guidance print).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Emotion ids must match generator/FM.py EMOTION_MAP and auto_emotion.py.
EMOTION_NAMES = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "surprise", 5: "fear"}
NUM_EMOTIONS = 6

INPUT_SR = 24_000          # matches auto_emotion.TARGET_SR (server mic PCM rate)
FRAME_LEN = 600            # 25 ms @ 24 kHz
HOP = 480                  # 20 ms hop (utterance-level stats need no finer grid;
                           # halves the FFT count vs 10 ms, aligns with the VAD frame)
NFFT = 2048                # zero-padded rFFT, shared by the spectral features and
                           # the autocorrelation pitch track (one FFT serves both)
F0_MIN_HZ = 60.0           # ACF search range
F0_MAX_HZ = 400.0
LAG_MIN = int(round(INPUT_SR / F0_MAX_HZ))   # 60
LAG_MAX = int(round(INPUT_SR / F0_MIN_HZ))   # 400
VOICED_ACF_MIN = 0.35      # normalised-ACF peak strength to call a frame voiced
NUM_MEL = 26
NUM_MFCC = 13
MEL_FMAX = 8000.0
SILENCE_DB = -60.0         # whole-utterance guard: below this -> no speech
TARGET_RMS = 0.1           # loudness-normalisation target (~-20 dBFS RMS):
                           # recording level is NOT an emotion cue. Without this
                           # normalisation the level-dependent features
                           # (rms_db_*, mfcc1_*, shimmer) go out-of-distribution
                           # for any non-studio-level input and the softmax
                           # collapses to one class for every file.
RMS_SILENCE = 10.0 ** (SILENCE_DB / 20.0)  # -60 dBFS utterance-RMS guard
Z_CLIP = 5.0               # winsorise standardised features: legit training
                           # variation is |z| <~ 3.5; clipping bounds the logit
                           # contribution of any out-of-distribution feature so
                           # it can never saturate the softmax (conf 1.00 on
                           # every input) again. In-distribution predictions
                           # are untouched.
WEIGHTS_FORMAT_VERSION = 2  # v2 = loudness-normalised features; v1 weights are
                           # incompatible (different feature semantics) and are
                           # rejected with a retrain hint.

# ---------------------------------------------------------------------------
# Feature layout (kept in ONE place; the trainer validates against this list)
# ---------------------------------------------------------------------------

_STAT_CONTOURS = ("rms_db", "f0", "zcr", "centroid", "rolloff", "flatness", "flux")
_STAT_NAMES = ("mean", "std", "min", "max", "range", "slope")
_EXTRA_NAMES = ("voiced_ratio", "syll_rate", "pause_ratio", "jitter", "shimmer", "duration")
_MFCC_STAT_NAMES = tuple(f"mfcc{k}_{s}" for k in (1, 2, 3, 4) for s in ("mean", "std"))

FEATURE_NAMES = (
    tuple(f"{c}_{s}" for c in _STAT_CONTOURS for s in _STAT_NAMES)
    + _EXTRA_NAMES
    + _MFCC_STAT_NAMES
)
FEATURE_DIM = len(FEATURE_NAMES)
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}



# ---------------------------------------------------------------------------
# Lazy precomputed tables (built once per process)
# ---------------------------------------------------------------------------

_TABLES: dict = {}


def _tables() -> dict:
    if _TABLES:
        return _TABLES
    win = np.hanning(FRAME_LEN).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / INPUT_SR).astype(np.float64)

    # Mel filterbank (linearly spaced in mel, triangular in Hz)
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(0.0), hz_to_mel(MEL_FMAX), NUM_MEL + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((NFFT + 1) * hz_pts / INPUT_SR).astype(int)
    mel_fb = np.zeros((NUM_MEL, NFFT // 2 + 1), dtype=np.float32)
    for m in range(NUM_MEL):
        lo, ctr, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        if ctr <= lo:
            ctr = lo + 1
        if hi <= ctr:
            hi = ctr + 1
        for k in range(lo, min(ctr, mel_fb.shape[1])):
            mel_fb[m, k] = (k - lo) / float(ctr - lo)
        for k in range(ctr, min(hi, mel_fb.shape[1])):
            mel_fb[m, k] = (hi - k) / float(hi - ctr)

    # DCT-II matrix for MFCCs
    n = np.arange(NUM_MEL)
    k = np.arange(NUM_MFCC)[:, None]
    dct = np.cos(np.pi / NUM_MEL * (n + 0.5) * k).astype(np.float32)

    _TABLES.update(win=win, freqs=freqs, mel_fb=mel_fb, dct=dct)
    return _TABLES


# ---------------------------------------------------------------------------
# Feature extraction (pure numpy, vectorised; ~2-5 ms per 3 s utterance)
# ---------------------------------------------------------------------------

def _contour_stats(vals: dict, name: str, v: np.ndarray, times: np.ndarray) -> None:
    """Closed-form mean/std/min/max/range/slope(per-second) for one contour."""
    m = int(v.shape[0])
    if m < 2:
        for s in _STAT_NAMES:
            vals[f"{name}_{s}"] = 0.0
        return
    v = np.asarray(v, dtype=np.float64)
    t = times[:m].astype(np.float64)
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    slope = float(np.dot(tc, v - v.mean()) / denom) if denom > 0 else 0.0
    vals[f"{name}_mean"] = float(v.mean())
    vals[f"{name}_std"] = float(v.std())
    vals[f"{name}_min"] = float(v.min())
    vals[f"{name}_max"] = float(v.max())
    vals[f"{name}_range"] = float(v.max() - v.min())
    vals[f"{name}_slope"] = slope


def extract_features(pcm: np.ndarray) -> np.ndarray:
    """24 kHz float32 mono utterance -> FEATURE_NAMES-ordered feature vector.

    Returns an all-zero vector for degenerate input (too short / fully
    silent); callers treat that as "no prediction possible".
    """
    tb = _tables()
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    n = int(x.shape[0])
    zeros = np.zeros(FEATURE_DIM, dtype=np.float32)
    if n < FRAME_LEN * 2:
        return zeros

    # Utterance-level silence guard (un-normalised signal), then loudness
    # normalisation so rms_db/mfcc/shimmer features are level-invariant.
    u_rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    if u_rms < RMS_SILENCE:
        return zeros
    x = (x * (TARGET_RMS / u_rms)).astype(np.float32)

    sw = np.lib.stride_tricks.sliding_window_view(x, FRAME_LEN)
    frames = sw[::HOP]                      # (n_frames, FRAME_LEN), a view
    nf = int(frames.shape[0])
    if nf < 2:
        return zeros
    fw = frames * tb["win"]

    # --- energy -------------------------------------------------------------
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
    rms_db = 20.0 * np.log10(rms + 1e-8)
    if float(rms_db.max()) < SILENCE_DB:
        return zeros

    # --- spectrum (one rFFT shared by the spectral features and the F0 ACF) --
    A = np.fft.rfft(fw, n=NFFT, axis=1)                      # (nf, 1025)
    mag = np.abs(A)
    pow_ = np.square(mag)
    msum = mag.sum(axis=1) + 1e-10
    centroid = (mag @ tb["freqs"]) / msum
    csum = np.cumsum(mag, axis=1)
    roll_idx = (csum >= (0.85 * msum)[:, None]).argmax(axis=1)
    rolloff = tb["freqs"][roll_idx]
    flatness = np.exp(np.mean(np.log(mag + 1e-10), axis=1)) / (np.mean(mag, axis=1) + 1e-10)
    dmag = np.diff(mag, axis=0)
    flux = np.concatenate([[0.0], np.sqrt(np.mean(np.square(dmag), axis=1))])
    zcr = np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1)

    # --- F0 via normalised autocorrelation (vectorised over frames) ---------
    acf = np.fft.irfft(pow_, n=NFFT, axis=1)[:, :LAG_MAX + 1]
    acf_n = acf / (acf[:, :1] + 1e-10)
    seg = acf_n[:, LAG_MIN:LAG_MAX + 1]
    peak_lag = seg.argmax(axis=1) + LAG_MIN
    peak_val = seg.max(axis=1)
    voiced = (peak_val > VOICED_ACF_MIN) & (rms_db > (float(rms_db.mean()) - 10.0))
    f0 = np.where(voiced, INPUT_SR / peak_lag.astype(np.float32), 0.0)
    f0v = f0[f0 > 0.0]

    # --- MFCCs ---------------------------------------------------------------
    mel_pow = tb["mel_fb"] @ pow_.T                          # (26, nf)
    mfcc = tb["dct"] @ np.log(mel_pow + 1e-10)               # (13, nf)

    # --- utterance-level statistics ------------------------------------------
    times = np.arange(nf, dtype=np.float32) * (HOP / INPUT_SR)
    vals: dict = {}
    _contour_stats(vals, "rms_db", rms_db, times)
    _contour_stats(vals, "f0", f0v, times)        # voiced frames only
    _contour_stats(vals, "zcr", zcr, times)
    _contour_stats(vals, "centroid", centroid, times)
    _contour_stats(vals, "rolloff", rolloff, times)
    _contour_stats(vals, "flatness", flatness, times)
    _contour_stats(vals, "flux", flux, times)

    duration = n / float(INPUT_SR)
    vals["voiced_ratio"] = float(f0v.shape[0]) / float(nf)

    # syllable rate: peaks of the smoothed RMS contour (>=120 ms apart)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    rms_s = np.convolve(rms, kernel, mode="same")
    thresh = float(rms_s.mean() + 0.5 * rms_s.std())
    cand = np.where(
        (rms_s[1:-1] > thresh)
        & (rms_s[1:-1] >= rms_s[:-2])
        & (rms_s[1:-1] > rms_s[2:])
    )[0] + 1
    min_dist = max(1, int(round(0.12 / (HOP / INPUT_SR))))
    peaks: list = []
    last = -min_dist
    for i in cand:
        i = int(i)
        if i - last >= min_dist:
            peaks.append(i)
            last = i
        elif peaks and rms_s[i] > rms_s[peaks[-1]]:
            peaks[-1] = i
            last = i
    vals["syll_rate"] = float(len(peaks)) / max(duration, 1e-6)

    vals["pause_ratio"] = float(np.mean(rms_db < (float(rms_db.mean()) - 8.0)))
    if f0v.shape[0] >= 3:
        vals["jitter"] = float(np.abs(np.diff(f0v)).mean() / (f0v.mean() + 1e-6))
    else:
        vals["jitter"] = 0.0
    vals["shimmer"] = float(np.abs(np.diff(rms)).mean() / (rms.mean() + 1e-8))
    vals["duration"] = float(duration)
    for k in (1, 2, 3, 4):
        vals[f"mfcc{k}_mean"] = float(mfcc[k].mean())
        vals[f"mfcc{k}_std"] = float(mfcc[k].std())

    out = np.array([vals[name] for name in FEATURE_NAMES], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)



# ---------------------------------------------------------------------------
# Softmax linear model (shared by the trained backend and the rule fallback)
# ---------------------------------------------------------------------------

class _SoftmaxEmotionModel:
    """z-scored features -> 6-class softmax. Inference is one matmul."""

    def __init__(self, W: np.ndarray, b: np.ndarray, feat_mean: np.ndarray,
                 feat_std: np.ndarray, temperature: float = 1.0) -> None:
        self.W = np.asarray(W, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        self.feat_mean = np.asarray(feat_mean, dtype=np.float32)
        self.feat_std = np.asarray(feat_std, dtype=np.float32)
        self.temperature = float(temperature)

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        z = (np.asarray(feats, dtype=np.float32) - self.feat_mean) / self.feat_std
        z = np.nan_to_num(z, nan=0.0, posinf=Z_CLIP, neginf=-Z_CLIP)
        z = np.clip(z, -Z_CLIP, Z_CLIP)
        logits = (self.W @ z + self.b) / max(self.temperature, 1e-3)
        logits = logits - logits.max()
        e = np.exp(logits)
        return (e / e.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Rule-based fallback: population priors + hand-crafted z-space weights.
# Rationale per emotion (prosody literature on acted/emotional speech):
#   happy    - high/wide F0, fast rate, rising pitch
#   sad      - low/flat F0, quiet, slow, more pauses
#   angry    - loud, noisy (high ZCR/flux), tense (jitter), bright spectrum
#   surprise - high F0 with a strong rising contour, short duration
#   fear     - high jitter/shimmer (tremor), breathy (flatness), tense
#   neutral  - the flat middle ground (negative weights on variability)
# ---------------------------------------------------------------------------

# Priors for the level-dependent features are calibrated to the
# loudness-normalised (v2) feature space: after normalisation every utterance
# has RMS 0.1, pinning rms_db_mean near -23 dB and flux near 0.5 (measured on
# the trainer's synthetic profiles). The old un-normalised priors
# (rms_db -30 dB, flux 0.03) would give every input the same large angry-biased
# z-scores.
_BASE_PRIORS = {
    "rms_db": (-23.0, 6.0), "f0": (150.0, 60.0), "zcr": (0.12, 0.05),
    "centroid": (1600.0, 500.0), "rolloff": (3200.0, 1000.0),
    "flatness": (0.06, 0.03), "flux": (0.5, 0.25),
}
_EXTRA_PRIORS = {
    "voiced_ratio": (0.75, 0.10), "syll_rate": (4.0, 1.5), "pause_ratio": (0.15, 0.10),
    "jitter": (0.02, 0.012), "shimmer": (0.08, 0.04), "duration": (3.0, 1.5),
}

_RULE_WEIGHTS = {
    0: {"f0_std": -0.5, "rms_db_std": -0.3, "jitter": -0.4, "zcr_std": -0.3, "flux_mean": -0.3},
    1: {"f0_mean": 0.9, "f0_std": 0.8, "f0_range": 0.5, "f0_slope": 0.4,
        "rms_db_mean": 0.4, "syll_rate": 0.5, "flux_mean": 0.3},
    2: {"f0_mean": -1.0, "rms_db_mean": -0.8, "syll_rate": -0.8, "f0_std": -0.6,
        "f0_slope": -0.5, "flux_mean": -0.3, "pause_ratio": 0.4, "duration": 0.2},
    3: {"rms_db_mean": 1.0, "f0_std": 0.7, "zcr_mean": 0.7, "flux_mean": 0.6,
        "jitter": 0.4, "f0_mean": 0.4, "centroid_mean": 0.4},
    4: {"f0_mean": 0.9, "f0_slope": 0.8, "f0_range": 0.6, "f0_max": 0.4,
        "duration": -0.4, "rms_db_max": 0.3},
    5: {"f0_mean": 0.5, "jitter": 1.0, "shimmer": 0.9, "flatness_mean": 0.4,
        "zcr_mean": 0.3, "syll_rate": 0.3},
}
_RULE_BIAS = {0: 0.5}  # mild prior toward neutral (the conversational default)


def _prior_for(feature_name: str) -> tuple:
    for contour, (mu, sigma) in _BASE_PRIORS.items():
        if feature_name.startswith(contour + "_"):
            stat = feature_name[len(contour) + 1:]
            if stat == "mean":
                return mu, max(sigma * 0.6, 1e-6)
            if stat == "std":
                return sigma, max(sigma * 0.6, 1e-6)
            if stat == "min":
                return mu - sigma, max(sigma, 1e-6)
            if stat == "max":
                return mu + sigma, max(sigma, 1e-6)
            if stat == "range":
                return 2.0 * sigma, max(sigma, 1e-6)
            return 0.0, max(sigma * 0.5, 1e-6)  # slope
    if feature_name in _EXTRA_PRIORS:
        mu, sigma = _EXTRA_PRIORS[feature_name]
        return mu, max(sigma, 1e-6)
    if feature_name.startswith("mfcc"):
        # cosmetic only - no rule weights touch the mfcc features
        return (40.0, 12.0) if feature_name.endswith("_mean") else (10.0, 4.0)
    return 0.0, 1.0


def _build_rule_model() -> _SoftmaxEmotionModel:
    mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    std = np.ones(FEATURE_DIM, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        mean[i], std[i] = _prior_for(name)
    W = np.zeros((NUM_EMOTIONS, FEATURE_DIM), dtype=np.float32)
    b = np.zeros(NUM_EMOTIONS, dtype=np.float32)
    for eid, weights in _RULE_WEIGHTS.items():
        for name, w in weights.items():
            if name in _FEATURE_INDEX:
                W[eid, _FEATURE_INDEX[name]] = w
    for eid, bias in _RULE_BIAS.items():
        b[eid] = bias
    return _SoftmaxEmotionModel(W, b, mean, std, temperature=1.5)


_RULE_MODEL = _build_rule_model()



# ---------------------------------------------------------------------------
# Weight file I/O (shared with scripts/train_prosody_emotion.py)
# ---------------------------------------------------------------------------

def save_weights(path, W: np.ndarray, b: np.ndarray, feat_mean: np.ndarray,
                 feat_std: np.ndarray, temperature: float = 1.0,
                 extra_files=None) -> None:
    """Persist a trained model (~7 KB npz).

    extra_files: optional list of user file names that were part of training
    (trainer --extra_dir), stored as metadata so the batch-test cell can flag
    predictions on those files as training-set results.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W=np.asarray(W, dtype=np.float32),
        b=np.asarray(b, dtype=np.float32),
        feat_mean=np.asarray(feat_mean, dtype=np.float32),
        feat_std=np.asarray(feat_std, dtype=np.float32),
        feature_names=np.array(FEATURE_NAMES),
        temperature=np.float32(temperature),
        format_version=np.int32(WEIGHTS_FORMAT_VERSION),
        extra_files=np.array(list(extra_files or []), dtype=str),
    )


def load_weights(path) -> _SoftmaxEmotionModel:
    data = np.load(path, allow_pickle=False)
    version = int(data["format_version"]) if "format_version" in data.files else 1
    if version != WEIGHTS_FORMAT_VERSION:
        raise ValueError(
            f"weight file {path} is format v{version} but this prosody_emotion.py "
            f"expects v{WEIGHTS_FORMAT_VERSION} (loudness-normalised features). "
            f"Retrain with scripts/train_prosody_emotion.py (or re-run the "
            f"training notebook) to regenerate prosody_emotion.npz."
        )
    names = [str(x) for x in data["feature_names"]]
    if tuple(names) != FEATURE_NAMES:
        raise ValueError(
            f"weight file {path} feature layout does not match this "
            f"prosody_emotion.py (retrain with scripts/train_prosody_emotion.py)"
        )
    return _SoftmaxEmotionModel(
        data["W"], data["b"], data["feat_mean"], data["feat_std"],
        float(data["temperature"]),
    )


# ---------------------------------------------------------------------------
# Public classifier (contract-compatible with SpeechEmotionClassifier.predict)
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent


class ProsodyEmotionClassifier:
    """predict(pcm_24k) -> (emotion_id, label, confidence). CPU-only."""

    def __init__(self, weights_path=None, verbose: bool = True) -> None:
        candidates = []
        if weights_path:
            candidates.append(Path(weights_path))
        env_path = os.environ.get("PROSODY_EMOTION_WEIGHTS", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(_MODULE_DIR / "checkpoints" / "prosody_emotion.npz")
        candidates.append(_MODULE_DIR / "prosody_emotion.npz")

        self.model: _SoftmaxEmotionModel | None = None
        self.backend = "rules"
        self.weights_path: Path | None = None
        for cand in candidates:
            try:
                if cand.is_file():
                    self.model = load_weights(cand)
                    self.backend = "softmax"
                    self.weights_path = cand
                    break
            except Exception as exc:
                print(f"[PROSODY-SER] failed to load {cand}: {exc!r}; trying next", flush=True)
        if self.model is None:
            self.model = _RULE_MODEL
            if verbose:
                print(
                    "[PROSODY-SER] no trained prosody_emotion.npz found - using the "
                    "rule-based fallback. Train the tiny softmax on RAVDESS with "
                    "scripts/train_prosody_emotion.py for better accuracy.",
                    flush=True,
                )
        elif verbose:
            print(f"[PROSODY-SER] loaded trained weights: {self.weights_path}", flush=True)

    def predict(self, pcm_24k: np.ndarray):
        """24 kHz float32 mono utterance -> (emotion_id, label, confidence)."""
        feats = extract_features(pcm_24k)
        if not feats.any():
            return 0, "neutral", 0.0
        probs = self.model.predict_proba(feats)
        eid = int(np.argmax(probs))
        return eid, EMOTION_NAMES[eid], float(probs[eid])

