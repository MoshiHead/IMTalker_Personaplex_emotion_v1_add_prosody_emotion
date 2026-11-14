#!/usr/bin/env python3
"""Train the tiny prosody softmax classifier (PROSODY_SER_V1) from scratch.

No pretrained models anywhere: features come from prosody_emotion.extract_features
(the exact same function used at inference, so train/serve parity is guaranteed)
and the classifier is a multinomial logistic regression trained with plain numpy
gradient descent. The result is a ~7 KB .npz that prosody_emotion.py loads
transparently (replacing its rule-based fallback).

Datasets (download manually, they are public but not pip-installable):
  * RAVDESS speech audio (Zenodo, Audio_Speech_Actors_01-24.zip): 24 actors,
    1,440 clips, 8 emotion categories. Mapping to the avatar ids:
    01 neutral -> 0, 02 calm -> 0, 03 happy -> 1, 04 sad -> 2, 05 angry -> 3,
    06 fearful -> 5, 07 disgust -> skipped (the emotion LoRA has no disgust),
    08 surprised -> 4.
  * CREMA-D (optional, its GitHub repo provides the wav archive): 91 actors,
    7,442 clips. ANG -> 3, FEA -> 5, HAP -> 1, NEU -> 0, SAD -> 2 (DIS skipped;
    CREMA-D has no surprise class).

Validation is speaker-grouped k-fold CV: no speaker appears in both train and
test, so the reported accuracy is an honest estimate for a *new* voice.

Channel robustness: --noise_aug (default ON) adds a noisy copy of every
training utterance (white/pink noise, random 6-30 dB SNR). Without it the
studio-trained model learns "bright/noisy spectrum = angry" and collapses
every real-world (mic + room noise) recording to angry - reproduced
experimentally: at 20 dB SNR all clean-trained predictions flip to angry.

--extra_dir accepts a user folder with files named <emotion>-*.wav
(angry/fear/happy/sad/neutral/surprise), added to training x10 so a handful of
files matters. They form their own CV speaker group, so the CV report shows an
honest held-out accuracy on the user's audio.

Usage:
  python scripts/train_prosody_emotion.py --ravdess_dir /data/ravdess \
      [--cremad_dir /data/cremad] [--augment] [--extra_dir /workspace/audio] \
      [--no_noise_aug] [--out IMTalker/checkpoints/prosody_emotion.npz]
  python scripts/train_prosody_emotion.py --selftest --out /tmp/prosody.npz

After training, re-run scripts/switch_ser_to_prosody.py to embed the new
weights into runpod_emotion_streaming.ipynb (Step 12 writes them onto the pod).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "IMTalker"))

import prosody_emotion as pe  # noqa: E402

SR = pe.INPUT_SR

# RAVDESS emotion-code -> avatar id (None = skip the sample)
RAVDESS_MAP = {"01": 0, "02": 0, "03": 1, "04": 2, "05": 3, "06": 5, "07": None, "08": 4}
# CREMA-D emotion-code -> avatar id (None = skip)
CREMAD_MAP = {"ANG": 3, "DIS": None, "FEA": 5, "HAP": 1, "NEU": 0, "SAD": 2}


# ---------------------------------------------------------------------------
# Audio loading (soundfile -> scipy -> stdlib wave fallback chain)
# ---------------------------------------------------------------------------

def _resample_to(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(src_sr), int(dst_sr))
        return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)
    except Exception:
        t_old = np.linspace(0, len(x) / src_sr, num=len(x), endpoint=False)
        n_new = int(round(len(x) * dst_sr / src_sr))
        t_new = np.linspace(0, len(x) / src_sr, num=n_new, endpoint=False)
        return np.interp(t_new, t_old, x).astype(np.float32)


def load_wav(path: Path, target_sr: int = SR) -> np.ndarray:
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return _resample_to(data.mean(axis=1), sr, target_sr)
    except ImportError:
        pass
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return _resample_to(data, sr, target_sr)
    except ImportError:
        pass
    import wave
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    data = raw.astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return _resample_to(data, sr, target_sr)


# ---------------------------------------------------------------------------
# Channel-noise augmentation (the real-world robustness fix).
# RAVDESS/CREMA-D are studio recordings; real audio carries mic/room noise.
# Without a noisy copy of every training utterance, the noise direction
# (high zcr/rolloff/mfcc-std/flux) stays discriminative and points at angry,
# collapsing non-studio input to one class. Adding noisy copies makes those
# features non-discriminative and forces the model onto noise-robust prosodic
# cues (f0, syllable rate, energy dynamics).
# ---------------------------------------------------------------------------

NOISE_SNR_RANGE = (6.0, 30.0)   # dB, uniform per augmented copy


def add_channel_noise(x: np.ndarray, rng: np.random.Generator,
                      snr_db: float | None = None) -> np.ndarray:
    """One noisy copy of x: white or pink noise at the given/random SNR."""
    p_sig = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12
    snr = float(rng.uniform(*NOISE_SNR_RANGE)) if snr_db is None else float(snr_db)
    noise = rng.standard_normal(x.shape[0])
    if rng.random() < 0.5:          # pink-ish: integrated white, closer to room noise
        noise = np.cumsum(noise)
        noise -= noise.mean()
    noise = noise / (float(np.sqrt(np.mean(np.square(noise)))) + 1e-12)
    return (x + noise * p_sig / 10 ** (snr / 20)).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset ingestion
# ---------------------------------------------------------------------------

# user extra-dir emotion prefix -> avatar id (None = skip the file)
EXTRA_MAP = {"neutral": 0, "calm": 0, "happy": 1, "sad": 2, "angry": 3,
             "surprise": 4, "surprised": 4, "fear": 5, "fearful": 5}
EXTRA_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
EXTRA_REPEAT = 10   # a few user files must matter among thousands of dataset clips


def scan_extra(root: Path):
    """Yield (wav_path, emotion_id, speaker_id) for user files named
    <emotion>-<anything>.<ext>, e.g. angry-1.wav. Unknown emotions are skipped."""
    out = []
    for wav in sorted(Path(root).rglob("*")):
        if wav.suffix.lower() not in EXTRA_EXTS:
            continue
        emo = EXTRA_MAP.get(wav.stem.split("-")[0].lower())
        if emo is None:
            continue
        out.append((wav, emo, "user_extra"))
    return out


def scan_ravdess(root: Path):
    """Yield (wav_path, emotion_id, speaker_id) for every usable RAVDESS clip."""
    out = []
    for wav in sorted(Path(root).rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        emo = RAVDESS_MAP.get(parts[2])
        if emo is None:
            continue
        out.append((wav, emo, f"rav_{parts[-1]}"))
    return out


def scan_cremad(root: Path):
    """Yield (wav_path, emotion_id, speaker_id) for every usable CREMA-D clip."""
    out = []
    for wav in sorted(Path(root).rglob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        emo = CREMAD_MAP.get(parts[2].upper())
        if emo is None:
            continue
        out.append((wav, emo, f"cre_{parts[0]}"))
    return out


def build_dataset(samples, augment: bool = False, noise_aug: bool = True,
                  rng: np.random.Generator | None = None):
    """samples: list of (wav_path, emotion_id, speaker_id).

    Returns X (n, FEATURE_DIM), y (n,), speakers (n,). Feature extraction uses
    the exact inference-time function, guaranteeing train/serve parity.
    Augmentation (optional, x5): gain {0.7, 1.3} and tape-style speed factors
    {0.9, 1.1} applied via resampling (pitch+tempo shift). With noise_aug
    (default on) every utterance also contributes one noisy copy (white/pink
    noise at a random 6-30 dB SNR) - the channel-robustness fix.
    """
    variants = [(1.0, 1.0)]
    if augment:
        variants += [(1.3, 1.0), (0.7, 1.0), (1.0, 1.1), (1.0, 0.9)]
    if rng is None:
        rng = np.random.default_rng(0)
    X, y, spk = [], [], []
    t0 = time.time()
    for i, (wav, emo, speaker) in enumerate(samples):
        try:
            base = load_wav(wav)
        except Exception as exc:
            print(f"  [warn] skipping {wav.name}: {exc!r}")
            continue
        for gain, speed in variants:
            x = base
            if speed != 1.0:
                t_old = np.linspace(0, len(x) / SR, num=len(x), endpoint=False)
                t_new = np.linspace(0, len(x) / SR / speed, num=int(len(x) / speed), endpoint=False)
                x = np.interp(t_new, t_old, x).astype(np.float32)
            if gain != 1.0:
                x = np.clip(x * gain, -1.0, 1.0)
            feats = pe.extract_features(x)
            if not feats.any():
                continue
            X.append(feats)
            y.append(emo)
            spk.append(speaker)
        if noise_aug:
            feats = pe.extract_features(add_channel_noise(base, rng))
            if feats.any():
                X.append(feats)
                y.append(emo)
                spk.append(speaker)
        if (i + 1) % 100 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  features: {i + 1}/{len(samples)} files ({rate:.0f}/s)")
    return np.stack(X), np.array(y), np.array(spk)


# ---------------------------------------------------------------------------
# Softmax regression, from scratch (numpy full-batch gradient descent)
# ---------------------------------------------------------------------------

def train_softmax(Z: np.ndarray, y: np.ndarray, n_classes: int = pe.NUM_EMOTIONS,
                  iters: int = 1000, lr: float = 0.5, l2: float = 1e-4,
                  seed: int = 0):
    """Class-weighted multinomial logistic regression. Z is pre-standardised."""
    n, d = Z.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(n_classes, d))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    cw = n / (n_classes * np.maximum(counts, 1))       # balanced classes
    sw = cw[y][:, None]
    for it in range(iters):
        logits = Z @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        E = (P - Y) * sw
        W -= lr * (E.T @ Z / n + l2 * W)
        b -= lr * E.mean(axis=0)
        if (it + 1) % 250 == 0:
            lr *= 0.5
    return W, b


def evaluate(W, b, Z, y):
    logits = Z @ W.T + b
    pred = logits.argmax(axis=1)
    return float((pred == y).mean()), pred


def speaker_cv(X, y, speakers, folds: int = 5, seed: int = 0):
    """Speaker-grouped k-fold CV. Returns (mean_acc, out_of_fold_preds, fold_accs)."""
    uniq = np.unique(speakers)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    groups = [uniq[i::folds] for i in range(folds)]
    oof = np.full(len(y), -1)
    fold_accs = []
    for k in range(folds):
        test_mask = np.isin(speakers, groups[k])
        train_mask = ~test_mask
        mu = X[train_mask].mean(0)
        # 1e-3 floor (not 1e-6): after loudness normalisation some features are
        # near-constant, and a near-zero std would make z-scores unstable.
        sd = np.maximum(X[train_mask].std(0), 1e-3)
        W, b = train_softmax((X[train_mask] - mu) / sd, y[train_mask], seed=seed + k)
        acc, pred = evaluate(W, b, (X[test_mask] - mu) / sd, y[test_mask])
        oof[test_mask] = pred
        fold_accs.append(acc)
        print(f"  fold {k + 1}/{folds}: {acc:.1%} "
              f"({int(test_mask.sum())} utts, {len(groups[k])} speakers held out)")
    return float(np.mean(fold_accs)), oof, fold_accs


def print_report(y, pred, title):
    print(f"\n{title}")
    print(f"  overall accuracy: {(pred == y).mean():.1%}  (n={len(y)})")
    print("  per-class:")
    print(f"  {'emotion':<10} {'acc':>6} {'support':>8}")
    for c in range(pe.NUM_EMOTIONS):
        mask = y == c
        if not mask.any():
            continue
        acc = float((pred[mask] == c).mean())
        print(f"  {pe.EMOTION_NAMES[c]:<10} {acc:>6.1%} {int(mask.sum()):>8}")
    print("  confusion (rows=true, cols=pred):")
    header = "          " + "".join(f"{pe.EMOTION_NAMES[c][:8]:>9}" for c in range(pe.NUM_EMOTIONS))
    print(header)
    for c in range(pe.NUM_EMOTIONS):
        row = [int(((y == c) & (pred == p)).sum()) for p in range(pe.NUM_EMOTIONS)]
        print(f"  {pe.EMOTION_NAMES[c][:8]:<8}" + "".join(f"{v:>9}" for v in row))


def print_feature_importance(W, feat_std, top: int = 15):
    imp = (np.abs(W) * feat_std[None, :]).mean(axis=0)
    order = np.argsort(-imp)[:top]
    print(f"\n  top-{top} features by |W|*std (class-averaged):")
    for i in order:
        print(f"    {pe.FEATURE_NAMES[i]:<18} {imp[i]:.3f}")



# ---------------------------------------------------------------------------
# --selftest: synthetic speakers with distinct prosody profiles.
# Validates the whole pipeline (features -> CV -> training -> npz) without any
# dataset download. NOT a substitute for RAVDESS/CREMA-D training.
# ---------------------------------------------------------------------------

_SELFTEST_PROFILES = {
    # emotion_id: (f0_mean, f0_vibrato_depth, amp, syllable_rate, noise)
    0: (140.0, 10.0, 0.12, 3.5, 0.002),   # neutral
    1: (235.0, 55.0, 0.22, 5.0, 0.002),   # happy
    2: (100.0, 6.0, 0.06, 2.0, 0.001),    # sad
    3: (190.0, 35.0, 0.32, 4.5, 0.010),   # angry
    4: (265.0, 80.0, 0.18, 3.0, 0.002),   # surprise
    5: (215.0, 18.0, 0.14, 4.2, 0.004),   # fear
}


def synth_selftest_sample(emo: int, rng: np.random.Generator) -> np.ndarray:
    f0m, depth, amp, syll, noise = _SELFTEST_PROFILES[emo]
    seconds = float(rng.uniform(1.5, 3.0))
    speaker_f0 = float(rng.uniform(-20, 20))          # simulated speaker offset
    t = np.arange(int(seconds * SR)) / SR
    f0 = f0m + speaker_f0 + depth * np.sin(2 * np.pi * rng.uniform(2, 4) * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros_like(t)
    for h, a in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        sig += a * np.sin(h * phase)
    env = amp * np.clip(0.55 + 0.45 * np.sin(2 * np.pi * syll * t), 0.05, 1.0)
    return (env * sig + noise * rng.standard_normal(t.shape[0])).astype(np.float32)


def run_selftest(out_path: Path, folds: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    X, y, spk = [], [], []
    n_per = 40
    speakers = [f"synthetic_{s}" for s in range(12)]
    for emo in range(pe.NUM_EMOTIONS):
        for s in range(n_per):
            speaker = speakers[s % len(speakers)]
            rng_spk = np.random.default_rng(rng.integers(0, 2**31))
            x = synth_selftest_sample(emo, rng_spk)
            for variant in (x, add_channel_noise(x, rng)):  # clean + noisy copy
                feats = pe.extract_features(variant)
                X.append(feats)
                y.append(emo)
                spk.append(speaker)
    X, y, spk = np.stack(X), np.array(y), np.array(spk)
    print(f"selftest dataset: {X.shape[0]} synthetic utterances (clean + noise-augmented), "
          f"{len(np.unique(spk))} simulated speakers")
    mean_acc, oof, _ = speaker_cv(X, y, spk, folds=folds, seed=seed)
    print_report(y, oof, "selftest speaker-grouped CV (synthetic data)")
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-3)
    W, b = train_softmax((X - mu) / sd, y, seed=seed)
    pe.save_weights(out_path, W, b, mu, sd)
    print(f"\n[ok] selftest weights written: {out_path} "
          f"({out_path.stat().st_size} bytes), CV acc {mean_acc:.1%}")
    if mean_acc < 0.55:
        sys.exit("selftest CV accuracy suspiciously low - the pipeline is broken")

    # Channel-robustness regression guard: synthetic profiles under 12 dB noise
    # must not collapse to one class (the all-angry bug this augmentation fixes).
    rng_guard = np.random.default_rng(seed + 99)
    clf = pe.ProsodyEmotionClassifier(weights_path=out_path, verbose=False)
    preds = [clf.predict(add_channel_noise(synth_selftest_sample(emo, rng_guard),
                                           rng_guard, snr_db=12.0))[0]
             for emo in range(pe.NUM_EMOTIONS)]
    n_ok = sum(int(p == e) for e, p in enumerate(preds))
    print(f"[ok] noise-robustness guard: {n_ok}/{pe.NUM_EMOTIONS} profiles correct at "
          f"12 dB SNR: {[pe.EMOTION_NAMES[p] for p in preds]}")
    if n_ok < pe.NUM_EMOTIONS - 1:
        sys.exit("noise-robustness guard failed: predictions collapse under channel noise")

    # Edge-silence guard: profiles padded with 2.4 s of edge silence must still
    # classify correctly (studio data is trimmed; real recordings are not).
    preds_pad = []
    for emo in range(pe.NUM_EMOTIONS):
        x = synth_selftest_sample(emo, rng_guard)
        pad = np.concatenate([np.zeros(int(0.6 * SR), np.float32), x,
                              np.zeros(int(1.8 * SR), np.float32)]).astype(np.float32)
        preds_pad.append(clf.predict(pad)[0])
    n_ok = sum(int(p == e) for e, p in enumerate(preds_pad))
    print(f"[ok] edge-silence guard: {n_ok}/{pe.NUM_EMOTIONS} profiles correct with "
          f"2.4 s of edge silence: {[pe.EMOTION_NAMES[p] for p in preds_pad]}")
    if n_ok < pe.NUM_EMOTIONS - 1:
        sys.exit("edge-silence guard failed: predictions change under edge padding")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ravdess_dir", default="", help="RAVDESS speech wav root")
    ap.add_argument("--cremad_dir", default="", help="CREMA-D wav root (optional)")
    ap.add_argument("--out", default=str(ROOT / "IMTalker" / "checkpoints" / "prosody_emotion.npz"))
    ap.add_argument("--augment", action="store_true",
                    help="5x data: gain {0.7,1.3} + tape speed {0.9,1.1}")
    ap.add_argument("--noise_aug", action=argparse.BooleanOptionalAction, default=True,
                    help="add a noisy copy of every utterance (default: on; the "
                         "channel-robustness fix for non-studio audio)")
    ap.add_argument("--extra_dir", default="",
                    help="optional user folder with <emotion>-*.wav files "
                         "(angry/fear/happy/sad/neutral/surprise); added x%d" % EXTRA_REPEAT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="train on synthetic speakers (no dataset needed; pipeline check)")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.selftest:
        run_selftest(out_path, args.folds, args.seed)
        return

    samples = []
    if args.ravdess_dir:
        found = scan_ravdess(Path(args.ravdess_dir))
        print(f"RAVDESS: {len(found)} usable clips under {args.ravdess_dir}")
        samples += found
    if args.cremad_dir:
        found = scan_cremad(Path(args.cremad_dir))
        print(f"CREMA-D: {len(found)} usable clips under {args.cremad_dir}")
        samples += found
    extra_names = []
    if args.extra_dir:
        found = scan_extra(Path(args.extra_dir))
        if found:
            emos = sorted({pe.EMOTION_NAMES[e] for _, e, _ in found})
            print(f"extra audio: {len(found)} usable clips under {args.extra_dir} "
                  f"({', '.join(emos)}), replicated x{EXTRA_REPEAT}")
            print("             note: these files are IN the training set - testing on the")
            print("             same folder afterwards measures training data, not generalisation")
            samples += found * EXTRA_REPEAT
            extra_names = sorted({wav.name for wav, _, _ in found})
        else:
            print(f"[warn] --extra_dir given but no <emotion>-*.wav files found under {args.extra_dir}")
    if not samples:
        sys.exit("no clips found - pass --ravdess_dir/--cremad_dir or use --selftest")

    aug_desc = []
    if args.augment:
        aug_desc.append("5x gain/speed")
    if args.noise_aug:
        aug_desc.append("+1 noisy copy each (6-30 dB SNR)")
    print(f"extracting features ({', '.join(aug_desc) or 'no augmentation'})...")
    X, y, speakers = build_dataset(samples, augment=args.augment,
                                   noise_aug=args.noise_aug,
                                   rng=np.random.default_rng(args.seed))
    print(f"dataset: {X.shape[0]} utterances, {len(np.unique(speakers))} speakers, "
          f"class counts {np.bincount(y, minlength=pe.NUM_EMOTIONS).tolist()}")

    print(f"\nspeaker-grouped {args.folds}-fold CV:")
    mean_acc, oof, fold_accs = speaker_cv(X, y, speakers, folds=args.folds, seed=args.seed)
    print_report(y, oof, "speaker-independent CV (held-out speakers = unseen voices)")
    if extra_names:
        user_mask = speakers == "user_extra"
        # user files are their own speaker group: one fold held them ALL out,
        # so their out-of-fold predictions are an honest generalisation estimate
        print(f"\n  user extra files (held out as their own speaker group): "
              f"{(oof[user_mask] == y[user_mask]).mean():.1%} honest accuracy on your audio "
              f"(n={int(user_mask.sum())})")

    # final model: all data
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-3)
    W, b = train_softmax((X - mu) / sd, y, seed=args.seed)
    train_acc, _ = evaluate(W, b, (X - mu) / sd, y)
    print(f"\nfinal model train accuracy (all data): {train_acc:.1%}")
    print_feature_importance(W, sd)

    pe.save_weights(out_path, W, b, mu, sd, extra_files=extra_names)
    print(f"\n[ok] weights written: {out_path} ({out_path.stat().st_size} bytes)")

    # quick latency check with the freshly trained model
    clf = pe.ProsodyEmotionClassifier(weights_path=out_path, verbose=False)
    probe = np.clip(0.15 * np.sin(2 * np.pi * 220 * np.arange(3 * SR) / SR), -1, 1).astype(np.float32)
    for _ in range(3):
        clf.predict(probe)
    t0 = time.perf_counter()
    for _ in range(20):
        clf.predict(probe)
    print(f"[ok] inference latency: {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms/utterance (CPU)")
    print("\nnext step: re-run scripts/switch_ser_to_prosody.py to embed these "
          "weights into runpod_emotion_streaming.ipynb before deploying.")


if __name__ == "__main__":
    main()

