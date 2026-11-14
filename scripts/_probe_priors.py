#!/usr/bin/env python3
"""Temporary probe: measure population stats of rule-weighted features under
loudness-normalised extract_features, using the trainer's selftest profiles.
Used to recalibrate the rule-fallback priors after the v2 normalisation."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "IMTalker"))
import prosody_emotion as pe  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "train_prosody_emotion", ROOT / "scripts" / "train_prosody_emotion.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

rng = np.random.default_rng(0)
feats = {e: [] for e in range(6)}
for e in range(6):
    for _ in range(12):
        feats[e].append(pe.extract_features(tr.synth_selftest_sample(e, rng)))

idx = {n: i for i, n in enumerate(pe.FEATURE_NAMES)}
keys = ["rms_db_mean", "rms_db_std", "rms_db_min", "rms_db_max", "rms_db_range",
        "f0_mean", "f0_std", "f0_range", "f0_slope", "f0_max",
        "zcr_mean", "zcr_std", "centroid_mean", "rolloff_mean",
        "flatness_mean", "flux_mean", "flux_std",
        "voiced_ratio", "syll_rate", "pause_ratio", "jitter", "shimmer", "duration",
        "mfcc1_mean", "mfcc1_std", "mfcc2_mean", "mfcc2_std"]

print(f"{'feature':<15} {'pop_mean':>9} {'pop_std':>8}   per-emotion means (neutral/happy/sad/angry/surp/fear)")
for k in keys:
    allv = [f[idx[k]] for e in range(6) for f in feats[e]]
    per_emo = " ".join(f"{np.mean([f[idx[k]] for f in feats[e]]):8.3f}" for e in range(6))
    print(f"{k:<15} {np.mean(allv):>9.3f} {np.std(allv):>8.3f}   {per_emo}")
