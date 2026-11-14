#!/usr/bin/env python3
"""Switch runpod_emotion_streaming.ipynb from the emotion2vec SER backend to
the built-in lightweight prosody backend (PROSODY_SER_V1).

What it does (idempotent - safe to re-run, e.g. after retraining the weights):
  * Parameters cell: SER_MODEL -> "prosody-v1", SER_DEVICE -> "cpu", comments
    rewritten; VAD_SILENCE_SEC 0.7 -> 0.45 and VAD_MIN_SPEECH_SEC 0.5 -> 0.35
    (classification is ~ms now, so the emotion switch can land sooner)
  * Step 6: removes the funasr install (the prosody backend is numpy-only)
  * Step 10b (md + code): no-op instead of the ~1.1 GB HF pre-download
  * Step 12: re-embeds IMTalker/auto_emotion.py, embeds the new
    IMTalker/prosody_emotion.py next to it, embeds the trained
    IMTalker/checkpoints/prosody_emotion.npz (base64) when present, and
    updates the --ser_model help text
  * Header point 6 + Step 17 ERROR_HINTS wording
"""

import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
AE_PATH = ROOT / "IMTalker" / "auto_emotion.py"
PE_PATH = ROOT / "IMTalker" / "prosody_emotion.py"
WEIGHTS_PATH = ROOT / "IMTalker" / "checkpoints" / "prosody_emotion.npz"

ae_content = AE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
pe_content = PE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
for label, content in (("auto_emotion.py", ae_content), ("prosody_emotion.py", pe_content)):
    if "'''" in content:
        sys.exit(f"{label} must not contain triple single quotes")
if "prosody-v1" not in ae_content:
    sys.exit("IMTalker/auto_emotion.py has not been switched to the prosody backend yet")

weights_b64 = ""
if WEIGHTS_PATH.is_file():
    weights_b64 = base64.b64encode(WEIGHTS_PATH.read_bytes()).decode("ascii")
    print(f"[ok] embedding trained weights: {WEIGHTS_PATH} "
          f"({WEIGHTS_PATH.stat().st_size} bytes -> {len(weights_b64)} b64 chars)")
else:
    print(f"[info] {WEIGHTS_PATH} not found - embedding the rule-based fallback "
          f"(train weights with scripts/train_prosody_emotion.py, then re-run this script)")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, s):
    cells[i]["source"] = s.splitlines(keepends=True)


def find_cell(cell_id):
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            return i
    raise KeyError(cell_id)


def replace_soft(text, old, new, label):
    """Idempotent replace: applies once, skips when already applied."""
    if new in text:
        print(f"[skip] {label} already applied")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"anchor for {label} found {n}x (expected exactly 1, new text absent)")
    print(f"[ok] {label}")
    return text.replace(old, new)


# --- 1. Parameters cell --------------------------------------------------------
i3 = find_cell("emotion-stream-03")
c3 = src(i3)

old_ser_comment = (
    "# Hugging Face SER model (emotion2vec+ via funasr, 16 kHz input).\n"
    "# Default: emotion2vec/emotion2vec_plus_base - ~90M params (~1.1GB checkpoint),\n"
    "# trained on ~4.8k hours of pseudo-labelled speech: 9 classes\n"
    "# angry/disgusted/fearful/happy/neutral/other/sad/surprised/unknown ->\n"
    "# covers all 6 avatar emotions (disgust maps to --ser_disgust_map;\n"
    "# other/unknown fall back to neutral). ~20-50ms per utterance on the\n"
    "# 5090 - zero streaming impact. Requires the funasr package (installed in\n"
    "# Step 6). Wav2Vec2 classification-head repos remain supported as\n"
    "# alternatives (auto_emotion.py detects them from the model name).\n"
    'SER_MODEL = "emotion2vec/emotion2vec_plus_base"\n'
)
new_ser_comment = (
    "# Built-in lightweight SER backend (PROSODY_SER_V1). Default: \"prosody-v1\" -\n"
    "# the pure-numpy prosodic classifier in prosody_emotion.py (written to\n"
    "# IMTalker/ by Step 12): no pretrained model, no GPU, no download, ~2-5ms\n"
    "# per utterance on CPU. It uses the trained prosody_emotion.npz weights\n"
    "# when present (trained from scratch on RAVDESS speaker-independent data by\n"
    "# scripts/train_prosody_emotion.py, embedded in Step 12) and a rule-based\n"
    "# fallback otherwise. Pretrained HF SER models remain supported as opt-in\n"
    "# alternatives (e.g. emotion2vec/emotion2vec_plus_base via funasr).\n"
    'SER_MODEL = "prosody-v1"\n'
)
c3 = replace_soft(c3, old_ser_comment, new_ser_comment, "Parameters SER_MODEL block")

old_device_comment = (
    '# Device for SER inference. "cuda" (default) = run on the avatar GPU:\n'
    "# ~20-50ms per utterance on a daemon thread, so the emotion switch lands ~1s\n"
    "# after the utterance ends (the previously-working configuration). \"cpu\" is\n"
    "# NOT recommended on a shared-CPU pod: SER inference takes seconds there\n"
    "# and competes with the JPEG-encode pool, so detections land late/stale and\n"
    "# the avatar expression stops tracking your voice.\n"
    'SER_DEVICE = "cuda"\n'
)
new_device_comment = (
    "# Device for SER inference. Ignored by the prosody backend (always CPU,\n"
    "# ~ms per utterance on a daemon thread - zero GPU contention, so the\n"
    "# emotion switch lands ~0.5s after the utterance ends). Only used by the\n"
    "# opt-in HF SER backends.\n"
    'SER_DEVICE = "cpu"\n'
)
c3 = replace_soft(c3, old_device_comment, new_device_comment, "Parameters SER_DEVICE block")

c3 = replace_soft(
    c3,
    "VAD_SILENCE_SEC = 0.7         # trailing silence that closes an utterance\n",
    "VAD_SILENCE_SEC = 0.45        # trailing silence that closes an utterance (lowered:\n"
    "                                # prosody classification is ~ms, so the switch lands sooner)\n",
    "Parameters VAD_SILENCE_SEC",
)
c3 = replace_soft(
    c3,
    'VAD_MIN_SPEECH_SEC = 0.5      # shorter utterances are ignored (coughs, "yes")\n',
    'VAD_MIN_SPEECH_SEC = 0.35     # shorter utterances are ignored (coughs, "yes") (lowered\n'
    "                                # to catch short reactive utterances)\n",
    "Parameters VAD_MIN_SPEECH_SEC",
)
set_src(i3, c3)

# --- 2. Step 6: remove the funasr install (prosody backend is numpy-only) ------
i21 = find_cell("emotion-stream-21")
c21 = src(i21)
funasr_block = (
    "# funasr: runtime for the emotion2vec SER model (AUTO_EMOTION; Step 10b / 12).\n"
    "# funasr does not depend on torch, so the cu128 pin is safe; its other deps\n"
    "# (modelscope, umap_learn, ...) are unpinned, so the project pins are\n"
    "# re-asserted afterwards in case pip resolved anything newer.\n"
    'run(f"{VENV_ACTIVATE} && pip install funasr", retries=2, retry_delay=15)\n'
    "run(\n"
    '    f\'{VENV_ACTIVATE} && pip install "numpy<2.0.0" "transformers==4.30.2" \'\n'
    '    f\'"pandas<2.2.0" "matplotlib<3.9.0"\',\n'
    "    retries=2, retry_delay=15,\n"
    ")\n"
    "assert_torch_pin()\n"
)
# (cannot use replace_soft here: the replacement text "assert_torch_pin()" is a
# common substring of the cell, so key idempotency off the funasr block itself)
if funasr_block in c21:
    c21 = c21.replace(funasr_block, "assert_torch_pin()\n")
    print("[ok] Step 6 funasr removal")
elif "funasr" not in c21:
    print("[skip] Step 6 funasr removal already applied")
else:
    sys.exit("Step 6 funasr block anchor not found (and funasr still present)")
set_src(i21, c21)

# --- 3. Step 10b md: no download -----------------------------------------------
i10bm = find_cell("emotion-stream-29b-md")
c10bm = src(i10bm)
old_10bm = (
    "## Step 10b — Pre-download the auto-emotion SER model\n"
    "\n"
    "Downloads the speech-emotion-recognition model into the Hugging Face cache\n"
    "so the server's first startup doesn't stall on a ~1.1 GB download. Skipped\n"
    "when `AUTO_EMOTION = False`.\n"
)
new_10bm = (
    "## Step 10b — Auto-emotion SER: nothing to download\n"
    "\n"
    "The built-in prosody SER backend (`prosody-v1`) is pure numpy and ships with\n"
    "the repo (written by Step 12), so there is no ~1.1 GB HF checkpoint to\n"
    "pre-download. The trained weights (`prosody_emotion.npz`, ~7 KB) are embedded\n"
    "in the Step 12 cell when present; without them the rule-based fallback is\n"
    "used. This cell is kept as a no-op so the step numbering stays stable.\n"
)
c10bm = replace_soft(c10bm, old_10bm, new_10bm, "Step 10b md")
set_src(i10bm, c10bm)

# --- 4. Step 10b code: no-op -----------------------------------------------------
i10b = find_cell("emotion-stream-29b")
c10b = src(i10b)
old_10b = (
    "if AUTO_EMOTION:\n"
    "    run(f'{VENV_ACTIVATE} && export HF_HUB_ENABLE_HF_TRANSFER=1 && hf download {SER_MODEL}',\n"
    "        retries=3, retry_delay=15)\n"
    '    print(f"[ok] SER model cached: {SER_MODEL}")\n'
    "else:\n"
    '    print("[skip] AUTO_EMOTION is False - no SER model to download")\n'
)
new_10b = (
    "if AUTO_EMOTION:\n"
    '    print(f"[ok] SER backend {SER_MODEL!r} is built in (prosody_emotion.py) - no download needed")\n'
    "else:\n"
    '    print("[skip] AUTO_EMOTION is False")\n'
)
c10b = replace_soft(c10b, old_10b, new_10b, "Step 10b code")
set_src(i10b, c10b)



# --- 5. Step 12: re-embed auto_emotion.py + embed prosody_emotion.py/weights ---
i33 = find_cell("emotion-stream-33")
c33 = src(i33)

# 5a. re-embed the auto_emotion module (always, so the cell tracks the repo file)
m = re.search(r"AUTO_EMOTION_MODULE = r'''(.*?)'''", c33, re.DOTALL)
if not m:
    sys.exit("AUTO_EMOTION_MODULE not found in Step 12 cell")
c33 = c33.replace(m.group(1), ae_content)
print("[ok] Step 12: AUTO_EMOTION_MODULE re-embedded")

# 5b. prosody module + weights block (insert once, refresh on re-runs)
PROSODY_BLOCK = f"""
# ---------------------------------------------------------------------------
# prosody_emotion.py - built-in lightweight SER backend (PROSODY_SER_V1).
# Pure numpy, CPU-only, no pretrained model. Written next to auto_emotion.py.
# The trained weights (prosody_emotion.npz, embedded as base64) go to
# IMTalker/checkpoints/; without them the rule-based fallback is used.
# ---------------------------------------------------------------------------
PROSODY_MODULE = r'''{pe_content}'''
PROSODY_WEIGHTS_B64 = "{weights_b64}"

pe_path = Path(IMTALKER_DIR) / "prosody_emotion.py"
if pe_path.exists() and pe_path.read_text(encoding="utf-8") == PROSODY_MODULE:
    print("[skip] prosody_emotion.py already up to date")
else:
    pe_path.write_text(PROSODY_MODULE, encoding="utf-8")
    print(f"[written] {{pe_path}} ({{len(PROSODY_MODULE)}} chars)")

pe_weights_path = Path(IMTALKER_DIR) / "checkpoints" / "prosody_emotion.npz"
if PROSODY_WEIGHTS_B64:
    import base64 as _b64
    _wb = _b64.b64decode(PROSODY_WEIGHTS_B64)
    if pe_weights_path.exists() and pe_weights_path.read_bytes() == _wb:
        print("[skip] prosody_emotion.npz already up to date")
    else:
        pe_weights_path.parent.mkdir(parents=True, exist_ok=True)
        pe_weights_path.write_bytes(_wb)
        print(f"[written] {{pe_weights_path}} ({{len(_wb)}} bytes)")
else:
    print("[info] no trained prosody weights embedded - the rule-based fallback will be used")
"""

if "PROSODY_MODULE = r'''" in c33:
    # re-run: refresh the module text and the embedded weights in place
    m = re.search(r"PROSODY_MODULE = r'''(.*?)'''", c33, re.DOTALL)
    if not m:
        sys.exit("PROSODY_MODULE marker present but unparseable in Step 12 cell")
    c33 = c33.replace(m.group(1), pe_content)
    c33 = re.sub(r'PROSODY_WEIGHTS_B64 = "[^"]*"',
                 f'PROSODY_WEIGHTS_B64 = "{weights_b64}"', c33, count=1)
    print("[ok] Step 12: PROSODY_MODULE / weights refreshed")
else:
    anchor = (
        'ae_path = Path(IMTALKER_DIR) / "auto_emotion.py"\n'
        'if ae_path.exists() and ae_path.read_text(encoding="utf-8") == AUTO_EMOTION_MODULE:\n'
        '    print("[skip] auto_emotion.py already up to date")\n'
        "else:\n"
        '    ae_path.write_text(AUTO_EMOTION_MODULE, encoding="utf-8")\n'
        '    print(f"[written] {ae_path} ({len(AUTO_EMOTION_MODULE)} chars)")\n'
    )
    if c33.count(anchor) != 1:
        sys.exit(f"ae_path write block anchor found {c33.count(anchor)}x (expected 1)")
    c33 = c33.replace(anchor, anchor + PROSODY_BLOCK)
    print("[ok] Step 12: PROSODY_MODULE / weights block inserted")

# 5c. server argparse default + help (inside SERVER_PATCHES json; quotes are
# backslash-escaped in the raw cell text)
c33 = replace_soft(
    c33,
    r'parser.add_argument(\"--ser_model\", default=\"emotion2vec/emotion2vec_plus_base\",',
    r'parser.add_argument(\"--ser_model\", default=\"prosody-v1\",',
    "Step 12 --ser_model default",
)
c33 = replace_soft(
    c33,
    r'help=\"Hugging Face speech-emotion-recognition model (emotion2vec+ via funasr, or a wav2vec2 classification head; 16 kHz input)\"',
    r'help=\"SER backend: prosody-v1 (built-in lightweight prosody classifier, default), or a Hugging Face speech-emotion-recognition model (emotion2vec+ via funasr, or a wav2vec2 classification head)\"',
    "Step 12 --ser_model help",
)
# 5d. start_winner_live.sh patch: shell-side defaults (env_overrides always
# passes explicit values; these are the fallbacks)
c33 = replace_soft(
    c33,
    r'--ser_model \"${SER_MODEL:-emotion2vec/emotion2vec_plus_base}\"',
    r'--ser_model \"${SER_MODEL:-prosody-v1}\"',
    "Step 12 start_winner --ser_model default",
)
c33 = replace_soft(
    c33,
    r'--vad_silence_sec \"${VAD_SILENCE_SEC:-0.7}\"',
    r'--vad_silence_sec \"${VAD_SILENCE_SEC:-0.45}\"',
    "Step 12 start_winner vad_silence_sec default",
)
c33 = replace_soft(
    c33,
    r'--vad_min_speech_sec \"${VAD_MIN_SPEECH_SEC:-0.5}\"',
    r'--vad_min_speech_sec \"${VAD_MIN_SPEECH_SEC:-0.35}\"',
    "Step 12 start_winner vad_min_speech_sec default",
)
set_src(i33, c33)

# --- 6. header markdown point 6 --------------------------------------------------
i0 = find_cell("emotion-stream-00")
c0 = src(i0)
old_p6 = (
    "   the user's mic into utterances; a daemon thread classifies each one with the\n"
    "   emotion2vec+ SER model (`emotion2vec/emotion2vec_plus_base`, ~90M params,\n"
    "   9 classes covering all 6 avatar emotions) and publishes the detected emotion\n"
)
new_p6 = (
    "   the user's mic into utterances; a daemon thread classifies each one with the\n"
    "   built-in prosody SER backend (`prosody-v1`: pure-numpy prosodic features +\n"
    "   a tiny softmax trained on RAVDESS - no GPU, no download, ~ms per utterance)\n"
    "   and publishes the detected emotion\n"
)
c0 = replace_soft(c0, old_p6, new_p6, "header point 6")
set_src(i0, c0)

# --- 7. Step 17: ERROR_HINTS wording ---------------------------------------------
i44 = find_cell("emotion-stream-44")
c44 = src(i44)
c44 = replace_soft(
    c44,
    "check the traceback above (HF download, VRAM, funasr/transformers import)",
    "check the traceback above (prosody_emotion.py weights/import, or the opt-in HF SER backend config)",
    "Step 17 ERROR_HINTS",
)
set_src(i44, c44)

# --- 8. post-checks ----------------------------------------------------------------
compile(ae_content, "auto_emotion.py", "exec")
compile(pe_content, "prosody_emotion.py", "exec")
full = "\n".join("".join(c["source"]) for c in cells)
for needle, where in (
    ('SER_MODEL = "prosody-v1"', "Parameters"),
    ('SER_DEVICE = "cpu"', "Parameters"),
    ("prosody_emotion.py", "Step 12 prosody block"),
    ("PROSODY_WEIGHTS_B64", "Step 12 weights"),
    (r'default=\"prosody-v1\"', "Step 12 server argparse"),
    (r'${SER_MODEL:-prosody-v1}', "Step 12 start_winner_live.sh"),
    ("VAD_SILENCE_SEC = 0.45", "Parameters VAD"),
    ("VAD_MIN_SPEECH_SEC = 0.35", "Parameters VAD"),
):
    if needle not in full:
        sys.exit(f"post-check failed: {needle!r} missing ({where})")
for bad, where in (
    ("pip install funasr", "Step 6"),
    ('SER_MODEL = "emotion2vec/emotion2vec_plus_base"', "Parameters"),
    (r'default=\"emotion2vec/emotion2vec_plus_base\"', "Step 12 server argparse"),
    (r'${SER_MODEL:-emotion2vec/emotion2vec_plus_base}', "Step 12 start_winner_live.sh"),
):
    if bad in full:
        sys.exit(f"post-check failed: {bad!r} still present ({where})")
print("[ok] post-checks passed")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[ok] notebook written: {NB_PATH}")

