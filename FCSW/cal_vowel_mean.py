#!/usr/bin/env python3
# coding: utf-8
"""
Compute vowel formant statistics from WAV files using TextGrid alignments.
Generates mean, standard deviation, and tolerance ranges for male/female speakers.
"""

import os
import json
import tgt
import librosa
import parselmouth
import numpy as np

# ================== Config ==================
wav_base_dir = "/PATH/TO/WAVS"       # Base folder containing input WAV files
textgrid_base_dir = "/PATH/TO/TEXTGRIDS"  # Base folder containing TextGrid files
vowels = {"a", "e", "i", "o", "u"}

# Tolerance multiplier for formant ranges
k = 2.5

# Output JSON path
output_path = "./vowel_formant_tolerance.json"

# ================== Containers ==================
vowel_formants_male = {v: [] for v in vowels}
vowel_formants_female = {v: [] for v in vowels}

# ================== Helper Functions ==================

def remove_tone(phone: str) -> str:
    """Remove tone marks from phoneme."""
    tone_marks = {"˥","˩","˧","˨","˦","˥˩","˧˥","˥˩˧","˨˩˦","˥˩˨","˧˥˩"}
    for t in tone_marks:
        phone = phone.replace(t, "")
    return phone

def get_formants(y: np.ndarray, sr: int):
    """Compute F1 and F2 using Parsemouth."""
    snd = parselmouth.Sound(y, sr)
    formant = snd.to_formant_burg()
    f1 = formant.get_value_at_time(1, snd.xmax / 2)
    f2 = formant.get_value_at_time(2, snd.xmax / 2)
    return f1, f2

def get_gender_from_filename(filename: str):
    """Extract speaker ID and determine gender."""
    try:
        speaker_id = filename[2:4]
        if speaker_id in ["04", "07"]:
            return "female"
        elif speaker_id in ["26", "39"]:
            return "male"
        else:
            return None
    except IndexError:
        return None

# ================== Processing ==================
for fn in os.listdir(wav_base_dir):
    if not fn.endswith(".wav"):
        continue

    gender = get_gender_from_filename(fn)
    if gender is None:
        continue  # Skip speakers not in the list

    wav_path = os.path.join(wav_base_dir, fn)
    tg_path = os.path.join(textgrid_base_dir, fn.replace(".wav", ".TextGrid"))
    if not os.path.exists(tg_path):
        continue

    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=16000)

        # Load TextGrid and select phone tier
        tg = tgt.read_textgrid(tg_path)
        tier = tg.get_tier_by_name("phones")

        for interval in tier.intervals:
            ph = remove_tone(interval.text.lower().strip())
            if ph not in vowels:
                continue

            start, end = int(interval.start_time * sr), int(interval.end_time * sr)
            seg = y[start:end]
            if len(seg) < 160:  # Skip too short segments
                continue

            try:
                f1, f2 = get_formants(seg, sr)
                if f1 and f2:
                    if gender == "male":
                        vowel_formants_male[ph].append((f1, f2))
                    else:
                        vowel_formants_female[ph].append((f1, f2))
            except Exception:
                continue

    except Exception as e:
        print(f"Error processing {fn}: {e}")
        continue

# ================== Compute Statistics ==================
tolerance_ranges = {"male": {}, "female": {}}

for gender, data in [("male", vowel_formants_male), ("female", vowel_formants_female)]:
    for v in sorted(vowels):
        arr = np.array(data[v])
        if arr.size == 0:
            # No data available
            tolerance_ranges[gender][v] = {
                "mean": {"F1": None, "F2": None},
                "tolerance": {"F1": [None, None], "F2": [None, None]}
            }
            continue

        f1s, f2s = arr[:, 0], arr[:, 1]
        mu_f1, mu_f2 = float(f1s.mean()), float(f2s.mean())
        sd_f1, sd_f2 = float(f1s.std(ddof=0)), float(f2s.std(ddof=0))
        tol_f1 = [mu_f1 - k * sd_f1, mu_f1 + k * sd_f1]
        tol_f2 = [mu_f2 - k * sd_f2, mu_f2 + k * sd_f2]

        tolerance_ranges[gender][v] = {
            "mean": {"F1": mu_f1, "F2": mu_f2},
            "tolerance": {"F1": tol_f1, "F2": tol_f2}
        }

# ================== Print Statistics ==================
for gender in ["male", "female"]:
    print(f"{gender.capitalize()} vowel formant stats and tolerance:")
    for v in sorted(vowels):
        stats = tolerance_ranges[gender][v]
        mu_f1 = stats["mean"]["F1"]
        mu_f2 = stats["mean"]["F2"]
        t1_low, t1_high = stats["tolerance"]["F1"]
        t2_low, t2_high = stats["tolerance"]["F2"]

        if mu_f1 is None:
            print(f" {v}: no data available")
        else:
            print(
                f" {v}: mean F1/F2 = ({mu_f1:.1f},{mu_f2:.1f}), "
                f"tol F1 = [{t1_low:.1f},{t1_high:.1f}], "
                f"tol F2 = [{t2_low:.1f},{t2_high:.1f}]"
            )

# ================== Save to JSON ==================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tolerance_ranges, f, ensure_ascii=False, indent=2)

print(f"Saved tolerance ranges to {output_path}")
