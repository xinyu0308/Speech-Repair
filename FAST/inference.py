#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import logging
from typing import Optional, Tuple

import librosa
import numpy as np
import parselmouth
import torchaudio
import tgt
import torch

# ================== Config ==================
SR = 16000               # Sampling rate
F0_THRESH = 160.0        # Pitch threshold for gender inference
CONTEXT_MS = 30          # Context window around phones (ms)
FADE_MS = 30             # Crossfade duration (ms)
ALPHA_SCALE = 2           # Scaling factor for alpha
MIN_ALPHA = 0.05          # Minimum alpha threshold
LOW_CUT = 80.0
HF_START = 2000.0
HF_GAIN_DB = 2.0

DEFAULT_VOWELS = set("aeiou")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)

# Load precomputed vowel formant statistics
with open("/Path/To/Reference_formant.json", "r", encoding="utf-8") as f:
    FORMANT_STATS = json.load(f)

# ================== Helper Functions ==================

def remove_tone(ph: str) -> str:
    """Remove tonal marks from phoneme."""
    for t in ("˥","˩","˧","˨","˦","˥˩","˧˥","˥˩˧","˨˩˦","˥˩˨","˧˥˩"):
        ph = ph.replace(t, "")
    return ph.strip()

def get_formants_praat(wav: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[float]]:
    """Compute Praat F1 and F2 from a waveform segment."""
    snd = parselmouth.Sound(wav, sr)
    formant = snd.to_formant_burg()
    try:
        t_mid = 0.5 * snd.duration
        f1 = formant.get_value_at_time(1, t_mid) or None
        f2 = formant.get_value_at_time(2, t_mid) or None
    except:
        return None, None
    return f1, f2

def get_target_formant_and_alpha(
    ph: str, gender: str, orig_f1: float, orig_f2: float
) -> Tuple[Optional[float], Optional[float], float]:
    """Compute target formants and dynamic alpha based on statistics."""
    stats_g = FORMANT_STATS.get(gender, {})
    ent = stats_g.get(ph)
    if not ent or ent["mean"]["F1"] is None:
        return None, None, 0.0
    m1, m2 = ent["mean"]["F1"], ent["mean"]["F2"]
    dev = max(abs(orig_f1 - m1) / (m1 + 1e-12),
              abs(orig_f2 - m2) / (m2 + 1e-12))
    alpha = min(1.0, dev * ALPHA_SCALE)
    if alpha < MIN_ALPHA:
        alpha = 0.0
    return m1, m2, alpha

def warp_spectrum_praat(seg: np.ndarray, sr: int, orig_f1: float, orig_f2: float,
                        tgt_f1: float, tgt_f2: float, alpha: float) -> np.ndarray:
    """Warp spectrum based on Praat formants."""
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    S = librosa.stft(seg, n_fft=n_fft, hop_length=hop, window=win)
    mag, ph = np.abs(S), np.angle(S)
    freqs = np.linspace(0, sr / 2, mag.shape[0])
    sigma1, sigma2 = max(orig_f1 / 4, 1.0), max(orig_f2 / 4, 1.0)
    delta1, delta2 = alpha * (tgt_f1 - orig_f1), alpha * (tgt_f2 - orig_f2)

    def warp(f):
        w1 = np.exp(-0.5 * ((f - orig_f1) / sigma1) ** 2)
        w2 = np.exp(-0.5 * ((f - orig_f2) / sigma2) ** 2)
        return f + delta1 * w1 + delta2 * w2

    f_warp = warp(freqs)
    mag_warp = np.vstack([
        np.interp(freqs, f_warp, mag[:, t]) for t in range(mag.shape[1])
    ]).T
    S_shift = mag_warp * np.exp(1j * ph)
    y = librosa.istft(S_shift, hop_length=hop, window=win, length=len(seg))
    return y.astype(np.float32)

def rms_normalize(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """RMS normalize target segment to match reference."""
    r = np.sqrt(np.mean(ref**2) + 1e-12)
    t = np.sqrt(np.mean(tgt**2) + 1e-12)
    return tgt * (r / (t + 1e-12))

def spectral_crossfade(orig: np.ndarray, new: np.ndarray, sr: int, fade_len_sec: float) -> np.ndarray:
    """Perform spectral crossfade between original and new segments."""
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    L = max(len(orig), len(new))
    o = np.pad(orig, (0, L - len(orig)), 'constant')
    n = np.pad(new, (0, L - len(new)), 'constant')
    So = librosa.stft(o, n_fft=n_fft, hop_length=hop, window=win)
    Sn = librosa.stft(n, n_fft=n_fft, hop_length=hop, window=win)
    mag_o, ph = np.abs(So), np.angle(So)
    mag_n = np.abs(Sn)
    T = mag_o.shape[1]
    fade_frames = max(1, int(round(fade_len_sec * sr / hop)))
    mask = np.ones(T)
    fi = np.linspace(0, 1, fade_frames)
    mask[:fade_frames] = 1 - fi
    mask[-fade_frames:] = fi
    mag = mag_o * mask + mag_n * (1 - mask)
    mix = mag * np.exp(1j * ph)
    y = librosa.istft(mix, hop_length=hop, window=win, length=L)
    return y.astype(np.float32)

def spectral_eq_boost_high(y: np.ndarray, sr: int, low_cut: float, hf_start: float, hf_gain_db: float) -> np.ndarray:
    """Apply high-frequency EQ boost and low-cut filter."""
    if len(y) < 2: return y
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / sr)
    gain = np.ones_like(f)
    if low_cut > 0:
        t = 0.2 * low_cut
        l0 = max(0, low_cut - t)
        gain[f < l0] = 0
        idx = (f >= l0) & (f <= low_cut)
        gain[idx] = (f[idx] - l0) / (low_cut - l0 + 1e-12)
    if hf_gain_db != 0:
        idx = f >= hf_start
        mg = 10 ** (hf_gain_db / 20)
        gain[idx] *= 1 + (mg - 1) * ((f[idx] - hf_start) / (f.max() - hf_start + 1e-12))
    Y *= gain
    out = np.fft.irfft(Y, n=len(y))
    return out.astype(np.float32)

# ================== Main Processing Functions ==================

def process_file(wav_path: str, tg_path: str, out_path: str,
                 stats_writer: Optional[csv.DictWriter] = None):
    """Process single WAV file using TextGrid and formant statistics."""
    try:
        y, sr = librosa.load(wav_path, sr=SR, dtype=np.float64)
        spk_id = os.path.basename(wav_path)[2:4]
        if spk_id in ("04", "07"):
            gender = "female"
        elif spk_id in ("26", "39"):
            gender = "male"
        else:
            pitch = parselmouth.Sound(y, sr).to_pitch()
            gender = "female" if np.median(pitch.selected_array['frequency']) > F0_THRESH else "male"
        logging.info(f"{os.path.basename(wav_path)}: gender={gender}")
    except Exception as e:
        logging.warning(f"Load failed {wav_path}: {e}")
        return

    tg = tgt.io.read_textgrid(tg_path)
    tier = tg.get_tier_by_name("phones")
    y_out = y.copy()
    ctx = int(CONTEXT_MS * SR / 1000)
    fade_s = FADE_MS / 1000

    for iv in tier.intervals:
        ph = remove_tone(iv.text.lower().strip())
        if ph == "" or ph not in FORMANT_STATS.get(gender, {}):
            continue

        s = max(0, int(iv.start_time * SR) - ctx)
        e = min(len(y), int(iv.end_time * SR) + ctx)
        seg = y[s:e]
        if len(seg) < int(0.01 * SR): continue

        orig_f1, orig_f2 = get_formants_praat(seg, SR)
        if orig_f1 is None or orig_f2 is None: continue

        tgt_f1, tgt_f2, alpha = get_target_formant_and_alpha(ph, gender, orig_f1, orig_f2)

        if stats_writer:
            stats_writer.writerow({
                'wav_file': os.path.basename(wav_path),
                'phone': ph,
                'start_time': iv.start_time,
                'end_time': iv.end_time,
                'f1_praat': orig_f1,
                'f2_praat': orig_f2,
                'tgt_f1': tgt_f1,
                'tgt_f2': tgt_f2,
                'alpha': alpha
            })

        if alpha <= 0: continue

        y_shift = warp_spectrum_praat(seg, SR, orig_f1, orig_f2, tgt_f1, tgt_f2, alpha)
        y_shift = rms_normalize(seg, y_shift)
        L = min(len(seg), len(y_shift))
        y_out[s:s + L] = spectral_crossfade(seg[:L], y_shift[:L], SR, fade_s)

    y_out = spectral_eq_boost_high(y_out, SR, LOW_CUT, HF_START, HF_GAIN_DB)
    tensor = torch.from_numpy(y_out.astype(np.float32)).unsqueeze(0)
    torchaudio.save(out_path, tensor, SR)
    logging.info(f"Saved: {out_path}")

def batch_process(wav_dir: str, tg_dir: str, out_dir: str):
    """Batch process all WAV files in a directory."""
    os.makedirs(out_dir, exist_ok=True)
    stats_csv = os.path.join(out_dir, "formant_stats_summary.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as cf:
        fields = ['wav_file','phone','start_time','end_time','f1_praat','f2_praat','tgt_f1','tgt_f2','alpha']
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        for fn in os.listdir(wav_dir):
            if not fn.lower().endswith(".wav"): continue
            wav_p = os.path.join(wav_dir, fn)
            tg_p = os.path.join(tg_dir, fn.replace(".wav", ".TextGrid"))
            if not os.path.exists(tg_p):
                logging.warning(f"No TextGrid for {fn}")
                continue
            out_p = os.path.join(out_dir, fn)
            process_file(wav_p, tg_p, out_p, stats_writer=writer)

# ================== Main ==================
if __name__ == "__main__":
    dys_wav_dir = "/PATH/TO/dys_wav_dir"
    dys_tg_dir = "/PATH/TO/dys_MFA_dir"
    out_dir = "/PATH/TO/out_dir"
    batch_process(dys_wav_dir, dys_tg_dir, out_dir)
