# Formant-Corrective Spectral Warping (FCSW)

This repository provides an implementation of the **Formant-Corrective Spectral Warping (FCSW)** module for vowel correction in dysarthric Mandarin speech.
The workflow includes forced alignment, vowel formant statistics extraction, and speech repair with formant repair.

---

## 📦 Requirements

* [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html)
* Python 3.8+ with common dependencies (NumPy, SciPy, librosa, etc.)

---

## 🚀 Workflow

### Step 1 — Forced Alignment with MFA

Run the Montreal Forced Aligner to obtain word/phone alignments:

```bash
mfa align ~/mfa_data/my_corpus mandarin_china_mfa mandarin_mfa ~/mfa_data/my_corpus_aligned
```

* `~/mfa_data/my_corpus`: Input speech corpus
* `mandarin_china_mfa`: Acoustic model for Mandarin (provided by MFA)
* `mandarin_mfa`: Pronunciation dictionary for Mandarin (provided by MFA)
* `~/mfa_data/my_corpus_aligned`: Output directory containing aligned TextGrid files

---

### Step 2 — Compute Vowel Statistics

Run the script to extract average vowel formant values:

```bash
python cal_vowel_mean.py
```

This will generate a JSON file containing reference formant statistics.

---

### Step 3 — Run Inference with FCSW

Run the inference script to generate vowel-corrected speech:

```bash
python inference.py
```

Before running, update the following paths inside `inference.py`:

```python
reference_formant = "/Path/To/Reference_formant.json"
dys_wav_dir      = "/PATH/TO/dys_wav_dir"
dys_tg_dir       = "/PATH/TO/dys_MFA_dir"
out_dir          = "/PATH/TO/out_dir"
```

* `reference_formant`: JSON file generated in Step 2
* `dys_wav_dir`: Directory containing dysarthric speech wav files
* `dys_tg_dir`: Directory containing MFA-aligned TextGrid files
* `out_dir`: Output directory where vowel-corrected wav files will be saved

---

## 📚 References

* [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html)
