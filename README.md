# Speech Repair Framework

This repository integrates three core modules for **Mandarin dysarthric speech repair**:

1. **TTS with Speaker Adaptation** — Speech generation using XTTS with a finetuned speaker encoder.
2. **Pretrained Model-based ASR** — Speech recognition using pretrained wav2vec2 models with ESPnet/Fairseq/S3PRL.
3. **Formant-Corrective Spectral Warping (FCSW)** — Vowel correction via forced alignment and spectral warping.

Each module has its own detailed README with installation and usage instructions. This document provides an overview.

---

## 📂 Modules

### 📝 Pretrained Model-based ASR

* **Goal:** Transcribe Mandarin dysarthric speech using pretrained self-supervised models.
* **Key Features:**

  * Integrates **ESPnet**, **Fairseq**, and **S3PRL**.
  * Uses pretrained **chinese-wav2vec2-large** for robust ASR.
* **Details:** See [Pretrained Model-based ASR README](./asr-pretrained/README.md).

---

### 🎶 Formant-Corrective Spectral Warping (FCSW)

* **Goal:** Repair vowel distortions by correcting formant frequencies.
* **Key Features:**

  * Uses **Montreal Forced Aligner (MFA)** for phone-level alignment.
  * Extracts vowel statistics and applies spectral warping for correction.
* **Details:** See [FCSW README](./fcws/README.md).

---
### 🔊 TTS with Speaker Adaptation

* **Goal:** Generate natural Mandarin speech from text, conditioned on dysarthric speakers.
* **Key Features:**

  * Finetune XTTS speaker encoder with your own dataset.
  * Generate speech using reference audio to preserve speaker identity.
* **Details:** See [TTS with Speaker Adaptation README](./tts-speaker-adaptation/README.md).

---

## 🔗 Workflow Integration

These modules can be combined into a **speech repair pipeline**:

1. **ASR Module** transcribes dysarthric speech into text.
2. **FCSW Module** normalizes distorted vowels for cleaner acoustic cues.
3. **TTS with Speaker Adaptation** regenerates repaired speech while preserving speaker identity.

---

## 📚 References

* [TencentGameMate](https://github.com/TencentGameMate)
* [ESPnet](https://github.com/espnet/espnet)
* [Fairseq](https://github.com/facebookresearch/fairseq)
* [S3PRL](https://github.com/s3prl/s3prl)
* [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html)
* [Coqui TTS](https://github.com/coqui-ai/TTS)

---

## 🎯 Summary

* Use **ASR** for transcription.
* Apply **FCSW** for vowel correction.
* Use **TTS with Speaker Adaptation** for natural speech reconstruction.
=