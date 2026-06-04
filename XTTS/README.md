# TTS with Speaker Adaptation

This repository provides a demo for **Formant-Guided Speech Repair** for Mandarin dysarthric speech using **XTTS** with a finetuned speaker encoder.
The workflow involves **speaker encoder finetuning** and **speech generation with reference audio**.

---

## 📦 Requirements

Please refer to the official [Coqui TTS](https://github.com/coqui-ai/TTS) repository for installation and setup.

---

## 🚀 Usage

### Step 1 — Finetune XTTS Speaker Encoder

Before generating speech, finetune the XTTS speaker encoder on your dataset to adapt to your speakers:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_xtts_speaker.py \
  --train-wav-scp /path/to/train/wav.scp \
  --train-utt2spk /path/to/train/utt2spk \
  --dev-wav-scp   /path/to/dev/wav.scp \
  --dev-utt2spk   /path/to/dev/utt2spk \
  --epochs 50 \
  --batch-size 64 \
  --lr 3e-4 \
  --chunk-secs 3.0 \
  --save xtts_speaker_finetuned.pt
```

**Notes:**

* `wav.scp` should contain:

  ```
  utt_id  /path/to/audio.wav
  ```
* `utt2spk` should contain:

  ```
  utt_id  speaker_id
  ```
* After training, the finetuned checkpoint will be saved as **`xtts_speaker_finetuned.pt`**.

---

### Step 2 — Prepare Inference

Edit **`inference.py`** with your paths:

```python
text_file = "PATH/TO/text/file"                 # Input text file: (filename, transcription)
reference_wav_folder = "PATH/TO/reference/wavs" # Folder containing reference wavs per speaker
output_folder = "PATH/TO/output/folder"         # Folder to save generated wavs
speaker_encoder_ckpt = "PATH/TO/finetuned/xtts_speaker_encoder.pt"  # Finetuned checkpoint
```

**Text file format:**

```
filename1  transcription1
filename2  transcription2
...
```

Each `filename` should correspond to a `.wav` file in the `reference_wav_folder`.
Generated outputs will be saved under `output_folder`, grouped by **speaker ID**.

---

### Step 3 — Run Inference

Run the demo to generate repaired speech:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py
```

The script will:

1. Load the XTTS model and your finetuned speaker encoder.
2. Load reference wavs and input text.
3. Generate repaired speech and save `.wav` files under the designated output folder.

---

## ✨ Reference

[Coqui TTS](https://github.com/coqui-ai/TTS).