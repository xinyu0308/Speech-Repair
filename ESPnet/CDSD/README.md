# Speech Recognition

This repository provides instructions for building a **speech recognition system** based on pretrained models using **ESPnet**, **Fairseq**, and **S3PRL**.
The workflow integrates a Chinese pretrained model (`chinese-wav2vec2-large`) with the ESPnet training pipeline.

---

## 📦 Requirements

Before starting, install the following toolkits:

* [ESPnet](https://github.com/espnet/espnet)
* [Fairseq](https://github.com/facebookresearch/fairseq)
* [S3PRL](https://github.com/s3prl/s3prl)

Download the pretrained model checkpoint from Hugging Face:
👉 [TencentGameMate/chinese-wav2vec2-large](https://huggingface.co/TencentGameMate/chinese-wav2vec2-large)

File to download:

```
chinese-wav2vec2-large-fairseq-ckpt.pt
```

---

## 🚀 Setup and Training

1. **Copy configuration files**

   * Copy the provided config files into:

     ```
     espnet/egs2/aishell/asr1/conf
     ```
   * Replace the existing `path.sh` in:

     ```
     espnet/egs2/aishell/asr1/path.sh
     ```
   * Copy the provided `run_ssl.sh` script into:

     ```
     espnet/egs2/aishell/asr1
     ```

2. **Train the model**
   Navigate to the AISHELL recipe directory:

   ```bash
   cd espnet/egs2/aishell/asr1
   ./run_ssl.sh
   ```

   This will launch training with the pretrained wav2vec2 model integrated into the ESPnet pipeline.

---

## 📚 References

* [TencentGameMate](https://github.com/TencentGameMate)
* [ESPnet](https://github.com/espnet/espnet)
* [Fairseq](https://github.com/facebookresearch/fairseq)
* [S3PRL](https://github.com/s3prl/s3prl)

