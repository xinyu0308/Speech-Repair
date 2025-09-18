import os
import torch
import torchaudio
from TTS.utils.manage import ModelManager
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig


# ==== Load (wav, text, speaker_id) list ====
def load_text_list(text_file):
    data = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            filename, hyp_text = parts
            # Extract speaker id (3rd~4th character in filename)
            if len(filename) < 4:
                continue
            speaker_id = filename[2:4]
            data.append((filename, hyp_text, speaker_id))
    return data


# ==== Get single reference wav ====
# Reference wav should have the same name as the text filename
def get_speaker_reference_wav(reference_wav_root, filename):
    wav_path = os.path.join(reference_wav_root, filename + ".wav")
    if not os.path.isfile(wav_path):
        raise Exception(f"No reference wav found for filename {filename}")
    return wav_path


# ========== Path Config ==========
# Replace the following paths with your own
text_file = "PATH/TO/text/file"  # input text file containing (filename, transcription)
reference_wav_folder = "PATH/TO/reference/wavs"  # folder containing reference wavs
output_folder = "PATH/TO/output/folder"  # folder to save generated wavs
os.makedirs(output_folder, exist_ok=True)

cdsd_data = load_text_list(text_file)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# ========== Load XTTS Model ==========
model_name = "tts_models/zh/xtts_v2"
model_path, _, _ = ModelManager().download_model(model_name)

config_path = os.path.join(model_path, "config.json")
config = XttsConfig()
config.load_json(config_path)

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)

# Load your fine-tuned checkpoint
# Replace with your own checkpoint path
speaker_encoder_ckpt = torch.load("PATH/TO/finetuned/xtts_speaker_encoder.pt", map_location="cpu")

# Load into XTTS speaker encoder
model.hifigan_decoder.speaker_encoder.load_state_dict(speaker_encoder_ckpt, strict=False)

model.to(device)


# ========== Inference ==========
for filename, transcription, speaker_id in cdsd_data:
    try:
        # Get the corresponding reference wav
        reference_wav = get_speaker_reference_wav(reference_wav_folder, filename)

        # Check if reference wav is at least 1 second long
        info = torchaudio.info(reference_wav)
        duration = info.num_frames / info.sample_rate
        if duration < 1.0:
            print(f"[Warning] Reference wav too short for file {filename}")
            continue

        # Extract conditioning latents
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[reference_wav])

        # Generate speech
        print(f"Generating speech for speaker {speaker_id}: {transcription} ...")
        out = model.inference(
            transcription,
            "zh",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7
        )

        # Save output wav
        speaker_output_folder = os.path.join(output_folder, f"S{speaker_id}")
        os.makedirs(speaker_output_folder, exist_ok=True)
        out_filename = filename + ".wav"
        output_path = os.path.join(speaker_output_folder, out_filename)

        torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
        print(f"Output saved to {output_path}")

    except Exception as e:
        print(f"[Error] Speaker {speaker_id} ({filename}): {e}")
