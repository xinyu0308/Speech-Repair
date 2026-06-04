#!/usr/bin/env python3
# coding: utf-8

import os
import math
import random
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# XTTS speaker encoder
from TTS.tts.layers.xtts.hifigan_decoder import ResNetSpeakerEncoder


# =============== Utils ===============

def read_kaldi_map(path: str) -> Dict[str, str]:
    """Read Kaldi-style wav.scp or utt2spk mapping."""
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            mp[parts[0]] = parts[1]
    return mp


def extract_spk_from_filename(filename: str) -> str:
    """Extract speaker ID from filename (3rd and 4th characters)."""
    fname = os.path.basename(filename).split(".")[0]
    return fname[2:4] if len(fname) >= 4 else "00"


def load_wav(path: str, target_sr: int) -> torch.Tensor:
    """Load wav, resample if needed, convert to mono."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(dim=0, keepdim=False)  # mono [T]
    return wav


def random_chunk_1d(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Randomly crop/pad 1D waveform to fixed length."""
    T = x.shape[0]
    if T == 0:
        return torch.zeros(num_samples)
    if T == num_samples:
        return x
    if T > num_samples:
        start = random.randint(0, T - num_samples)
        return x[start:start + num_samples]
    pad = torch.zeros(num_samples - T, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-9):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_scores(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix."""
    a = l2_normalize(mat_a, dim=-1)
    b = l2_normalize(mat_b, dim=-1)
    return a @ b.t()


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC for binary classification."""
    order = np.argsort(-scores)  # descending
    scores = scores[order]
    labels = labels[order]
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return 0.5
    tp = fp = 0.0
    prev_score = None
    points = [(0.0, 0.0)]
    for s, y in zip(scores, labels):
        if prev_score is None or s != prev_score:
            points.append((fp / N, tp / P))
            prev_score = s
        if y == 1:
            tp += 1
        else:
            fp += 1
    points.append((fp / N, tp / P))
    area = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        area += (x1 - x0) * (y0 + y1) * 0.5
    return area


# =============== Dataset ===============

class KaldiSpeakerDataset(Dataset):
    """Dataset for speaker finetuning from Kaldi-style wav.scp and utt2spk."""

    def __init__(
        self,
        wav_scp: str,
        utt2spk: str,
        sample_rate: int = 16000,
        chunk_secs: float = 3.0,
        external_mel: bool = False,
        n_mels: int = 64,
    ):
        super().__init__()
        self.sr = sample_rate
        self.chunk_samples = int(chunk_secs * sample_rate)
        self.external_mel = external_mel

        self.wav_map = read_kaldi_map(wav_scp)
        self.utt2spk = {}
        for utt, path in self.wav_map.items():
            spk = extract_spk_from_filename(os.path.basename(path))
            self.utt2spk[utt] = spk

        speakers = sorted(set(self.utt2spk.values()))
        self.spk2id = {s: i for i, s in enumerate(speakers)}
        self.utterances = [u for u in self.wav_map.keys() if u in self.utt2spk]

        if external_mel:
            self.mel_extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=512,
                win_length=400,
                hop_length=160,
                n_mels=n_mels,
                power=2.0,
                center=True,
                mel_scale="htk",
                f_min=0.0,
                f_max=None,
            )
            self.n_mels = n_mels

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx: int):
        utt = self.utterances[idx]
        spk = self.utt2spk[utt]
        spk_id = self.spk2id[spk]

        wav_path = self.wav_map[utt]
        wav = load_wav(wav_path, self.sr)

        if wav.abs().max().item() > 0:
            wav = wav / wav.abs().max().clamp(min=1e-6)

        wav = random_chunk_1d(wav, self.chunk_samples)

        if self.external_mel:
            mel = self.mel_extractor(wav)
            mel = mel.clamp(min=1e-5).transpose(0, 1).contiguous()
            return mel, spk_id
        else:
            return wav, spk_id


def collate_waveform(batch):
    wavs, spk_ids = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    wavs = pad_sequence(wavs, batch_first=True)
    spk_ids = torch.tensor(spk_ids, dtype=torch.long)
    return wavs, lengths, spk_ids


def collate_mel(batch):
    mels, spk_ids = zip(*batch)
    lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    mels = pad_sequence(mels, batch_first=True)
    spk_ids = torch.tensor(spk_ids, dtype=torch.long)
    return mels, lengths, spk_ids


# =============== Model ===============

class SpeakerIdHead(nn.Module):
    """Classification head for speaker ID prediction."""
    def __init__(self, emb_dim: int, num_speakers: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_speakers)

    def forward(self, emb: torch.Tensor):
        return self.fc(emb)


class XTTSFinetuneModel(nn.Module):
    """XTTS speaker encoder with classification head."""
    def __init__(self, use_torch_spec: bool, audio_cfg: dict, input_dim: int, proj_dim: int, num_speakers: int):
        super().__init__()
        self.encoder = ResNetSpeakerEncoder(
            input_dim=input_dim,
            proj_dim=proj_dim,
            log_input=True,
            use_torch_spec=use_torch_spec,
            audio_config=audio_cfg,
        )
        self.head = SpeakerIdHead(proj_dim, num_speakers)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb)
        return emb, logits


# =============== Train / Eval ===============

def train_one_epoch(loader, model, opt, criterion, device, max_grad_norm=5.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, lengths, y = batch
        x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)
        _, logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_verification_auc(loader, model, device, max_utts_per_spk=10, imposter_pairs=1000):
    """Evaluate speaker verification AUC using centroid-based scoring."""
    model.eval()
    all_embs: Dict[int, List[torch.Tensor]] = {}

    for batch in loader:
        x, lengths, y = batch
        x = x.to(device)
        emb, _ = model(x)
        for e, sid in zip(emb.cpu(), y):
            sid = int(sid)
            all_embs.setdefault(sid, [])
            if len(all_embs[sid]) < max_utts_per_spk:
                all_embs[sid].append(e)

    centroids = {sid: torch.stack(embs).mean(dim=0) for sid, embs in all_embs.items() if len(embs) > 0}

    pos_scores, neg_scores = [], []
    rng = np.random.default_rng(0)

    # Positive scores (embedding vs centroid of same speaker)
    for sid, embs in all_embs.items():
        if sid not in centroids:
            continue
        centroid = centroids[sid]
        for e in embs:
            s = torch.cosine_similarity(e, centroid, dim=0).item()
            pos_scores.append(s)

    # Negative scores (embedding vs centroid of different speaker)
    sids = list(all_embs.keys())
    for _ in range(imposter_pairs):
        if len(sids) < 2:
            break
        a, b = rng.choice(sids, size=2, replace=False)
        ea = rng.choice(all_embs[a])
        cb = centroids[b]
        s = torch.cosine_similarity(ea, cb, dim=0).item()
        neg_scores.append(s)

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    scores = np.array(pos_scores + neg_scores, dtype=np.float64)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=np.int32)
    return auc(scores, labels)


# =============== Main ===============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-wav-scp", required=True)
    parser.add_argument("--train-utt2spk", required=True)
    parser.add_argument("--dev-wav-scp", required=True)
    parser.add_argument("--dev-utt2spk", required=True)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-secs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=64)

    parser.add_argument("--external-mel", action="store_true",
                        help="If set, dataset precomputes mel-spectrograms (use_torch_spec=False).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="xtts_speaker_finetuned.pt")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = KaldiSpeakerDataset(
        wav_scp=args.train_wav_scp,
        utt2spk=args.train_utt2spk,
        sample_rate=args.sample_rate,
        chunk_secs=args.chunk_secs,
        external_mel=args.external_mel,
        n_mels=args.n_mels,
    )
    dev_set = KaldiSpeakerDataset(
        wav_scp=args.dev_wav_scp,
        utt2spk=args.dev_utt2spk,
        sample_rate=args.sample_rate,
        chunk_secs=args.chunk_secs,
        external_mel=args.external_mel,
        n_mels=args.n_mels,
    )

    if args.external_mel:
        collate_fn = collate_mel
        input_dim = args.n_mels
        use_torch_spec = False
    else:
        collate_fn = collate_waveform
        input_dim = args.n_mels
        use_torch_spec = True

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_fn, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                            collate_fn=collate_fn, pin_memory=True)

    audio_cfg = {
        "fft_size": 512,
        "win_length": 400,
        "hop_length": 160,
        "sample_rate": args.sample_rate,
        "preemphasis": 0.97,
        "num_mels": args.n_mels,
    }
    num_speakers = len(train_set.spk2id)

    model = XTTSFinetuneModel(
        use_torch_spec=use_torch_spec,
        audio_cfg=audio_cfg,
        input_dim=input_dim,
        proj_dim=args.proj_dim,
        num_speakers=num_speakers,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, device)
        dev_auc = eval_verification_auc(dev_loader, model, device)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  dev_verif_auc={dev_auc:.4f}")

        if dev_auc >= best_auc:
            best_auc = dev_auc
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(args),
                "train_speakers": train_set.spk2id,
            }
            torch.save(state, args.save)
            print(f"  -> Saved best checkpoint to {args.save} (AUC={best_auc:.4f})")

    print(f"Best dev AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
