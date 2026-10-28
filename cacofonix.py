#!/usr/bin/env python3
"""Recompose a sung vocal with five random melodies and gibberish lyrics.

Gibberish substitutions are constrained by approximate mouth-shape classes.
Use --keep-lyrics to preserve the original words. Both workflows target the
source performance's timing and compare recognized phonemes/visemes afterward.
Scores estimate compatibility from audio; they do not verify visible lips.

Subcommands
  run      generate five random melodies with gibberish and metadata (default)
  analyze  extract F0, segment it into notes, classify registers
  plan     write a recomposed melody as JSON (no synthesis)
  render   synthesise audio (--baseline keeps the original F0 unchanged)
  measure  compare a rendered file against the original: visemes, timing, pitch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np

from visemes import compare_visemes, recognize_phonemes, viseme_of

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / ".seedvc_runtime"
SEEDVC = RUNTIME / "seed-vc"
os.environ.setdefault("HF_HOME", str(RUNTIME / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(RUNTIME / "hf_cache" / "hub"))
if str(SEEDVC) not in sys.path:
    sys.path.insert(0, str(SEEDVC))

SR = 44100
HOP = 512                 # mel hop, 11.61 ms
MEL_FPS = SR / HOP        # 86.13
F0_FPS = 100.0            # RMVPE frame rate at 16 kHz
F0_BIN_MIN = 50.0         # f0_to_coarse() quantiser limits
F0_BIN_MAX = 1100.0

SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- io

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unique_path(path: Path) -> Path:
    """Never overwrite: fall back to name_2, name_3, ... if taken."""
    if not path.exists() and not path.is_symlink():
        return path
    stem, suffix, n = path.stem, path.suffix, 2
    while True:
        cand = path.with_name(f"{stem}_{n}{suffix}")
        if not cand.exists() and not cand.is_symlink():
            print(f"[guard] {path.name} exists; writing {cand.name} instead")
            return cand
        n += 1


def write_exclusive(path: Path, data: bytes) -> Path:
    path = unique_path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return path


def save_wav(path: Path, wave: np.ndarray, sr: int = SR) -> Path:
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wave, dtype=np.float32), sr, subtype="PCM_24", format="WAV")
    return write_exclusive(path, buf.getvalue())


def midi_of(f0: np.ndarray) -> np.ndarray:
    out = np.full(f0.shape, np.nan, dtype=np.float64)
    v = f0 > 1.0
    out[v] = 69.0 + 12.0 * np.log2(f0[v] / 440.0)
    return out


def note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{PC_NAMES[m % 12]}{m // 12 - 1}"


# ------------------------------------------------------------------- f0 + notes

_RMVPE = None


def require_seedvc_source():
    if not (SEEDVC / "modules" / "rmvpe.py").is_file():
        raise SystemExit(
            f"Cacofonix needs the Seed-VC source at {SEEDVC}.\n"
            "Installing requirements.txt does not download it. Run:\n"
            f'  git clone https://github.com/Plachtaa/seed-vc.git "{SEEDVC}"\n'
            "Then run Cacofonix again."
        )


def load_rmvpe(device):
    global _RMVPE
    if _RMVPE is None:
        require_seedvc_source()
        from huggingface_hub import hf_hub_download
        from modules.rmvpe import RMVPE
        path = hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt")
        _RMVPE = RMVPE(path, is_half=False, device=device)
    return _RMVPE


def extract_f0(audio44k: np.ndarray, device) -> np.ndarray:
    """RMVPE F0 on the 16 kHz view of the audio, 100 fps."""
    import torch
    import torchaudio
    rmvpe = load_rmvpe(device)
    a = torch.from_numpy(np.asarray(audio44k, dtype=np.float32))[None]
    a16 = torchaudio.functional.resample(a, SR, 16000)
    return rmvpe.infer_from_audio(a16[0], thred=0.03)


@dataclass
class Note:
    index: int
    start: int            # inclusive frame on the F0 grid
    end: int              # exclusive
    pitch: float          # median midi over the stable core
    band: str = "mid"     # low | mid | high
    new_pitch: float = 0.0

    @property
    def start_s(self) -> float:
        return self.start / F0_FPS

    @property
    def dur_s(self) -> float:
        return (self.end - self.start) / F0_FPS


def voiced_runs(f0: np.ndarray, min_frames: int = 5):
    v = f0 > 1.0
    runs, i, n = [], 0, len(v)
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            if j - i >= min_frames:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def segment_notes(f0: np.ndarray, step_st: float = 0.75, min_dur_s: float = 0.075,
                  confirm_s: float = 0.045) -> list[Note]:
    """Split voiced regions at sustained pitch changes; vibrato must not split."""
    from scipy.ndimage import median_filter
    midi = midi_of(f0)
    min_len = max(2, int(round(min_dur_s * F0_FPS)))
    confirm = max(2, int(round(confirm_s * F0_FPS)))
    notes: list[Note] = []

    for a, b in voiced_runs(f0, min_frames=min_len):
        seg = midi[a:b]
        smooth = median_filter(seg, size=min(len(seg), 9), mode="nearest")
        bounds = [0]
        ref = float(np.median(smooth[: min(len(smooth), min_len)]))
        pending = 0
        for i in range(1, len(smooth)):
            if abs(smooth[i] - ref) > step_st:
                pending += 1
                if pending >= confirm and (i - pending + 1) - bounds[-1] >= min_len:
                    bounds.append(i - pending + 1)
                    nxt = smooth[bounds[-1]: bounds[-1] + min_len]
                    ref = float(np.median(nxt)) if len(nxt) else float(smooth[i])
                    pending = 0
            else:
                pending = 0
                lo = max(bounds[-1], i - min_len + 1)
                ref = float(np.median(smooth[lo: i + 1]))
        bounds.append(len(smooth))

        merged = [bounds[0]]
        for x in bounds[1:]:
            if x - merged[-1] >= min_len or x == bounds[-1]:
                merged.append(x)
        if len(merged) > 2 and merged[-1] - merged[-2] < min_len:
            merged.pop(-2)

        for s, e in zip(merged[:-1], merged[1:]):
            core = seg[s:e]
            trim = min(len(core) // 4, 3)          # ignore the glide into/out of the note
            if len(core) - 2 * trim >= 3:
                core = core[trim: len(core) - trim]
            notes.append(Note(index=len(notes), start=a + s, end=a + e,
                              pitch=float(np.median(core))))
    for i, nt in enumerate(notes):
        nt.index = i
    return notes


def classify_bands(notes: list[Note], low_frac: float, high_frac: float):
    pitches = np.array([n.pitch for n in notes])
    lo, hi = float(np.percentile(pitches, 5)), float(np.percentile(pitches, 95))
    span = max(hi - lo, 1e-6)
    low_thr, high_thr = lo + low_frac * span, lo + high_frac * span
    for n in notes:
        q = round(n.pitch)      # the same sung note must always land in the same band
        n.band = "low" if q <= low_thr else "high" if q >= high_thr else "mid"
    return {"p05": lo, "p95": hi, "span_st": span,
            "low_threshold": low_thr, "high_threshold": high_thr}


def detect_key(notes: list[Note]):
    w = np.zeros(12)
    for n in notes:
        w[int(round(n.pitch)) % 12] += n.dur_s
    best = None
    for tonic in range(12):
        for name, degs in SCALES.items():
            score = sum(w[(tonic + d) % 12] for d in degs)
            if best is None or score > best[0]:
                best = (score, tonic, name)
    _, tonic, name = best
    total = w.sum() or 1.0
    return {"tonic": tonic, "tonic_name": PC_NAMES[tonic], "mode": name,
            "in_scale_fraction": float(best[0] / total),
            "pitch_class_weight": {PC_NAMES[i]: round(float(w[i]), 3) for i in range(12)}}


# -------------------------------------------------------------- melody planning

def scale_pitches(tonic: int, mode: str, lo: float, hi: float) -> np.ndarray:
    degs = SCALES[mode]
    out = [m for m in range(int(np.floor(lo)) - 2, int(np.ceil(hi)) + 3)
           if (m - tonic) % 12 in degs]
    return np.array(sorted(out), dtype=float)


def plan_melody(notes: list[Note], bands: dict, key: dict, mode: str,
                max_leap: float, min_change: float, protect_short_s: float,
                pattern: tuple[int, ...], extremes: int = 0, seed: int = 0) -> dict:
    """Recompose mid-band notes; optionally nudge low/high notes within their band."""
    allowed = scale_pitches(key["tonic"], key["mode"],
                            bands["low_threshold"], bands["high_threshold"])
    inner = allowed[(allowed > bands["low_threshold"]) & (allowed < bands["high_threshold"])]
    if len(inner) < 2:
        inner = allowed
    centre = float((bands["low_threshold"] + bands["high_threshold"]) / 2.0)

    for n in notes:
        n.new_pitch = n.pitch
    nudged = nudge_extremes(notes, bands, key, extremes, np.random.default_rng(seed))

    mids = [n for n in notes if n.band == "mid" and n.dur_s >= protect_short_s]
    changed = 0
    for k, n in enumerate(mids):
        if mode == "invert":
            target = 2.0 * centre - n.pitch
        elif mode == "pattern":
            idx = int(np.argmin(np.abs(inner - n.pitch)))
            target = float(inner[int(np.clip(idx + pattern[k % len(pattern)],
                                             0, len(inner) - 1))])
        else:
            raise ValueError(f"unknown plan mode {mode!r}")

        cand = float(inner[int(np.argmin(np.abs(inner - target)))])

        # stay singable relative to the note that will actually precede it
        prev = notes[n.index - 1].new_pitch if n.index > 0 else cand
        if abs(cand - prev) > max_leap:
            near = inner[np.abs(inner - prev) <= max_leap]
            if len(near):
                cand = float(near[int(np.argmin(np.abs(near - target)))])
        if abs(cand - n.pitch) < min_change:
            alt = inner[np.abs(inner - n.pitch) >= min_change]
            if len(alt):
                near = alt[np.abs(alt - prev) <= max_leap]
                pick = near if len(near) else alt
                cand = float(pick[int(np.argmin(np.abs(pick - target)))])
        # where the original moved, the new line must move too (no merged notes)
        if n.index > 0 and abs(n.pitch - notes[n.index - 1].pitch) >= 0.75                 and abs(cand - prev) < 0.5:
            alt = inner[(np.abs(inner - prev) >= 0.5) & (np.abs(inner - prev) <= max_leap)]
            if len(alt):
                cand = float(alt[int(np.argmin(np.abs(alt - target)))])
        n.new_pitch = cand
        if abs(cand - n.pitch) >= 0.25:
            changed += 1

    return {"mode": mode, "notes_changed": changed + nudged, "mid_candidates": len(mids),
            "extremes_nudged": nudged,
            "centre_st": centre,
            "inner_grid": [note_name(p) for p in inner]}


def nudge_extremes(notes: list[Note], bands: dict, key: dict, steps: int, rng) -> int:
    """Move low/high notes by up to `steps` scale degrees without leaving their band.

    Repeated notes stay repeated and the up/down direction from the previous
    note is kept, so the contour (and the tune) stays recognisable.
    """
    if steps <= 0:
        return 0
    scale = scale_pitches(key["tonic"], key["mode"], 30, 100)

    def in_band(p, band):
        q = round(p)
        return q <= bands["low_threshold"] if band == "low" else q >= bands["high_threshold"]

    moved = 0
    for n in notes:
        if n.band == "mid":
            continue
        prev = notes[n.index - 1] if n.index > 0 else None
        if prev is not None and prev.band == n.band and abs(prev.pitch - n.pitch) < 0.5:
            n.new_pitch = n.pitch + (prev.new_pitch - prev.pitch)
            moved += abs(n.new_pitch - n.pitch) >= 0.25
            continue
        base = int(np.argmin(np.abs(scale - round(n.pitch))))
        first = int(rng.choice([-1, 1]))
        for d in (first, -first):
            done = False
            for k in range(steps, 0, -1):
                i = base + d * k
                if not 0 <= i < len(scale):
                    continue
                cand = n.pitch + (scale[i] - round(n.pitch))
                if not in_band(cand, n.band):
                    continue
                if prev is not None:
                    want = np.sign(round(n.pitch - prev.pitch))
                    if np.sign(round(cand - prev.new_pitch)) != want:
                        continue
                n.new_pitch = float(cand)
                moved += 1
                done = True
                break
            if done:
                break
    return moved


def build_f0(f0: np.ndarray, notes: list[Note], glide_s: float = 0.055) -> np.ndarray:
    """Apply per-note semitone deltas to the original F0.

    The voiced/unvoiced mask, frame grid and all micro-pitch detail (vibrato,
    drift, attack scoops) survive untouched; only note centres move. Deltas are
    smoothed inside each voiced run so a note change glides instead of stepping,
    which keeps onsets on their original frames and avoids clicks.
    """
    delta = np.zeros(len(f0), dtype=np.float64)
    for n in notes:
        delta[n.start:n.end] = n.new_pitch - n.pitch

    width = max(1, int(round(glide_s * F0_FPS)))
    if width > 1:
        win = np.hanning(2 * width + 1)
        win /= win.sum()
        for a, b in voiced_runs(f0, min_frames=1):
            seg = delta[a:b]
            if len(seg) < 3:
                continue
            pad = np.pad(seg, (width, width), mode="edge")
            delta[a:b] = np.convolve(pad, win, mode="same")[width: width + len(seg)]

    out = np.zeros_like(f0)
    v = f0 > 1.0
    out[v] = f0[v] * (2.0 ** (delta[v] / 12.0))
    out[v] = np.clip(out[v], F0_BIN_MIN + 0.5, F0_BIN_MAX - 0.5)
    return out


# ------------------------------------------------------------------- synthesis

def load_seedvc(device):
    require_seedvc_source()
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from modules.commons import build_model, load_checkpoint, recursive_munch

    cfg_path = hf_hub_download("Plachta/Seed-VC",
                               "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
    ckpt_path = hf_hub_download(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")
    with open(cfg_path, encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)
    params = recursive_munch(config["model_params"])
    params.dit_type = "DiT"
    model = build_model(params, stage="DiT")
    model, _, _, _ = load_checkpoint(model, None, ckpt_path, load_only_params=True,
                                     ignore_modules=[], is_distributed=False)
    for k in model:
        model[k].eval().to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    from modules.campplus.DTDNN import CAMPPlus
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(
        hf_hub_download("funasr/campplus", "campplus_cn_common.bin"), map_location="cpu"))
    campplus.eval().to(device)

    from modules.bigvgan import bigvgan
    vocoder = bigvgan.BigVGAN.from_pretrained(params.vocoder.name, use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    from transformers import AutoFeatureExtractor, WhisperModel
    name = params.speech_tokenizer.name
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    whisper = WhisperModel.from_pretrained(name, torch_dtype=dtype).to(device)
    del whisper.decoder
    feat_ext = AutoFeatureExtractor.from_pretrained(name)

    def semantic_fn(waves_16k):
        ins = feat_ext([waves_16k.squeeze(0).cpu().numpy()], return_tensors="pt",
                       return_attention_mask=True)
        feats = whisper._mask_input_features(ins.input_features,
                                             attention_mask=ins.attention_mask).to(device)
        with torch.no_grad():
            out = whisper.encoder(feats.to(whisper.encoder.dtype), head_mask=None,
                                  output_attentions=False, output_hidden_states=False,
                                  return_dict=True)
        return out.last_hidden_state.float()[:, : waves_16k.size(-1) // 320 + 1]

    sp = config["preprocess_params"]["spect_params"]
    mel_args = {"n_fft": sp["n_fft"], "win_size": sp["win_length"],
                "hop_size": sp["hop_length"], "num_mels": sp["n_mels"],
                "sampling_rate": config["preprocess_params"]["sr"],
                "fmin": sp.get("fmin", 0),
                "fmax": None if sp.get("fmax", "None") == "None" else 8000,
                "center": False}
    from modules.audio import mel_spectrogram
    return model, semantic_fn, vocoder, campplus, (lambda x: mel_spectrogram(x, **mel_args))


def synthesise(source44k: np.ndarray, ref44k: np.ndarray, f0_src: np.ndarray,
               f0_ref: np.ndarray, bundle, device, steps: int, cfg_rate: float,
               fp16: bool = True, content=None):
    import torch
    import torchaudio
    model, semantic_fn, vocoder, campplus, mel_fn = bundle

    n_in = len(source44k)
    pad = (-n_in) % HOP     # whole frames in, so the vocoder returns exactly n_in samples
    source44k = np.pad(np.asarray(source44k, np.float32), (0, pad))
    src = torch.from_numpy(source44k)[None].to(device)
    ref = torch.from_numpy(np.asarray(ref44k, np.float32))[None].to(device)
    src16 = torchaudio.functional.resample(src, SR, 16000)
    ref16 = torchaudio.functional.resample(ref, SR, 16000)
    if src16.size(-1) > 16000 * 30:
        raise RuntimeError("source longer than the model's 30 s context; render in phrases")

    S_src = semantic_fn(src16) if content is None else content
    S_ref = semantic_fn(ref16)
    mel_src, mel_ref = mel_fn(src.float()), mel_fn(ref.float())
    len_src = torch.LongTensor([mel_src.size(2)]).to(device)
    len_ref = torch.LongTensor([mel_ref.size(2)]).to(device)

    feat = torchaudio.compliance.kaldi.fbank(ref16, num_mel_bins=80, dither=0,
                                             sample_frequency=16000)
    style = campplus((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0))

    t_src = torch.from_numpy(np.asarray(f0_src, np.float32)).to(device)[None]
    t_ref = torch.from_numpy(np.asarray(f0_ref, np.float32)).to(device)[None]

    cond, *_ = model.length_regulator(S_src, ylens=len_src, n_quantizers=3, f0=t_src)
    prompt, *_ = model.length_regulator(S_ref, ylens=len_ref, n_quantizers=3, f0=t_ref)

    budget = SR // HOP * 30 - mel_ref.size(2)
    if cond.size(1) > budget:
        raise RuntimeError(f"source needs {cond.size(1)} frames but only {budget} fit "
                           f"beside the reference; lower --ref-dur to avoid chunk seams")

    cat = torch.cat([prompt, cond], dim=1)
    with torch.no_grad(), torch.autocast(device_type=device.type,
                                        dtype=torch.float16,
                                        enabled=fp16 and device.type == "cuda"):
        target = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(device),
                                     mel_ref, style, None, steps,
                                     inference_cfg_rate=cfg_rate)
        target = target[:, :, mel_ref.size(-1):]
    with torch.no_grad():
        wave = vocoder(target.float()).squeeze().cpu().numpy()
    if len(wave) != len(source44k):
        raise RuntimeError(f"vocoder returned {len(wave)} samples for {len(source44k)} in")
    wave = wave[:n_in]
    return wave, int(mel_src.size(2)), int(mel_ref.size(2))


def match_envelope(wave: np.ndarray, orig: np.ndarray, max_db: float = 8.0,
                   smooth_s: float = 0.03) -> np.ndarray:
    """Re-impose the original loudness contour (gain capped at +/- max_db)."""
    import librosa
    from scipy.ndimage import uniform_filter1d
    hop, frame = 256, 1024
    ro = librosa.feature.rms(y=orig, frame_length=frame, hop_length=hop)[0]
    rg = librosa.feature.rms(y=wave, frame_length=frame, hop_length=hop)[0]
    m = min(len(ro), len(rg))
    g = 20.0 * np.log10((ro[:m] + 1e-5) / (rg[:m] + 1e-5))
    g = np.clip(g, -max_db, max_db)
    # Do not boost model hiss to match very quiet source material.
    quiet = ro[:m] < max(float(ro.max()) * 10 ** (-45 / 20), 1e-6)
    g[quiet] = np.minimum(g[quiet], 0.0)
    g = uniform_filter1d(g, size=max(1, int(round(smooth_s * SR / hop))), mode="nearest")
    t = np.arange(m) * hop
    gain = 10.0 ** (np.interp(np.arange(len(wave)), t, g) / 20.0)
    out = wave * gain
    peak = np.abs(out).max()
    return out / peak * 0.98 if peak > 0.98 else out


def clean_output(wave: np.ndarray, orig: np.ndarray) -> tuple[np.ndarray, dict]:
    """Attenuate generated noise in source silences without using the F0 mask.

    Unvoiced consonants and breaths are audio too. A source RMS envelope with
    a 40 ms guard and soft fades keeps them, unlike gating on voiced frames.
    This does not remove noise underneath audible singing.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    if len(wave) != len(orig) or not len(wave):
        raise ValueError("cleanup requires nonempty, sample-aligned audio")
    if not np.isfinite(wave).all() or not np.isfinite(orig).all():
        raise ValueError("audio contains non-finite samples")
    rms = np.sqrt(np.maximum(uniform_filter1d(
        np.asarray(orig, dtype=np.float64) ** 2, size=441, mode="nearest"), 0))
    threshold = max(float(rms.max()) * 10 ** (-50 / 20), 10 ** (-70 / 20))
    activity = np.clip((rms - threshold) / (threshold * 2), 0, 1)
    activity = maximum_filter1d(activity, size=3529, mode="nearest")
    gain = uniform_filter1d(activity, size=883, mode="nearest")
    gain = 0.01 + 0.99 * gain  # up to 40 dB attenuation, smoothly applied
    out = np.asarray(wave, dtype=np.float64) * gain
    peak = float(np.max(np.abs(out)))
    scale = min(1.0, 0.98 / max(peak, 1e-12))
    return (out * scale).astype(np.float32), {
        "method": "source-guided silence attenuation", "max_attenuation_db": 40,
        "source_rms_threshold": threshold, "peak_scale": scale,
        "attenuated_sample_fraction": float(np.mean(gain < 0.99)),
    }


# ---------------------------------------------------------------------- lyrics
#
# New words are spoken by a TTS voice, then warped syllable by syllable onto the
# original performance: consonant onset, vowel start, coda start and syllable
# end of every new syllable are pinned to the matching points of the original
# syllable. Seed-VC then sings the warped content with the planned F0 in the
# original singer's voice. Pauses in the original stay silent.

SIL_T = 0.1   # a moment inside the TTS lead-in silence


def lyric_words(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("‘", "'")
    words = []
    for w in re.split(r"[\s/,.!?;:\"()\[\]]+", text):
        w = re.sub(r"[^a-z'\-]", "", w).strip("-'")
        if re.search(r"[a-z]", w):
            words.append(w)
    return words


def _is_vowel(word: str, i: int) -> bool:
    return word[i] in "aeiou" or (word[i] == "y" and i > 0)


def split_syllables(word: str) -> list[str]:
    """Hyphens are honoured ("ev-er-more"); otherwise a vowel-group heuristic."""
    if "-" in word:
        return [p for p in word.split("-") if re.search(r"[a-z]", p)]
    groups, i = [], 0
    while i < len(word):
        if _is_vowel(word, i):
            j = i
            while j < len(word) and _is_vowel(word, j):
                j += 1
            groups.append((i, j))
            i = j
        else:
            i += 1
    if (len(groups) > 1 and groups[-1] == (len(word) - 1, len(word)) and word.endswith("e")
            and not (word.endswith("le") and len(word) > 2 and not _is_vowel(word, len(word) - 3))):
        groups.pop()                       # silent final e
    if len(groups) <= 1:
        return [word]
    cuts = []
    for (_, b1), (a2, _) in zip(groups, groups[1:]):
        cuts.append(b1 if a2 - b1 <= 1 else b1 + 1)
    parts, prev = [], 0
    for c in cuts:
        parts.append(word[prev:c])
        prev = c
    parts.append(word[prev:])
    return parts


def syllable_table(words: list[str], splits=None) -> list[dict]:
    table = []
    for wi, w in enumerate(words):
        flat, off = w.replace("-", ""), 0
        for p in (splits[wi] if splits else split_syllables(w)):
            table.append({"text": p, "word": wi, "c0": off, "c1": off + len(p), "flat": flat})
            off += len(p)
    return table


_FA = None


def align_chars(audio16k: np.ndarray, words: list[str], device):
    """Forced alignment (torchaudio MMS_FA): per word, per char (start_s, end_s)."""
    global _FA
    import torch
    import torchaudio
    if _FA is None:
        os.environ.setdefault("TORCH_HOME", str(RUNTIME / "torch_home"))
        b = torchaudio.pipelines.MMS_FA
        _FA = (b.get_model(with_star=False).to(device), b.get_tokenizer(), b.get_aligner())
    model, tok, aligner = _FA
    flat = [w.replace("-", "") for w in words]
    with torch.inference_mode():
        em, _ = model(torch.from_numpy(np.asarray(audio16k, np.float32))[None].to(device))
        spans = aligner(em[0], tok(flat))
    ratio = len(audio16k) / 16000 / em.shape[1]
    return [[(s.start * ratio, s.end * ratio) for s in sp] for sp in spans]


def sound_end(audio16k: np.ndarray, t_from: float, t_cap: float,
              thr_db: float = -32.0, hold_s: float = 0.06) -> float:
    import librosa
    hop = 160
    rms = librosa.feature.rms(y=audio16k, frame_length=640, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-6)
    db -= db.max()
    i, quiet, hold = int(t_from * 100), 0, max(1, int(hold_s * 100))
    while i < len(db) and i / 100 < t_cap:
        quiet = quiet + 1 if db[i] < thr_db else 0
        if quiet >= hold:
            return min(t_cap, (i - hold + 1) / 100)
        i += 1
    return min(t_cap, len(db) / 100)


def syllable_anchors(audio16k, words, device, snap_to=None, snap_s=0.1, splits=None):
    """S (onset), V (vowel), C (coda) and E (end) times for every syllable."""
    chars = align_chars(audio16k, words, device)
    syl = syllable_table(words, splits)
    for s in syl:
        t = chars[s["word"]][s["c0"]:s["c1"]]
        flat = s["flat"]
        idx = [i for i in range(s["c0"], s["c1"]) if flat[i] != "'"]
        vow = [i for i in idx if _is_vowel(flat, i)] or idx[:1]
        s["S"] = t[idx[0] - s["c0"]][0]
        s["V"] = t[vow[0] - s["c0"]][0]
        after = [i for i in idx if i > vow[-1]]
        s["C"] = t[after[0] - s["c0"]][0] if after else None
        s["V_raw"] = s["V"]
    if snap_to is not None and len(snap_to):
        for s in syl:
            j = int(np.argmin(np.abs(snap_to - s["V"])))
            if abs(snap_to[j] - s["V"]) <= snap_s:
                shift = snap_to[j] - s["V"]
                s["V"] = float(snap_to[j])
                if s["C"] is not None:
                    s["C"] = max(s["C"], s["V"] + 0.02)
                s["S"] = min(s["S"], s["V"])
                s["snap_ms"] = round(shift * 1000)
    dur = len(audio16k) / 16000
    for i, s in enumerate(syl):
        nxt = syl[i + 1]["S"] if i + 1 < len(syl) else dur
        s["S"] = min(s["S"], s["V"])
        s["E"] = max(sound_end(audio16k, s["V"], nxt), s["V"] + 0.03)
        s["E"] = min(s["E"], nxt) if nxt > s["V"] else s["E"]
        if s["C"] is None or not (s["V"] < s["C"] < s["E"]):
            s["C"] = s["E"]
    return syl


def build_warp(orig: list[dict], new: list[dict]):
    """Piecewise-linear map from original time to TTS time, per syllable."""
    segs, prev_end = [], 0.0
    for o, n in zip(orig, new):
        S, V, C, E = o["S"], o["V"], o["C"], o["E"]
        s, v, c, e = n["S"], n["V"], n["C"], n["E"]
        if v - s > 0.02 and V - S < 0.02:          # new onset consonant needs room
            S = max(prev_end, V - min(v - s, 0.10))
        if e - c > 0.02 and E - C < 0.02:          # new coda consonant needs room
            C = max(V + 0.4 * (E - V), E - min(e - c, 0.12))
        segs.append({"S": S, "V": V, "C": C, "E": E, "s": s, "v": v, "c": c, "e": e})
        prev_end = E

    def lookup(t: float):
        for g in segs:
            if g["S"] <= t < g["E"]:
                for (A, B, a, b, part) in ((g["S"], g["V"], g["s"], g["v"], "onset"),
                                           (g["V"], g["C"], g["v"], g["c"], "vowel"),
                                           (g["C"], g["E"], g["c"], g["e"], "coda")):
                    if A <= t < B:
                        return a + (b - a) * (t - A) / max(B - A, 1e-6), part
        return None, "pause"
    return segs, lookup


def tts_speak(text: str, voice: str, speed: float, phonemes: str | None = None) -> np.ndarray:
    """Kokoro TTS at 24 kHz, with a short lead-in silence."""
    pipe = kokoro_pipe()
    gen = (pipe.generate_from_tokens(phonemes, voice=voice, speed=speed) if phonemes
           else pipe(text, voice=voice, speed=speed))
    chunks = [r.audio.numpy() for r in gen]
    audio = np.concatenate(chunks) if chunks else np.zeros(24000, np.float32)
    lead = np.zeros(int(0.3 * 24000), np.float32)
    return np.concatenate([lead, audio, lead])


def transcribe(audio44k: np.ndarray, device) -> str:
    import librosa
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    name = "openai/whisper-small"
    processor = AutoProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name).to(device).eval()
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    a16 = librosa.resample(audio44k, orig_sr=SR, target_sr=16000)
    features = processor.feature_extractor(a16, sampling_rate=16000,
                                           return_tensors="pt").input_features.to(device)
    with torch.inference_mode():
        tokens = model.generate(input_features=features, language="en", task="transcribe")
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def lyrics_conditioning(piece44k, f0_plan, notes, orig_text, new_text, voice, speed,
                        semantic_fn, device, splits=None, phonemes=None):
    """Content features (50 fps) and F0 (100 fps) that sing new words on old timing."""
    import librosa
    import torch
    ow = lyric_words(orig_text)
    nw = ["".join(p) for p in splits] if splits else lyric_words(new_text)
    o_tab, n_tab = syllable_table(ow), syllable_table(nw, splits)
    if len(o_tab) != len(n_tab):
        msg = (f"syllable count differs: original {len(o_tab)} vs new {len(n_tab)}\n"
               f"  original: {' '.join(s['text'] for s in o_tab)}\n"
               f"  new:      {' '.join(s['text'] for s in n_tab)}\n"
               "Use hyphens to split words (ev-er-more) so both lines have the same count.")
        sys.exit(msg)

    orig16 = librosa.resample(piece44k, orig_sr=SR, target_sr=16000)
    onsets = np.array(sorted({n.start / F0_FPS for n in notes} |
                             {a / F0_FPS for a, _ in voiced_runs(f0_plan, 3)}))
    o_syl = syllable_anchors(orig16, ow, device, snap_to=onsets)
    runs = voiced_runs(f0_plan, 3)
    for i, o in enumerate(o_syl):
        nxt = o_syl[i + 1]["S"] if i + 1 < len(o_syl) else len(orig16) / 16000
        ends = [b / F0_FPS for a, b in runs if a / F0_FPS < nxt and b / F0_FPS > o["V"]]
        coda = 0.12 if o["C"] < o["E"] else 0.03
        o["E"] = min(nxt, max(ends) + coda) if ends else min(nxt, o["V"] + 0.15)
        o["E"] = max(o["E"], o["V"] + 0.03)
        if not (o["V"] < o["C"] < o["E"]):
            o["C"] = o["E"]

    tts24 = tts_speak(new_text, voice, speed, phonemes)
    tts16 = librosa.resample(tts24, orig_sr=24000, target_sr=16000)
    n_syl = syllable_anchors(tts16, nw, device, splits=splits)

    segs, lookup = build_warp(o_syl, n_syl)

    S_tts = semantic_fn(torch.from_numpy(tts16)[None].to(device))
    # breaks the original singer takes inside a syllable stay silent
    gap = f0_plan <= 1.0
    long_gap = np.zeros_like(gap)
    i = 0
    while i < len(gap):
        if gap[i]:
            j = i
            while j < len(gap) and gap[j]:
                j += 1
            if j - i >= 6:
                long_gap[i:j] = True
            i = j
        else:
            i += 1

    n_out = len(orig16) // 320 + 1
    idx = np.empty(n_out, dtype=np.int64)
    for k in range(n_out):
        u, part = lookup(k / 50.0)
        jf = min(len(long_gap) - 1, int(round(k / 50.0 * F0_FPS)))
        if u is None or (part == "vowel" and long_gap[jf]):
            u = SIL_T
        idx[k] = int(round(u * 50.0))
    idx = np.clip(idx, 0, S_tts.size(1) - 1)
    content = S_tts[:, torch.from_numpy(idx).to(S_tts.device)]

    tts44 = librosa.resample(tts24, orig_sr=24000, target_sr=SR)
    tts_v = extract_f0(tts44, device) > 1.0
    m = midi_of(f0_plan)
    have = np.where(np.isfinite(m))[0]
    m_full = np.interp(np.arange(len(m)), have, m[have])
    orig_v = f0_plan > 1.0
    vowel_at = np.array([o["V"] for o in o_syl])
    f0 = np.zeros_like(f0_plan)
    for j in range(len(f0)):
        t = j / F0_FPS
        u, part = lookup(t)
        if part == "vowel":
            voiced = bool(orig_v[j])
        elif part in ("onset", "coda"):
            tv = bool(tts_v[min(len(tts_v) - 1, int(round(u * F0_FPS)))])
            lead = part == "onset" and np.any((vowel_at - t > 0) & (vowel_at - t <= 0.08))
            voiced = tv and (bool(orig_v[j]) or lead)
        else:
            voiced = False
        if voiced:
            f0[j] = 440.0 * 2.0 ** ((m_full[j] - 69.0) / 12.0)
    f0 = np.where(f0 > 0, np.clip(f0, F0_BIN_MIN + 0.5, F0_BIN_MAX - 0.5), 0.0)

    table = [{"orig": o["text"], "new": n["text"],
              "orig_onset": round(o["S"], 3), "orig_vowel": round(o["V"], 3),
              "orig_vowel_aligner": round(o["V_raw"], 3),
              "orig_end": round(o["E"], 3), "snap_ms": o.get("snap_ms")}
             for o, n in zip(o_syl, n_syl)]
    return content, f0, {"syllables": table, "orig_words": ow, "new_words": nw,
                         "tts_seconds": len(tts24) / 24000}


def lyric_timing_check(wave44k, new_text, table, device, splits=None) -> dict:
    """Re-align the rendered audio to the new words; compare vowel starts.

    Both sides use the same aligner without snapping, so its bias cancels.
    """
    import librosa
    out16 = librosa.resample(np.asarray(wave44k, np.float32), orig_sr=SR, target_sr=16000)
    words = ["".join(p) for p in splits] if splits else lyric_words(new_text)
    got = syllable_anchors(out16, words, device, splits=splits)
    rows, d = [], []
    for row, g in zip(table, got):
        delta = (g["V_raw"] - row["orig_vowel_aligner"]) * 1000
        d.append(delta)
        rows.append({**row, "rendered_vowel_aligner": round(g["V_raw"], 3),
                     "delta_ms": round(delta)})
    d = np.abs(np.array(d))
    return {"rows": rows, "median_abs_ms": round(float(np.median(d))),
            "p90_abs_ms": round(float(np.percentile(d, 90))), "max_abs_ms": round(float(d.max()))}


# ------------------------------------------------------------------ gibberish
#
# Viseme-constrained nonsense lyrics: substitutions stay in the same approximate
# mouth-shape class. Rendered audio must still be checked independently. Keep
# syllable count and consonant clusters intact. Symbols are misaki/Kokoro
# phonemes; diphthongs are single letters (A=ei, I=ai, W=au, Y=oi, O=ou).

CONS_GROUPS = ["pbm", "fv", "θ", "ð", "tdnɾ", "l", "sz", "ʃʒʧʤ", "kɡŋ", "h", "w", "ɹ", "j"]
VOWEL_GROUPS = ["æʌəɐ", "ɑ", "ᵊ", "ɛ", "A", "iɪᵻ", "u", "ʊ", "O", "ɔ", "ɜɚ", "I", "W", "Y"]
VOWEL_SET = set("".join(VOWEL_GROUPS))
# spellings only for the aligner and for display; vowels use vowel letters only
SPELL = {"p": "p", "b": "b", "m": "m", "f": "f", "v": "v", "θ": "th", "ð": "th",
         "t": "t", "d": "d", "n": "n", "l": "l", "ɾ": "t", "s": "s", "z": "z",
         "ʃ": "sh", "ʒ": "zh", "ʧ": "ch", "ʤ": "j", "k": "k", "ɡ": "g", "h": "h",
         "w": "w", "ɹ": "r", "j": "y", "ŋ": "ng",
         "æ": "a", "ɑ": "a", "ʌ": "u", "ə": "u", "ɐ": "u", "ᵊ": "u", "ɛ": "e",
         "A": "ei", "i": "ee", "ɪ": "i", "ᵻ": "i", "u": "oo", "ʊ": "u", "O": "o",
         "ɔ": "ao", "ɜ": "e", "ɚ": "e", "I": "ai", "W": "au", "Y": "oi"}

_KPIPE = None


def kokoro_pipe():
    global _KPIPE
    if _KPIPE is None:
        from kokoro import KPipeline
        _KPIPE = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _KPIPE


def _group(c: str) -> str:
    return next((g for g in VOWEL_GROUPS + CONS_GROUPS if c in g), c)


def replacement_phones(phone: str) -> list[str]:
    """Prefer distinct sounds over reduced vowels and allophonic variants."""
    # Prefer a larger change within the same mouth shape: avoid reduced-vowel
    # swaps and choose n instead of a t/d/flap swap when it is available.
    nearby_variants = ("ʌəɐᵊ", "ɪᵻ", "ɜɚ", "tdɾ")
    alternatives = [p for p in _group(phone) if p != phone]
    preferred = [p for p in alternatives if p not in "əɐᵻɾɚ"
                 and not any(phone in group and p in group for group in nearby_variants)]
    return preferred


def _phone_units(ph: str):
    units, stress = [], ""
    for ch in ph:
        if ch in "ˈˌ":
            stress = ch
        elif ch in VOWEL_SET:
            units.append((ch, stress))
            stress = ""
        elif ch in SPELL:
            units.append((ch, ""))
        elif not ch.isspace() and ch not in "ːˑ.,;:!?…":
            raise ValueError(f"Unsupported phoneme {ch!r}; cannot safely preserve its viseme.")
    return units


def _phone_syllables(units):
    nuc = [i for i, (c, _) in enumerate(units) if c in VOWEL_SET]
    if len(nuc) <= 1:
        return [units]
    cuts = [a + 1 if b - a - 1 <= 1 else a + 2 for a, b in zip(nuc, nuc[1:])]
    parts, prev = [], 0
    for c in cuts:
        parts.append(units[prev:c])
        prev = c
    parts.append(units[prev:])
    return parts


def gibberize(orig_text: str, rng, return_details=False):
    """Returns (syllable spellings per word, phoneme string for TTS, display text)."""
    pipe = kokoro_pipe()
    out_parts, out_ph, substitutions, original_phones = [], [], [], []
    aliases = {"A": "eɪ", "O": "oʊ", "I": "aɪ", "W": "aʊ", "Y": "ɔɪ", "ᵊ": "ə"}
    for w in lyric_words(orig_text):
        ph, _ = pipe.g2p(w.replace("-", ""))
        original_phones.append(ph)
        syl = _phone_syllables(_phone_units(ph))
        if not syl or not any(syl):
            raise ValueError(f"No supported phonemes found for {w!r}.")
        target = len(split_syllables(w))
        while len(syl) > target:
            syl[-2] = syl[-2] + syl[-1]
            syl.pop()
        if len(syl) < target:
            sys.exit(f"could not split '{w}' into {target} syllables from its phonemes "
                     f"({ph}); hyphenate it in --orig-lyrics to match how it is sung")
        new_syl = []
        for s in syl:
            ns = []
            for c, st in s:
                alternatives = replacement_phones(c)
                if alternatives:
                    c = str(rng.choice(alternatives))
                ns.append((c, st))
            new_syl.append(ns)
            for (old, _), (new, _) in zip(s, ns):
                old_viseme = viseme_of(aliases.get(old, old))
                new_viseme = viseme_of(aliases.get(new, new))
                if old_viseme is None or old_viseme != new_viseme:
                    raise ValueError(f"Substitution {old!r} -> {new!r} changes its viseme.")
                substitutions.append({"word": w, "original": old, "replacement": new,
                                      "viseme": old_viseme})
        out_parts.append(["".join(SPELL[c] for c, _ in s) for s in new_syl])
        out_ph.append("".join(st + c for s in new_syl for c, st in s))
    display = " ".join("-".join(p) for p in out_parts)
    result = (out_parts, " ".join(out_ph), display)
    if return_details:
        return (*result, {"original_phonemes": " ".join(original_phones),
                          "substitutions": substitutions, "planned_visemes_preserved": True,
                          "changed_phonemes": sum(s["original"] != s["replacement"]
                                                   for s in substitutions)})
    return result


# ------------------------------------------------------------------ measurement

def envelope(audio: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.onset.onset_strength(y=np.asarray(audio, np.float32), sr=SR,
                                        hop_length=HOP)


def local_lags(a: np.ndarray, b: np.ndarray, win_s: float = 0.5, max_lag_s: float = 0.14):
    """Per-window best alignment lag of b against a, in milliseconds."""
    w = int(round(win_s * MEL_FPS))
    m = int(round(max_lag_s * MEL_FPS))
    n = min(len(a), len(b))
    out = []
    for s in range(m, n - w - m, w):
        ref = a[s: s + w]
        if ref.std() < 1e-6:
            continue
        best, best_lag = -2.0, 0
        for lag in range(-m, m + 1):
            seg = b[s + lag: s + lag + w]
            if len(seg) != w or seg.std() < 1e-6:
                continue
            c = float(np.corrcoef(ref, seg)[0, 1])
            if c > best:
                best, best_lag = c, lag
        out.append({"t": round(s / MEL_FPS, 3),
                    "lag_ms": round(best_lag / MEL_FPS * 1000, 1) if best > -2 else None,
                    "corr": round(best, 3)})
    return out


def lip_sync_score(report: dict) -> dict:
    """Uncalibrated audio timing heuristic, not a visual lip-sync probability."""
    windows = report["local_timing_lag_ms"]["windows"]
    reliable = [w for w in windows if w["lag_ms"] is not None and w["corr"] >= 0.5]
    coverage = len(reliable) / len(windows) if windows else 0.0
    duration_delta = abs(report["duration_s"]["original"] -
                         report["duration_s"]["rendered"]) * 1000
    agreement = report["voicing"]["frame_agreement"]
    result = {
        "score": None, "status": "inconclusive", "kind": "audio timing heuristic",
        "visual_lip_sync_verified": False,
        "reliable_windows": len(reliable), "total_windows": len(windows),
        "reliable_window_fraction": round(coverage, 3),
        "duration_difference_ms": round(duration_delta, 3),
        "voicing_agreement": agreement,
        "limitation": "Does not verify words, phonemes, or visible mouth shapes. "
                      "Melody changes can affect acoustic onset measurements.",
        "scoring": "v1: 60% lag (0 at p90 >= 140 ms), 30% voicing agreement, "
                   "10% duration (0 at difference >= 100 ms); multiplied by reliable "
                   "window fraction. Requires >= 3 reliable windows and >= 50% coverage.",
    }
    if len(reliable) < 3 or coverage < 0.5 or agreement is None:
        return result
    p90 = float(np.percentile([abs(w["lag_ms"]) for w in reliable], 90))
    score = 100 * coverage * (0.6 * max(0, 1 - p90 / 140) +
                              0.3 * agreement + 0.1 * max(0, 1 - duration_delta / 100))
    result.update({"score": int(round(score)), "p90_lag_ms": round(p90, 2),
                   "status": "audio_timing_consistent" if (
                       p90 <= 40 and agreement >= 0.9 and duration_delta <= 10 and
                       coverage >= 0.8) else "review_needed"})
    return result


def timing_report(orig44k: np.ndarray, out44k: np.ndarray, f0_plan: np.ndarray,
                  device) -> dict:
    eo, eg = envelope(orig44k), envelope(out44k)
    n = min(len(eo), len(eg))
    lags = local_lags(eo[:n], eg[:n])
    lag_vals = np.array([l["lag_ms"] for l in lags if l["lag_ms"] is not None])

    f0o = extract_f0(orig44k, device)
    f0g = extract_f0(out44k, device)
    m = min(len(f0o), len(f0g), len(f0_plan))
    vo, vg = f0o[:m] > 1.0, f0g[:m] > 1.0
    both = vo & vg & (f0_plan[:m] > 1.0)
    plan_v = f0_plan[:m] > 1.0

    err = np.array([])
    if both.any():
        err = 12.0 * np.log2(f0g[:m][both] / f0_plan[:m][both])

    env_corr = (float(np.corrcoef(eo[:n], eg[:n])[0, 1])
                if n > 2 and eo[:n].std() > 1e-6 and eg[:n].std() > 1e-6 else None)
    report = {
        "duration_s": {"original": len(orig44k) / SR, "rendered": len(out44k) / SR},
        "samples": {"original": int(len(orig44k)), "rendered": int(len(out44k))},
        "envelope_correlation": round(env_corr, 4) if env_corr is not None else None,
        "local_timing_lag_ms": {
            "mean_abs": round(float(np.abs(lag_vals).mean()), 2) if lag_vals.size else None,
            "median_abs": round(float(np.median(np.abs(lag_vals))), 2) if lag_vals.size else None,
            "p90_abs": round(float(np.percentile(np.abs(lag_vals), 90)), 2) if lag_vals.size else None,
            "max_abs": round(float(np.abs(lag_vals).max()), 2) if lag_vals.size else None,
            "windows": lags,
        },
        "voicing": {
            "frame_agreement": round(float((vo == vg).mean()), 4) if m else None,
            "original_voiced_frac": round(float(vo.mean()), 4) if m else None,
            "rendered_voiced_frac": round(float(vg.mean()), 4) if m else None,
        },
        "pitch_vs_plan_semitones": {
            "frames_compared": int(both.sum()),
            "mean_abs": round(float(np.abs(err).mean()), 3) if err.size else None,
            "median_abs": round(float(np.median(np.abs(err))), 3) if err.size else None,
            "p95_abs": round(float(np.percentile(np.abs(err), 95)), 3) if err.size else None,
        },
        "plan_vs_original_voicing_identical": bool((plan_v == vo).all()),
    }
    report["audio_timing"] = lip_sync_score(report)
    return report


# ------------------------------------------------------------------ subcommands

def _load(path: Path):
    import librosa
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _analysis(args):
    src = Path(args.source).resolve()
    digest = sha256(src)
    audio = _load(src)
    f0 = extract_f0(audio, _device())
    notes = segment_notes(f0, step_st=args.step_st, min_dur_s=args.min_note)
    if not notes:
        raise ValueError("No pitched notes detected; use a clear sung vocal recording.")
    bands = classify_bands(notes, args.low_frac, args.high_frac)
    key = detect_key(notes)
    return src, digest, audio, f0, notes, bands, key


def cmd_analyze(args):
    src, digest, audio, f0, notes, bands, key = _analysis(args)
    print(f"source      {src.name}  sha256={digest[:16]}...  {len(audio)/SR:.6f} s")
    print(f"f0 frames   {len(f0)}  voiced {(f0>1).mean()*100:.1f}%  "
          f"range {f0[f0>1].min():.1f}-{f0[f0>1].max():.1f} Hz")
    print(f"key guess   {key['tonic_name']} {key['mode']}  "
          f"(in-scale weight {key['in_scale_fraction']*100:.0f}%)")
    print(f"bands       low <= {note_name(bands['low_threshold'])} "
          f"({bands['low_threshold']:.2f})   high >= {note_name(bands['high_threshold'])} "
          f"({bands['high_threshold']:.2f})   span {bands['span_st']:.1f} st")
    print(f"\n{len(notes)} notes")
    print(f"{'#':>3} {'start':>7} {'dur':>6} {'pitch':>7} {'note':>6} {'band':>5}")
    for n in notes:
        print(f"{n.index:>3} {n.start_s:>7.3f} {n.dur_s:>6.3f} {n.pitch:>7.2f} "
              f"{note_name(n.pitch):>6} {n.band:>5}")
    counts = {b: sum(1 for n in notes if n.band == b) for b in ("low", "mid", "high")}
    print(f"\nband counts {counts}")
    if args.json:
        payload = {"source": str(src), "sha256": digest, "key": key, "bands": bands,
                   "notes": [asdict(n) for n in notes]}
        p = write_exclusive(Path(args.json), json.dumps(payload, indent=2).encode())
        print(f"wrote {p}")


def cmd_plan(args):
    src, digest, audio, f0, notes, bands, key = _analysis(args)
    info = plan_melody(notes, bands, key, args.mode, args.max_leap, args.min_change,
                       args.protect_short, tuple(int(x) for x in args.pattern.split(",")),
                       args.extremes, args.seed)
    print(f"key {key['tonic_name']} {key['mode']}   mode={info['mode']}   "
          f"changed {info['notes_changed']}/{info['mid_candidates']} mid notes")
    print(f"{'#':>3} {'start':>7} {'dur':>6} {'band':>5} {'from':>6} {'to':>6} {'dst':>6}")
    for n in notes:
        d = n.new_pitch - n.pitch
        flag = "" if abs(d) < 0.25 else " *"
        print(f"{n.index:>3} {n.start_s:>7.3f} {n.dur_s:>6.3f} {n.band:>5} "
              f"{note_name(n.pitch):>6} {note_name(n.new_pitch):>6} {d:>+6.2f}{flag}")
    payload = {"source": str(src), "sha256": digest, "key": key, "bands": bands,
               "plan": info,
               "params": {"mode": args.mode, "seed": args.seed, "extremes": args.extremes,
                          "max_leap": args.max_leap,
                          "min_change": args.min_change,
                          "protect_short": args.protect_short,
                          "pattern": args.pattern, "step_st": args.step_st,
                          "min_note": args.min_note, "low_frac": args.low_frac,
                          "high_frac": args.high_frac},
               "notes": [asdict(n) for n in notes]}
    p = write_exclusive(Path(args.out), json.dumps(payload, indent=2).encode())
    print(f"\nwrote {p}")


def cmd_render(args, bundle=None, source_audio=None, source_phones=None):
    device = _device()
    src = Path(args.source).resolve()
    digest_before = sha256(src)
    audio = _load(src) if source_audio is None else source_audio
    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is None:
        args.seed = secrets.randbits(32)

    s0 = int(round(args.start * SR))
    s1 = len(audio) if args.dur <= 0 else min(len(audio), s0 + int(round(args.dur * SR)))
    piece = audio[s0:s1]
    if not len(piece):
        raise ValueError("Selected source segment is empty.")
    f0 = extract_f0(piece, device)

    if args.baseline:
        f0_new, notes = f0.copy(), []
        label = getattr(args, "baseline_label", None) or "baseline"
    else:
        plan = json.loads(Path(args.plan).read_text())
        notes = [Note(**n) for n in plan["notes"]]
        shift = s0 / SR * F0_FPS
        kept = []
        for n in notes:
            n.start, n.end = int(round(n.start - shift)), int(round(n.end - shift))
            if n.end > 0 and n.start < len(f0):
                n.start, n.end = max(0, n.start), min(len(f0), n.end)
                kept.append(n)
        notes = kept
        f0_new = build_f0(f0, notes, glide_s=args.glide)
        label = Path(args.plan).stem.removeprefix(f"{src.stem}_")
        if getattr(args, "melody_only_label", None):
            label = args.melody_only_label

    r0 = int(round(args.ref_start * SR))
    r1 = len(audio) if args.ref_dur <= 0 else min(len(audio),
                                                  r0 + int(round(args.ref_dur * SR)))
    ref = audio[r0:r1]
    if not len(ref):
        raise ValueError("Selected reference segment is empty.")
    f0_ref = extract_f0(ref, device)

    print(f"source  {src.name}  segment {s0/SR:.3f}-{s1/SR:.3f} s  ({len(piece)} samples)")
    print(f"ref     {r0/SR:.3f}-{r1/SR:.3f} s  ({len(ref)/SR:.2f} s)")
    print(f"f0      {len(f0)} frames, voiced {(f0>1).mean()*100:.1f}%; "
          f"mask identical: {bool(((f0_new>1)==(f0>1)).all())}")
    if notes:
        ch = sum(1 for n in notes if abs(n.new_pitch - n.pitch) >= 0.25)
        print(f"plan    {label}: {ch} of {len(notes)} notes repitched")

    if bundle is None:
        bundle = load_seedvc(device)
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    content, lyr = None, None
    if getattr(args, "lyrics", None):
        orig_text = args.orig_lyrics or transcribe(piece, device)
        splits = phonemes = None
        lyric_plan = None
        new_text = args.lyrics
        if new_text.strip().lower() in ("auto", "gibberish"):
            splits, phonemes, new_text, lyric_plan = gibberize(
                orig_text, np.random.default_rng(args.seed), return_details=True)
        print(f"words   original: {orig_text}")
        print(f"        new:      {new_text}" + ("   (lip-matched gibberish)" if splits else ""))
        content, f0_new, lyr = lyrics_conditioning(piece, f0_new, notes, orig_text,
                                                   new_text, args.tts_voice,
                                                   args.tts_speed, bundle[1], device,
                                                   splits=splits, phonemes=phonemes)
        lyr["orig_text"], lyr["new_text"], lyr["phonemes"] = orig_text, new_text, phonemes
        lyr["splits"] = splits
        lyr["viseme_plan"] = lyric_plan
        print(f"        {len(lyr['syllables'])} syllables mapped "
              f"(TTS {lyr['tts_seconds']:.2f} s warped onto the original timing)")
        for r in lyr["syllables"]:
            print(f"          {r['orig']:>8} -> {r['new']:<8} vowel at {r['orig_vowel']:6.2f} s")
        label += "_gibberish" if splits else "_lyrics"

    t0 = time.time()
    # Isolate diffusion randomness from lazy TTS/model initialization above.
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        wave, n_mel_src, n_mel_ref = synthesise(piece, ref, f0_new, f0_ref, bundle, device,
                                                args.steps, args.cfg_rate, content=content)
    print(f"render  {time.time()-t0:.1f} s   mel frames src={n_mel_src} ref={n_mel_ref}   "
          f"out {len(wave)} samples ({len(wave)/SR:.6f} s)")

    stem = f"{src.stem}_{label}_{int(args.start*1000)}ms_{args.steps}st_s{args.seed}"
    raw_path = None
    if args.keep_raw:
        raw_path = save_wav(out_dir / f"{stem}_raw.wav", wave)
        print(f"wrote {raw_path.name}  (model output, no envelope match)")
    if args.envelope:
        wave = match_envelope(wave, piece, max_db=args.envelope_db,
                              smooth_s=0.08 if lyr else 0.03)
    cleanup = None
    if args.cleanup:
        wave, cleanup = clean_output(wave, piece)
    rep = timing_report(piece, wave, f0_new, device)
    if args.viseme_check:
        print("Checking recognized phonemes and viseme timing against the original...")
        if source_phones is None:
            source_phones = recognize_phonemes(piece, SR)
        rendered_phones = recognize_phonemes(wave, SR)
        rep["lip_sync"] = compare_visemes(source_phones, rendered_phones)
        rep["phoneme_recognition"] = {"source": source_phones, "rendered": rendered_phones}
    else:
        rep["lip_sync"] = {"score": None, "status": "disabled",
                           "kind": "recognized phoneme/viseme timing",
                           "visual_lip_sync_verified": False}
    score = rep["lip_sync"]["score"]
    stem += f"_sync{score:03d}" if score is not None else "_syncNA"
    out = save_wav(out_dir / f"{stem}.wav", wave)
    print(f"wrote {out.name}")

    cmp_path = None
    if args.export_reference_excerpt and (s0 or s1 != len(audio)):
        cmp_path = save_wav(out_dir / f"{src.stem}_origexcerpt_{int(args.start*1000)}ms.wav",
                            piece)
        print(f"wrote {cmp_path.name}  (A/B against the original)")

    if lyr:
        lyr["check"] = lyric_timing_check(wave, lyr["new_text"], lyr["syllables"], device,
                                          splits=lyr["splits"])
        rep["lyrics"] = lyr
    rep.update({"source": str(src), "source_sha256_before": digest_before,
                "source_sha256_after": sha256(src), "output": str(out),
                "original_excerpt": str(cmp_path) if cmp_path else None,
                "segment_s": [s0 / SR, s1 / SR], "reference_s": [r0 / SR, r1 / SR],
                "plan": label, "seed": args.seed, "cleanup": cleanup,
                "diffusion_steps": args.steps, "cfg_rate": args.cfg_rate,
                "glide_s": args.glide, "raw_output": str(raw_path) if raw_path else None,
                "envelope_match_db": args.envelope_db if args.envelope else None,
                "mel_frames": {"source": n_mel_src, "reference": n_mel_ref}})
    rep["source_unchanged"] = rep["source_sha256_before"] == rep["source_sha256_after"]
    p = write_exclusive(out_dir / f"{stem}_report.json", json.dumps(rep, indent=2).encode())

    t = rep["local_timing_lag_ms"]
    print("\n-- timing --")
    print(f"envelope corr      {rep['envelope_correlation']}")
    print(f"local lag |ms|     mean {t['mean_abs']}  median {t['median_abs']}  "
          f"p90 {t['p90_abs']}  max {t['max_abs']}")
    print(f"voicing agreement  {rep['voicing']['frame_agreement']}")
    print(f"viseme sync score  {score if score is not None else 'N/A'}/100 "
          f"({rep['lip_sync']['status']}; visual lip-sync not verified)")
    pv = rep["pitch_vs_plan_semitones"]
    print(f"pitch vs plan      mean |d| {pv['mean_abs']} st, p95 {pv['p95_abs']} st")
    if lyr:
        c = lyr["check"]
        print(f"new-word timing    vowel starts vs original: median {c['median_abs_ms']} ms, "
              f"p90 {c['p90_abs_ms']} ms, max {c['max_abs_ms']} ms")
    print(f"source unchanged   {rep['source_unchanged']}")
    print(f"wrote {p.name}")
    print("\nNOT VERIFIED until auditioned: naturalness, voice-only status, seams.")
    return {"output": out.name, "report": p.name, "seed": args.seed,
            "output_sha256": sha256(out), "samples": len(wave), "cleanup": cleanup,
            "lip_sync": {k: v for k, v in rep["lip_sync"].items() if k != "matches"},
            "audio_timing": rep["audio_timing"],
            "lyrics": {k: lyr[k] for k in ("orig_text", "new_text", "phonemes", "viseme_plan")}
                      if lyr else None}


def cmd_run(args):
    """Generate a reproducible batch in a new timestamped directory."""
    if args.input:
        args.source = args.input
    if not Path(args.source).is_file():
        sys.exit(f"input not found: {args.source}")
    batch_seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(batch_seed)
    seeds = [batch_seed]
    while len(seeds) < args.count:
        candidate = int(rng.integers(0, 2**32))
        if candidate not in seeds:
            seeds.append(candidate)
    args.step_st, args.min_note, args.low_frac, args.high_frac = 0.75, 0.075, 0.28, 0.72
    src, digest, audio, f0, notes, bands, key = _analysis(args)
    s0 = int(round(args.start * SR))
    s1 = len(audio) if args.dur <= 0 else min(len(audio), s0 + int(round(args.dur * SR)))
    r0 = int(round(args.ref_start * SR))
    r1 = len(audio) if args.ref_dur <= 0 else min(len(audio), r0 + int(round(args.ref_dur * SR)))
    if s1 <= s0 or r1 <= r0:
        raise ValueError("Selected source or reference segment is empty.")
    if (s1 - s0 + HOP - 1) // HOP + (r1 - r0) // HOP > SR // HOP * 30:
        raise ValueError("Source plus reference exceeds the model's 30 s context. "
                         "Select a shorter --dur or --ref-dur (e.g. --dur 20 --ref-dur 8).")
    source_phones = recognize_phonemes(audio[s0:s1], SR) if args.viseme_check else None
    if args.lyrics and not args.orig_lyrics:
        args.orig_lyrics = transcribe(audio[s0:s1], _device())
    bundle = load_seedvc(_device())
    created = datetime.now().astimezone()
    parent = Path(args.output_dir).resolve() if args.output_dir else src.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Atomic directory creation also handles simultaneous runs and clock collisions.
    base = parent / created.strftime("%Y-%m-%d_%H-%M-%S_%f")
    folder, suffix = base, 1
    while True:
        try:
            folder.mkdir()
            break
        except FileExistsError:
            suffix += 1
            folder = parent / f"{base.name}_{suffix}"
    source_copy = folder / f"original{src.suffix}"
    shutil.copy2(src, source_copy)
    if sha256(source_copy) != digest:
        raise RuntimeError("Source changed while copying; retry with an unchanged source.")
    packages = {}
    for package in ("numpy", "scipy", "librosa", "torch", "torchaudio", "transformers",
                    "huggingface-hub", "soundfile", "kokoro"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    settings = {k: v for k, v in vars(args).items() if k != "func"}
    metadata = {
        "schema_version": 1, "created_at": created.isoformat(), "status": "running",
        "source": str(src), "source_copy": source_copy.name, "source_sha256": digest,
        "seed": batch_seed, "variant_seeds": seeds, "requested_count": args.count,
        "completed_count": 0, "sample_rate": SR, "source_samples": len(audio),
        "source_duration_s": len(audio) / SR, "settings": settings,
        "key": key, "bands": bands, "device": str(_device()),
        "python": sys.version, "packages": packages, "script_sha256": sha256(Path(__file__)),
        "viseme_code_sha256": sha256(HERE / "visemes.py"),
        "model": "Plachta/Seed-VC / whisper-base-f0-44k / BigVGAN",
        "voice": "original singer; randomized melody and synthesis",
        "outputs": [],
    }
    def save_metadata():
        if not folder.is_dir():
            raise FileNotFoundError(f"Output folder disappeared during the run: {folder}")
        tmp = folder / "metadata.json.tmp"
        tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tmp.replace(folder / "metadata.json")

    save_metadata()
    noun = "baseline renders" if args.baseline else "mangled melodies"
    print(f"Generating {args.count} {noun} in {folder}")
    print(f"Batch seed: {batch_seed}")
    args.output_dir = str(folder)
    # Render from the archived bytes, so every output has the same source.
    args.source = str(source_copy)
    args.export_reference_excerpt = False
    try:
        if source_phones is not None:
            write_exclusive(folder / "source_phonemes.json",
                            json.dumps(source_phones, indent=2).encode())
            metadata["source_phonemes"] = "source_phonemes.json"
        metadata["source_lyrics"] = args.orig_lyrics
        save_metadata()
        for index, seed in enumerate(seeds, 1):
            args.seed = seed
            args.plan = None
            if not args.baseline:
                rng = np.random.default_rng(seed)
                pattern = tuple(int(x) for x in rng.choice([-2, -1, 1, 2, 3], size=32))
                info = plan_melody(notes, bands, key, "pattern", 5.0, 1.0, 0.09,
                                   pattern, args.extremes, seed)
                payload = {"source": str(src), "sha256": digest, "key": key,
                           "bands": bands, "plan": info,
                           "params": {"mode": "pattern", "seed": seed,
                                      "pattern": list(pattern), "max_leap": 5.0,
                                      "min_change": 1.0, "protect_short": 0.09,
                                      "extremes": args.extremes},
                           "notes": [asdict(n) for n in notes]}
                args.plan = str(write_exclusive(folder / f"melody_{index:02d}_seed{seed}.json",
                                               json.dumps(payload, indent=2).encode()))
            print(f"\n[{index}/{args.count}] seed {seed}")
            result = cmd_render(args, bundle=bundle, source_audio=audio,
                                source_phones=source_phones)
            result.update({"index": index,
                           "plan": Path(args.plan).name if args.plan else None})
            metadata["outputs"].append(result)
            metadata["completed_count"] = len(metadata["outputs"])
            save_metadata()
            if args.lyrics and not args.baseline and args.lyrics_only:
                # Same seed gives the same new words, sung on the original melody.
                print(f"\n[{index}/{args.count}] seed {seed}: new words, original melody")
                plan_path = args.plan
                args.baseline, args.plan = True, None
                args.baseline_label = f"lyricsonly_{index:02d}"
                try:
                    extra = cmd_render(args, bundle=bundle, source_audio=audio,
                                       source_phones=source_phones)
                finally:
                    args.baseline, args.plan, args.baseline_label = False, plan_path, None
                result["lyrics_only"] = extra
                save_metadata()
            if args.lyrics and not args.baseline and args.melody_only:
                # Reuse the altered melody, with content from the original vocal.
                print(f"\n[{index}/{args.count}] seed {seed}: original words, new melody")
                melody_args = argparse.Namespace(**vars(args))
                melody_args.lyrics = None
                melody_args.melody_only_label = f"melodyonly_{index:02d}"
                extra = cmd_render(melody_args, bundle=bundle, source_audio=audio,
                                   source_phones=source_phones)
                extra["plan"] = result["plan"]
                result["melody_only"] = extra
                save_metadata()
        metadata["source_unchanged"] = sha256(src) == digest
        metadata["status"] = "complete"
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        save_metadata()
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if folder.is_dir():
            save_metadata()
        raise
    print(f"\nGenerated {len(metadata['outputs'])} {noun} in {folder}")


def cmd_measure(args):
    device = _device()
    orig = _load(Path(args.source))
    out = _load(Path(args.rendered))
    f0 = extract_f0(orig, device)
    rep = timing_report(orig, out, f0, device)
    if args.viseme_check:
        source_phones, rendered_phones = recognize_phonemes(orig, SR), recognize_phonemes(out, SR)
        rep["lip_sync"] = compare_visemes(source_phones, rendered_phones)
        rep["phoneme_recognition"] = {"source": source_phones, "rendered": rendered_phones}
    rep.update({"source": args.source, "rendered": args.rendered})
    print(json.dumps({k: v for k, v in rep.items() if k != "local_timing_lag_ms"},
                     indent=2))
    t = rep["local_timing_lag_ms"]
    print(f"local lag |ms|: mean {t['mean_abs']} median {t['median_abs']} "
          f"max {t['max_abs']}")
    if args.json:
        print(f"wrote {write_exclusive(Path(args.json), json.dumps(rep, indent=2).encode())}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--source", default=str(HERE / "Dont_say_No.wav"))
        p.add_argument("--step-st", type=float, default=0.75, dest="step_st",
                       help="semitone change that starts a new note")
        p.add_argument("--min-note", type=float, default=0.075, dest="min_note")
        p.add_argument("--low-frac", type=float, default=0.28, dest="low_frac")
        p.add_argument("--high-frac", type=float, default=0.72, dest="high_frac")

    a = sub.add_parser("analyze")
    common(a)
    a.add_argument("--json")
    a.set_defaults(func=cmd_analyze)

    p = sub.add_parser("plan")
    common(p)
    p.add_argument("--mode", choices=["invert", "pattern"], default="invert")
    p.add_argument("--max-leap", type=float, default=5.0, dest="max_leap")
    p.add_argument("--min-change", type=float, default=1.0, dest="min_change")
    p.add_argument("--protect-short", type=float, default=0.09, dest="protect_short")
    p.add_argument("--pattern", default="2,-1,3,-2")
    p.add_argument("--out", default=str(HERE / "melody_plan.json"))
    p.add_argument("--extremes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None, help="random when omitted")
    p.set_defaults(func=cmd_plan)

    def render_opts(r):
        r.add_argument("--source", default=str(HERE / "Dont_say_No.wav"))
        r.add_argument("--baseline", action="store_true",
                       help="re-sing the original melody unchanged (timing check)")
        r.add_argument("--start", type=float, default=0.0)
        r.add_argument("--dur", type=float, default=-1.0)
        r.add_argument("--ref-start", type=float, default=0.0, dest="ref_start")
        r.add_argument("--ref-dur", type=float, default=8.0, dest="ref_dur")
        r.add_argument("--steps", type=int, default=50)
        r.add_argument("--cfg-rate", type=float, default=0.7, dest="cfg_rate")
        r.add_argument("--glide", type=float, default=0.055)
        r.add_argument("--seed", type=int, default=None,
                       help="random when omitted; set an integer to repeat a batch")
        r.add_argument("--output-dir", help="output directory (run creates a dated subfolder)")
        r.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                       help="disable source-guided silence attenuation and peak protection")
        r.add_argument("--keep-raw", action="store_true", help="also save unprocessed model audio")
        r.add_argument("--no-viseme-check", dest="viseme_check", action="store_false",
                       help="skip phoneme recognition; filename score becomes syncNA")
        lyrics = r.add_mutually_exclusive_group()
        lyrics.add_argument("--keep-lyrics", dest="lyrics", action="store_const", const=None,
                            help="preserve original words instead of generating gibberish")
        lyrics.add_argument("--lyrics", help='new words, e.g. "please stay here ..." '
                       '(hyphenate to control syllables: ev-er-more), or "auto" for '
                       'viseme-constrained gibberish (varies with --seed; default for run)')
        r.add_argument("--orig-lyrics", dest="orig_lyrics",
                       help="words actually sung in the input (default: transcribe)")
        r.add_argument("--tts-voice", dest="tts_voice", default="af_heart")
        r.add_argument("--tts-speed", dest="tts_speed", type=float, default=0.85)
        r.add_argument("--extremes", type=int, default=1,
                       help="scale steps the low/high notes may move, staying in band (0 = keep)")
        r.add_argument("--no-envelope", dest="envelope", action="store_false")
        r.add_argument("--envelope-db", type=float, default=8.0, dest="envelope_db")
        r.add_argument("--no-excerpt", dest="export_reference_excerpt",
                       action="store_false")

    q = sub.add_parser("run", help="plan + render in one go (default command)")
    q.add_argument("input", nargs="?", help="sung vocal to recompose (wav/flac/mp3...)")
    q.add_argument("--count", type=int, default=5, help="number of variations (default: 5)")
    q.add_argument("--no-lyrics-only", dest="lyrics_only", action="store_false",
                   help="skip the extra file per variation with the new words on the "
                        "original melody")
    q.add_argument("--no-melody-only", dest="melody_only", action="store_false",
                   help="skip the extra file per variation with the original words on the "
                        "new melody")
    render_opts(q)
    q.set_defaults(func=cmd_run, lyrics="auto")

    r = sub.add_parser("render")
    render_opts(r)
    r.add_argument("--plan", default=str(HERE / "melody_plan.json"))
    r.set_defaults(func=cmd_render)

    m = sub.add_parser("measure")
    m.add_argument("--source", default=str(HERE / "Dont_say_No.wav"))
    m.add_argument("rendered")
    m.add_argument("--json")
    m.add_argument("--no-viseme-check", dest="viseme_check", action="store_false")
    m.set_defaults(func=cmd_measure)

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in sub.choices and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv
    args = ap.parse_args(argv)
    if hasattr(args, "seed") and args.seed is not None and not 0 <= args.seed < 2**32:
        ap.error("--seed must be between 0 and 4294967295")
    if getattr(args, "count", 1) < 1:
        ap.error("--count must be positive")
    if hasattr(args, "steps"):
        if args.steps < 1:
            ap.error("--steps must be positive")
        for name in ("start", "ref_start", "glide", "envelope_db"):
            value = getattr(args, name)
            if not np.isfinite(value) or value < 0:
                ap.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
        for name in ("dur", "ref_dur", "cfg_rate", "tts_speed"):
            if not np.isfinite(getattr(args, name)):
                ap.error(f"--{name.replace('_', '-')} must be finite")
    if args.cmd == "plan" and args.seed is None:
        args.seed = secrets.randbits(32)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
