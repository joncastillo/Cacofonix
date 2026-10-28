"""Audio-derived English viseme checks; no transcript or video is required.

Recognize phonemes independently in both recordings, then compare mouth-shape
events in sequence and time. CTC emission spans are timing evidence, not precise
articulator boundaries. Speech-trained recognition is less reliable on singing.
"""
import json
import re

import numpy as np

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
MAPPING_SOURCE = (
    "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/"
    "how-to-speech-synthesis-viseme")

# English mouth-shape classes from Microsoft's published IPA table. Extensions
# for common English allophones and diphthongs are explicit below. Diphthongs
# remain separate trajectories, rather than being reduced to their first vowel.
VISEME_GROUPS = {
    "1": ("æ", "ə", "ʌ", "ɐ"), "2": ("ɑ", "a"), "3": ("ɔ",),
    "4": ("ɛ", "ʊ"), "5": ("ɝ", "ɜ", "ɚ"), "6": ("j", "i", "ɪ", "ᵻ"),
    "7": ("w", "u"), "8": ("o",), "9": ("aʊ",), "10": ("ɔɪ",),
    "11": ("aɪ",), "12": ("h",), "13": ("ɹ", "r"), "14": ("l", "ɫ"),
    "15": ("s", "z"), "16": ("ʃ", "tʃ", "dʒ", "ʒ", "ʧ", "ʤ"),
    "17": ("ð",), "18": ("f", "v"), "19": ("d", "t", "n", "θ", "ɾ"),
    "20": ("k", "g", "ɡ", "ŋ"), "21": ("p", "b", "m"),
    "DIPH_EI": ("eɪ",), "DIPH_OU": ("oʊ", "əʊ"),
    "DIPH_IA": ("ɪə",), "DIPH_EA": ("ɛə", "eə"), "DIPH_UA": ("ʊə",),
}
PHONE_TO_VISEME = {phone: group for group, phones in VISEME_GROUPS.items() for phone in phones}
_RECOGNIZER = None


def viseme_of(phone):
    # Strip stress, length, and the IPA affricate tie bar; retain other diacritics
    # as unknown instead of silently claiming an unsupported mouth shape.
    phone = re.sub("[ˈˌːˑ͡]", "", phone)
    return PHONE_TO_VISEME.get(phone)


def ctc_events(ids, confidence, vocabulary, blank_id, frame_s, offset_s=0.0):
    """Collapse repeated CTC labels, retaining repeats separated by blanks.

    Blank means 'no label emitted', NOT silence. Never score blank agreement.
    Unknown phonemes stay in the event list so coverage reflects them.
    """
    events, i = [], 0
    while i < len(ids):
        j = i + 1
        while j < len(ids) and ids[j] == ids[i]:
            j += 1
        token = vocabulary.get(int(ids[i]), "<unk>")
        if int(ids[i]) != blank_id and token not in ("<s>", "</s>", "|", " "):
            events.append({
                "phoneme": token, "viseme": viseme_of(token),
                "start_s": round(i * frame_s + offset_s, 4),
                "end_s": round(j * frame_s + offset_s, 4),
                "confidence": round(float(np.mean(confidence[i:j])), 4),
            })
        i = j
    return events


def recognize_phonemes(audio, sample_rate=44100):
    """Recognize audible phones without forcing the planned words on the audio.

    CPU inference keeps the verification model out of the synthesis GPU budget.
    Loading vocabulary directly avoids requiring an eSpeak text tokenizer just
    to decode model output. All files use the application's Hugging Face cache.
    """
    global _RECOGNIZER
    import torch
    from scipy.signal import resample_poly
    from math import gcd
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    if _RECOGNIZER is None:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).eval().cpu()
        with open(hf_hub_download(MODEL_ID, "vocab.json"), encoding="utf-8") as f:
            vocabulary = {int(v): k for k, v in json.load(f).items()}
        _RECOGNIZER = extractor, model, vocabulary
    extractor, model, vocabulary = _RECOGNIZER
    divisor = gcd(sample_rate, 16000)
    audio16 = resample_poly(np.asarray(audio, dtype=np.float32),
                            16000 // divisor, sample_rate // divisor)
    if len(audio16) < 400 or not np.isfinite(audio16).all():
        raise ValueError("Viseme recognition needs finite audio of at least 25 ms.")
    inputs = extractor(audio16, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = logits.softmax(dim=-1)
        confidence, ids = probabilities.max(dim=-1)
    stride, receptive = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    events = ctc_events(ids.numpy(), confidence.numpy(), vocabulary,
                        model.config.pad_token_id, stride / 16000,
                        max(0, (receptive - stride) / 2) / 16000)
    return {"model": MODEL_ID, "model_revision": getattr(model.config, "_commit_hash", None),
            "frame_s": stride / 16000, "duration_s": len(audio) / sample_rate,
            "events": events}


def compare_visemes(original, rendered):
    """Sequence alignment penalizes missing/extra phones and large timing drift.

    No time warping or global-lag correction is applied to reported errors.
    Low-confidence/unknown events reduce coverage, never count as matches.
    """
    threshold = 0.5
    raw_a, raw_b = original["events"], rendered["events"]
    a = [e for e in raw_a if e["viseme"] is not None and e["confidence"] >= threshold]
    b = [e for e in raw_b if e["viseme"] is not None and e["confidence"] >= threshold]
    coverage_a = len(a) / len(raw_a) if raw_a else 0
    coverage_b = len(b) / len(raw_b) if raw_b else 0
    result = {
        "score": None, "status": "inconclusive", "kind": "recognized phoneme/viseme timing",
        "visual_lip_sync_verified": False, "mapping_version": "english-v1",
        "mapping_source": MAPPING_SOURCE, "recognizer": MODEL_ID,
        "source_event_count": len(raw_a), "rendered_event_count": len(raw_b),
        "source_coverage": round(coverage_a, 4), "rendered_coverage": round(coverage_b, 4),
        "confidence_threshold": threshold,
        "limitation": "Uncalibrated audio-derived estimate. Speech recognition may err "
                      "on singing; CTC emission spans are not exact phoneme boundaries. "
                      "No video or coarticulation is assessed.",
        "scoring": "v2: 100 * matched-viseme fraction * min(source, rendered coverage) * "
                   "max(0, 1 - p90 emission-boundary error / 150 ms) * "
                   "max(0, 1 - duration difference / 100 ms). At least 3 reliable "
                   "events on each side and 50% coverage required.",
    }
    if min(len(a), len(b)) < 3 or min(coverage_a, coverage_b) < 0.5:
        return result
    n, m = len(a), len(b)
    cost = np.zeros((n+1, m+1))
    cost[:, 0], cost[0, :] = np.arange(n+1), np.arange(m+1)
    move = np.zeros((n+1, m+1), dtype=np.uint8)
    for i in range(1, n+1):
        for j in range(1, m+1):
            same = a[i-1]["viseme"] == b[j-1]["viseme"]
            delta = abs(a[i-1]["start_s"] - b[j-1]["start_s"])
            substitution = min(1.5, delta / 0.15) if same else 2.0
            options = (cost[i-1, j-1] + substitution, cost[i-1, j] + 1, cost[i, j-1] + 1)
            move[i, j] = int(np.argmin(options))
            cost[i, j] = options[move[i, j]]
    pairs, i, j = [], n, m
    while i or j:
        direction = int(move[i, j]) if i and j else (1 if i else 2)
        if direction == 0:
            x, y = a[i-1], b[j-1]
            if x["viseme"] == y["viseme"]:
                pairs.append({"source": x, "rendered": y,
                              "start_delta_ms": round((y["start_s"]-x["start_s"])*1000, 2),
                              "end_delta_ms": round((y["end_s"]-x["end_s"])*1000, 2)})
            i, j = i-1, j-1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    match_fraction = len(pairs) / max(n, m)
    errors = [abs(p[k]) for p in pairs for k in ("start_delta_ms", "end_delta_ms")]
    p90 = float(np.percentile(errors, 90)) if errors else None
    duration_delta = abs(original["duration_s"] - rendered["duration_s"]) * 1000
    score = (100 * match_fraction * min(coverage_a, coverage_b) *
             max(0, 1-p90/150) * max(0, 1-duration_delta/100)) if p90 is not None else 0
    result.update({"score": int(round(score)), "matched_viseme_fraction": round(match_fraction, 4),
                   "matched_events": len(pairs), "p90_boundary_error_ms": p90,
                   "duration_difference_ms": round(duration_delta, 3), "matches": pairs,
                   "status": "viseme_timing_consistent" if score >= 80 else "review_needed"})
    return result
