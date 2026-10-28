import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import soundfile as sf

import cacofonix as c
import visemes as v


class SetupChecks(unittest.TestCase):
    def test_missing_seedvc_explains_setup_for_both_loaders(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(c, "SEEDVC", Path(folder) / "seed-vc"), \
                patch.object(c, "_RMVPE", None):
            for loader in (c.load_rmvpe, c.load_seedvc):
                with self.subTest(loader=loader.__name__):
                    with self.assertRaises(SystemExit) as error:
                        loader("cpu")
                    self.assertIn("git clone", str(error.exception))
                    self.assertIn(str(c.SEEDVC), str(error.exception))


class AudioChecks(unittest.TestCase):
    def test_cleanup_reduces_silence_noise_preserves_active_sound_and_length(self):
        rng = np.random.default_rng(11)
        source = np.zeros(c.SR * 3, dtype=np.float32)
        # A noisy unvoiced sound must survive just like a pitched vowel.
        source[c.SR:2*c.SR] = rng.normal(0, 0.08, c.SR)
        generated = source + rng.normal(0, 0.005, len(source))
        cleaned, info = c.clean_output(generated, source)
        quiet = slice(0, c.SR // 2)
        active = slice(c.SR + 4000, 2*c.SR - 4000)
        self.assertEqual(len(source), len(cleaned))
        self.assertLess(np.std(cleaned[quiet]), np.std(generated[quiet]) * 0.02)
        np.testing.assert_allclose(cleaned[active], generated[active], atol=1e-7)
        self.assertGreater(info["attenuated_sample_fraction"], 0.3)

    def test_cleanup_peak_protection(self):
        cleaned, _ = c.clean_output(np.full(5000, 2.0), np.full(5000, 0.2))
        self.assertLessEqual(float(np.abs(cleaned).max()), 0.980001)

    def test_score_detects_shift_and_handles_missing_evidence(self):
        rng = np.random.default_rng(42)
        original = rng.random(600)
        same = c.local_lags(original, original)
        delayed = c.local_lags(original, np.pad(original[:-8], (8, 0)))

        def score(windows):
            return c.lip_sync_score({
                "duration_s": {"original": 7, "rendered": 7},
                "voicing": {"frame_agreement": 1.0},
                "local_timing_lag_ms": {"windows": windows},
            })

        self.assertEqual(score(same)["score"], 100)
        self.assertLess(score(delayed)["score"], 70)
        self.assertEqual(score(delayed)["status"], "review_needed")
        self.assertIsNone(score([])["score"])
        self.assertIsNone(score(c.local_lags(original, np.zeros(600)))["score"])
        self.assertFalse(score(same)["visual_lip_sync_verified"])

    def test_silent_timing_report_is_not_perfect_or_nan(self):
        with patch.object(c, "envelope", return_value=np.zeros(400)), \
                patch.object(c, "extract_f0", return_value=np.zeros(400)):
            report = c.timing_report(np.zeros(c.SR * 4), np.zeros(c.SR * 4),
                                     np.zeros(400), "cpu")
        self.assertIsNone(report["audio_timing"]["score"])
        self.assertIsNone(report["local_timing_lag_ms"]["mean_abs"])
        json.dumps(report, allow_nan=False)


class BatchChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "sung vocal.wav"
        t = np.arange(c.SR * 4) / c.SR
        self.audio = (0.2 * np.sin(2*np.pi*220*t)).astype(np.float32)
        sf.write(self.source, self.audio, c.SR)
        self.original_bytes = self.source.read_bytes()
        self.out = self.root / "outputs"
        self.out.mkdir()

    def analysis(self, args):
        notes = [c.Note(i, i*40, (i+1)*40, p) for i, p in enumerate(
            (60, 62, 64, 65, 67, 69, 67, 65, 64, 60))]
        bands = c.classify_bands(notes, 0.28, 0.72)
        return (self.source, c.sha256(self.source), self.audio, np.full(400, 220),
                notes, bands, c.detect_key(notes))

    def run_batch(self, *extra, failure=False):
        before = set(self.out.iterdir())
        replacement_content = object()
        fake_audio = lambda *a, **kw: (self.audio.copy(), 344, 344)
        onsets = np.random.default_rng(5).random(344)
        capture = io.StringIO()
        phones = {"duration_s": 4.0, "events": [
            {"phoneme": p, "viseme": v.viseme_of(p), "start_s": i * 0.2,
             "end_s": i * 0.2 + 0.05, "confidence": 0.95}
            for i, p in enumerate(["b", "æ", "t"] * 3)]}

        def fake_lyrics(piece, f0, notes, old, new, *args, **kwargs):
            return replacement_content, f0, {"tts_seconds": 2.0, "syllables": [
                {"orig": "bat", "new": "pat", "orig_vowel": 0.2}]}

        with contextlib.redirect_stdout(capture), \
                patch.object(c, "_analysis", side_effect=self.analysis), \
                patch.object(c, "load_seedvc", return_value=(None, None)) as loader, \
                patch.object(c, "extract_f0", return_value=np.full(400, 220.0)), \
                patch.object(c, "envelope", return_value=onsets), \
                patch.object(c, "recognize_phonemes", return_value=phones), \
                patch.object(c, "transcribe", return_value="bat bat bat"), \
                patch.object(c, "kokoro_pipe", return_value=SimpleNamespace(
                    g2p=lambda word: ("bæt", None))), \
                patch.object(c, "lyrics_conditioning", side_effect=fake_lyrics), \
                patch.object(c, "lyric_timing_check", return_value={
                    "median_abs_ms": 0, "p90_abs_ms": 0, "max_abs_ms": 0}), \
                patch.object(c, "match_envelope", side_effect=lambda wave, *a, **kw: wave), \
                patch.object(c, "synthesise", side_effect=(
                    [(self.audio.copy(), 344, 344), RuntimeError("test render failure")]
                    if failure else fake_audio)) as synthesis:
            command = [str(self.source), "--output-dir", str(self.out), *extra]
            if failure:
                with self.assertRaisesRegex(RuntimeError, "test render failure"):
                    c.main(command)
            else:
                c.main(command)
            self.assertEqual(loader.call_count, 1)
            for call in synthesis.call_args_list:
                self.assertIs(call.kwargs["content"],
                              None if "--keep-lyrics" in extra else replacement_content)
        folder, = set(self.out.iterdir()) - before
        return folder, json.loads((folder / "metadata.json").read_text()), capture.getvalue()

    def test_default_five_outputs_metadata_copy_score_and_random_seed(self):
        with patch.object(c.secrets, "randbits", return_value=123456):
            folder, meta, log = self.run_batch()
        self.assertEqual(meta["seed"], 123456)
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(meta["completed_count"], 5)
        self.assertEqual(len(set(meta["variant_seeds"])), 5)
        self.assertEqual((folder / meta["source_copy"]).read_bytes(), self.original_bytes)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertEqual(len(list(folder.glob("*.wav"))), 11)  # 5 + 5 lyrics-only + copy
        self.assertIn(f"Generated 5 mangled melodies in {folder.resolve()}", log)
        self.assertEqual(meta["settings"]["steps"], 50)
        self.assertEqual(meta["settings"]["lyrics"], "auto")
        self.assertEqual(meta["source_lyrics"], "bat bat bat")
        for output in meta["outputs"]:
            self.assertIn("_sync100.wav", output["output"])
            self.assertEqual(output["lip_sync"]["score"], 100)
            self.assertTrue(output["lyrics"]["viseme_plan"]["planned_visemes_preserved"])
            self.assertGreater(output["lyrics"]["viseme_plan"]["changed_phonemes"], 0)
            self.assertEqual(output["output_sha256"], c.sha256(folder / output["output"]))
            self.assertTrue((folder / output["report"]).is_file())
            samples, sample_rate = sf.read(folder / output["output"])
            self.assertEqual(len(samples), len(self.audio))
            self.assertEqual(sample_rate, c.SR)
            extra = output["lyrics_only"]
            self.assertIn("lyricsonly_", extra["output"])
            self.assertEqual(extra["seed"], output["seed"])
            self.assertEqual(extra["lyrics"]["new_text"], output["lyrics"]["new_text"])
            self.assertTrue((folder / extra["output"]).is_file())

    def test_lyrics_only_can_be_disabled_and_is_skipped_when_words_are_kept(self):
        for option in ("--no-lyrics-only", "--keep-lyrics"):
            with self.subTest(option=option):
                folder, meta, _ = self.run_batch(option, "--count", "1")
                self.assertNotIn("lyrics_only", meta["outputs"][0])
                self.assertEqual(len(list(folder.glob("*.wav"))), 2)

    def test_seed_repeats_plans_and_runs_never_overwrite(self):
        folder1, first, _ = self.run_batch("--seed", "0", "--count", "2")
        folder2, second, _ = self.run_batch("--seed", "0", "--count", "2")
        self.assertNotEqual(folder1, folder2)
        self.assertEqual(first["variant_seeds"], second["variant_seeds"])
        for a, b in zip(first["outputs"], second["outputs"]):
            self.assertEqual((folder1 / a["plan"]).read_bytes(),
                             (folder2 / b["plan"]).read_bytes())

    def test_partial_failure_preserves_metadata_and_first_output(self):
        folder, meta, log = self.run_batch("--seed", "7", failure=True)
        self.assertEqual(meta["status"], "failed")
        self.assertEqual(meta["completed_count"], 1)
        self.assertIn("test render failure", meta["error"])
        self.assertTrue((folder / meta["outputs"][0]["output"]).is_file())
        self.assertNotIn("Generated 5 mangled", log)

    def test_invalid_args_fail_before_model_load(self):
        with patch.object(c, "load_seedvc") as loader, contextlib.redirect_stderr(io.StringIO()):
            for option in (("--count", "0"), ("--seed", "-1"), ("--steps", "0"),
                           ("--start", "-1"), ("--dur", "nan")):
                with self.assertRaises(SystemExit):
                    c.main([str(self.source), *option])
            loader.assert_not_called()

    def test_preserve_words_and_disabled_check_are_explicit(self):
        folder, meta, _ = self.run_batch("--keep-lyrics", "--no-viseme-check", "--count", "1")
        output = meta["outputs"][0]
        self.assertIsNone(output["lyrics"])
        self.assertIsNone(output["lip_sync"]["score"])
        self.assertIn("_syncNA.wav", output["output"])


class VisemeChecks(unittest.TestCase):
    def test_replacements_change_every_eligible_sound_without_allophone_only_swaps(self):
        phones = {"don't": "dˈOnt", "love": "lˈʌv", "me": "mˈi", "you": "ju"}
        with patch.object(c, "kokoro_pipe", return_value=SimpleNamespace(
                g2p=lambda word: (phones[word], None))):
            for seed in range(20):
                _, output, _, plan = c.gibberize(
                    "don't love me you", np.random.default_rng(seed), return_details=True)
                self.assertTrue(plan["planned_visemes_preserved"])
                self.assertNotIn("ɾ", output)
                self.assertNotIn("ᵻ", output)
                for change in plan["substitutions"]:
                    old, new = change["original"], change["replacement"]
                    if c.replacement_phones(old):
                        self.assertNotEqual(old, new)
                self.assertIn("lˈæf", output)
                self.assertTrue(output.endswith("ju"))  # No safe alternative.

    def test_tts_uses_replacement_phonemes_instead_of_display_spelling(self):
        from unittest.mock import Mock
        pipe = Mock()
        pipe.generate_from_tokens.return_value = [SimpleNamespace(
            audio=SimpleNamespace(numpy=lambda: np.ones(2400, dtype=np.float32)))]
        with patch.object(c, "kokoro_pipe", return_value=pipe):
            wave = c.tts_speak("baf", "af_heart", 0.85, phonemes="bˈæf")
        pipe.generate_from_tokens.assert_called_once_with("bˈæf", voice="af_heart", speed=0.85)
        pipe.assert_not_called()
        self.assertEqual(len(wave), 2400 + 2 * 7200)

    @staticmethod
    def recording(phones, shift=0, confidence=0.95):
        return {"duration_s": 3.0, "events": [
            {"phoneme": p, "viseme": v.viseme_of(p), "start_s": round(i*0.25+shift, 3),
             "end_s": round(i*0.25+shift+0.05, 3), "confidence": confidence}
            for i, p in enumerate(phones)]}

    def test_same_visemes_different_words_score_high(self):
        a = self.recording(["p", "æ", "f", "i", "t", "s"])
        b = self.recording(["b", "ʌ", "v", "ɪ", "d", "z"])
        result = v.compare_visemes(a, b)
        self.assertEqual(result["score"], 100)
        self.assertFalse(result["visual_lip_sync_verified"])

    def test_drift_and_wrong_mouth_shapes_are_penalized(self):
        phones = ["p", "æ", "f", "i", "t", "s"]
        a = self.recording(phones)
        self.assertLess(v.compare_visemes(a, self.recording(phones, shift=0.09))["score"], 50)
        self.assertEqual(v.compare_visemes(a, self.recording(["w"] * 6))["score"], 0)
        self.assertLess(v.compare_visemes(a, self.recording(phones[:-2]))["score"], 100)

    def test_unknown_low_confidence_and_silence_are_inconclusive(self):
        for phones, confidence in ((["p", "i", "t"], 0.2), (["unknown"]*5, 1.0), ([], 1.0)):
            with self.subTest(phones=phones, confidence=confidence):
                a = self.recording(phones, confidence=confidence)
                self.assertIsNone(v.compare_visemes(a, a)["score"])

    def test_ctc_blanks_are_not_silence_and_repeats_are_retained(self):
        events = v.ctc_events([0, 1, 1, 0, 1, 2, 0], [0.95]*7,
                              {0: "<pad>", 1: "p", 2: "aɪ"}, 0, 0.02)
        self.assertEqual([e["phoneme"] for e in events], ["p", "p", "aɪ"])
        self.assertEqual(events[0]["start_s"], 0.02)
        self.assertEqual(events[0]["end_s"], 0.06)
        self.assertEqual(events[-1]["viseme"], "11")
        self.assertEqual(v.viseme_of("t͡ʃ"), "16")
        self.assertEqual(v.viseme_of("iː"), "6")

    def test_gibberish_preserves_visemes_and_diphthongs(self):
        with patch.object(c, "kokoro_pipe", return_value=SimpleNamespace(
                g2p=lambda word: ("plˈA", None))):
            a = c.gibberize("play", np.random.default_rng(9), return_details=True)
            b = c.gibberize("play", np.random.default_rng(9), return_details=True)
        self.assertEqual(a, b)
        self.assertTrue(a[3]["planned_visemes_preserved"])
        self.assertIn("lˈA", a[1])
        self.assertNotEqual(a[1][0], "p")
        with self.assertRaisesRegex(ValueError, "Unsupported phoneme"):
            c._phone_units("pʘ")


if __name__ == "__main__":
    unittest.main()
