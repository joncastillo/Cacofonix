# Cacofonix

A tool that mangles an audio track as though it were sung by a truly terrible bard, while preserving the timing and phrasing so the bard's lips still appear to match the original song.

## How to run

You'll need Python 3.10-3.12 and an audio file. A recording of just the vocals works
best. Run these commands in PowerShell to install the dependencies, then try it
with your own file in place of `input.wav`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
git clone https://github.com/Plachtaa/seed-vc.git .seedvc_runtime/seed-vc

# Make five versions of the first 15 seconds
python cacofonix.py "input.wav" --start 0 --dur 15

# Keep the original words and change only the melody
python cacofonix.py "input.wav" --start 0 --dur 15 --keep-lyrics
```

The first time you run Cacofonix, it downloads the models it needs.
Keep the clips short: the model can handle about 30 seconds at a time, including
the voice sample it uses to match the singer. That sample takes up 8 seconds by
default, leaving about 22 seconds for the part you want to change.

Cacofonix saves the results in a new folder next to your audio file, named with the
date and time. You'll find five WAV files, a copy of your original recording, and
JSON reports with the settings and timing scores. Add `--count 1` if you only want
one version, or `--seed 12345` to use the same melody choices when you run it again.

## Architecture

```text
                     Your recording
                            |
              +-------------+-------------+
              |                           |
      Find the original notes     Find the words and their timing
              |                           |
      Change the melody,          Make gibberish with similar
      keep the timing             mouth shapes
              |                           |
              |                   Have Kokoro speak the words,
              |                   then match the original timing
              |                           |
              +-------------+-------------+
                            |
                 Sing with Seed-VC <--- Sample of the singer's voice
                            |
                 Reduce noise and adjust volume
                            |
                 Check timing and mouth shapes
                            |
                 Save WAV files and JSON reports
```

Cacofonix changes the tune and replaces the words with gibberish that should look
similar on the singer's lips. Kokoro speaks the new words, and Cacofonix adjusts
their timing to match the original before Seed-VC sings them in the original
singer's voice. If you use `--keep-lyrics`, it keeps the words and skips the Kokoro
step. It then reduces noise and checks how closely the sounds and timing match.
The lip-sync score comes from the audio alone, so you'll still need to watch the
video to judge how well the lips match.
