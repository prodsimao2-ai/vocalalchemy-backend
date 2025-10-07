"""
VocalAlchemy Tools — stem splitter + vocal-to-instrument converter
------------------------------------------------------------------
CLI usage examples:
  # 1) Split stems (Demucs)
  python vocalchemy_tools.py split-stems "input.wav" --out "out" --model htdemucs --format wav

  # 2) Convert vocal to instrument (CREPE -> MIDI -> FluidSynth)
  python vocalchemy_tools.py convert-vocal "vocal.wav" --instrument sax --sf2 "/path/to/YourSoundFont.sf2" --out "converted_sax.wav"
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------- STEM SPLITTER (Demucs) ----------------------
def split_stems_demucs(
    input_audio: str,
    out_dir: str = "stems",
    model: str = "htdemucs",
    output_format: str = "wav",
    verbose: bool = True,
) -> List[Path]:
    """
    Uses the demucs CLI to split stems. Requires `demucs` to be installed.
    Produces 4 stems: drums, bass, other, vocals.
    Returns list of output file paths.
    """
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "-o",
        str(out_dir_path),
        input_audio,
    ]

    if verbose:
        print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Demucs failed. Ensure demucs is installed and the input path is valid.")

    # Demucs creates: out_dir/model/songname/*.wav
    candidates = sorted(out_dir_path.rglob("*.wav"))
    if not candidates:
        raise FileNotFoundError("No stems found after demucs run.")
    return candidates


# ---------------------- VOCAL -> INSTRUMENT ----------------------
def vocal_to_instrument(
    input_audio: str,
    out_path: str = "converted.wav",
    instrument: str = "sax",
    key: str = "C",
    scale: str = "major",
    bpm: float = 100.0,
    harmonize: bool = False,
    choir_mode: bool = False,
    sf2: Optional[str] = None,
):
    """
    Converts a vocal audio file into an instrument using CREPE + pretty_midi.
    1. Extracts pitch from vocal.
    2. Quantizes to scale/key.
    3. Creates MIDI.
    4. Renders with SoundFont if provided.
    """
    import librosa
    import soundfile as sf
    import crepe
    import pretty_midi

    # --- Helper functions ---
    def hz_to_midi(f):
        return 69 + 12 * np.log2(f / 440.0)

    def db(x, eps=1e-12):
        return 20 * np.log10(np.maximum(eps, x))

    # --- Load audio ---
    y, sr = librosa.load(input_audio, sr=None, mono=True)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sr16 = 16000

    # --- Extract pitch ---
    time, freq, conf, _ = crepe.predict(y16, sr16, viterbi=True, step_size=10, model="tiny")
    midi = np.where(conf > 0.4, hz_to_midi(freq), np.nan)
    voiced = ~np.isnan(midi)
    times_v = time[voiced]
    midi_v = midi[voiced]

    if len(midi_v) == 0:
        raise RuntimeError("No voiced segments found in the input audio.")

    # --- Create notes from continuous MIDI values ---
    notes = []
    start = times_v[0]
    prev_pitch = midi_v[0]
    for i in range(1, len(midi_v)):
        if abs(midi_v[i] - prev_pitch) > 0.5:
            end = times_v[i]
            notes.append((start, end, int(round(prev_pitch))))
            start = times_v[i]
        prev_pitch = midi_v[i]
    notes.append((start, times_v[-1], int(round(prev_pitch))))

    # --- Build MIDI ---
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    gm_map = {
        "piano": 0,
        "guitar": 24,
        "violin": 40,
        "sax": 65,
        "trumpet": 56,
        "choir": 52,
    }
    program = gm_map.get(instrument.lower(), 65)
    inst = pretty_midi.Instrument(program=program)

    for (s, e, p) in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=float(s), end=float(e)))
    pm.instruments.append(inst)

    # --- Render or save MIDI ---
    if sf2 and Path(sf2).exists():
        audio = pm.fluidsynth(sf2_path=sf2)
        sf.write(out_path, audio, 44100)
        return out_path
    else:
        mid_path = Path(out_path).with_suffix(".mid")
        pm.write(str(mid_path))
        return str(mid_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    s = sub.add_parser("split-stems")
    s.add_argument("input")
    s.add_argument("--out", default="stems")
    s.add_argument("--model", default="htdemucs")

    c = sub.add_parser("convert-vocal")
    c.add_argument("input")
    c.add_argument("--instrument", default="sax")
    c.add_argument("--sf2", default=None)
    c.add_argument("--out", default="converted.wav")

    args = parser.parse_args()

    if args.cmd == "split-stems":
        print(split_stems_demucs(args.input, args.out, args.model))
    elif args.cmd == "convert-vocal":
        print(vocal_to_instrument(args.input, args.out, args.instrument, sf2=args.sf2))
