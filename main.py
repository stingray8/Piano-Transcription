import os
import sys
import ssl
import urllib.request
from pathlib import Path
import yt_dlp
import pretty_midi
import music21
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate


def download_progress(count, block_size, total_size):
    """Callback to print model weights download percentage."""
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r      [*] Downloading model weights: {min(percent, 100)}% complete")
        sys.stdout.flush()


def ensure_checkpoint_exists():
    """Downloads AI model weights via Python if missing or incomplete."""
    checkpoint_dir = Path.home() / "piano_transcription_inference_data"
    checkpoint_path = checkpoint_dir / "note_F1=0.9677_pedal_F1=0.9186.pth"
    
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 1.6e8:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print("      [*] Model checkpoint missing/incomplete. Starting download (~165 MB)...")
        
        zenodo_url = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
        ssl_context = ssl._create_unverified_context()
        
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(zenodo_url, checkpoint_path, reporthook=download_progress)
        print("\n      [✓] Model checkpoint downloaded successfully.")


def download_youtube_audio(youtube_url: str, output_wav_path: str = "temp_audio.wav") -> str:
    """Downloads audio from a YouTube link and converts it to a standard WAV file."""
    print(f"\n[1/4] Downloading audio from: {youtube_url}")
    
    if os.path.exists(output_wav_path):
        os.remove(output_wav_path)

    out_base = os.path.splitext(output_wav_path)[0]

    ydl_opts = {
        'format': 'ba/b',
        'outtmpl': f'{out_base}.%(ext)s',
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    if not os.path.exists(output_wav_path):
        raise FileNotFoundError(f"Audio extraction failed. File not found at: {output_wav_path}")

    return output_wav_path


def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def transcribe_audio_to_midi(audio_path: str, raw_midi_path: str = "raw_output.mid"):
    """Runs high-precision acoustic piano neural network transcription."""
    print("[2/4] Running AI Transcription Model...")
    ensure_checkpoint_exists()

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    device = 'cuda' if torch_cuda_available() else 'cpu'
    print(f"      Processing on compute device: {device.upper()}")

    transcriptor = PianoTranscription(device=device)
    transcriptor.transcribe(audio, raw_midi_path)
    print(f"      Raw transcription generated: {raw_midi_path}")


import numpy as np
from music21 import stream, note as m21note, chord as m21chord, clef, key as m21key
from music21 import meter, tempo as m21tempo, layout


def estimate_bpm(audio_path: str) -> int:
    """Estimate tempo and fold it into a sensible 60-140 BPM range."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    while bpm < 60:
        bpm *= 2
    while bpm > 140:
        bpm /= 2
    return int(round(bpm))

def reduce_chord(pitches, anchor, max_notes=5, max_span=12):
    """
    Trim a chord so one hand can play it.
    anchor='top' keeps the highest note (right hand melody).
    anchor='bottom' keeps the lowest note (left hand bass).
    Inner notes are dropped first, then notes farthest from the anchor.
    """
    p = sorted(set(pitches))
    anchor_note = p[-1] if anchor == "top" else p[0]

    while len(p) > max_notes:
        inner = [x for x in p if x != p[0] and x != p[-1]]
        mid = (p[0] + p[-1]) / 2
        p.remove(min(inner, key=lambda x: abs(x - mid)))

    # 2. Too wide: drop the note farthest from the anchor
    while len(p) > 1 and (p[-1] - p[0]) > max_span:
        p.remove(p[0] if anchor == "top" else p[-1])

    return p

def extract_hand_events(input_midi_path, audio_path, bpm, grid=0.25,
                        min_note_duration=0.08, min_velocity=15, split_pitch=60,
                        max_notes=5, max_span=12):
    """
    Returns (rh_events, lh_events). Each event is (onset_beat, [midi_pitches], duration_beats).
    For every chord, picks the split between hands that keeps both hands playable
    and closest to where each hand just was. Notes are only dropped if no split works.
    """
    max_duration = librosa.get_duration(path=audio_path)
    midi_data = pretty_midi.PrettyMIDI(input_midi_path)
    spb = 60.0 / bpm

    clusters = {}
    for inst in midi_data.instruments:
        for n in inst.notes:
            if (n.end - n.start) < min_note_duration or n.velocity < min_velocity:
                continue
            if n.start >= max_duration:
                continue
            s = round((n.start / spb) / grid) * grid
            e = round((min(n.end, max_duration) / spb) / grid) * grid
            if e <= s:
                e = s + grid
            cl = clusters.setdefault(s, {})
            cl[n.pitch] = max(e, cl.get(n.pitch, e))

    prev = {"lh": 48.0, "rh": 72.0} 
    lh_raw, rh_raw = [], [] 

    for s in sorted(clusters):
        cl = clusters[s]
        pitches = sorted(cl)
        n = len(pitches)
        default_k = sum(1 for p in pitches if p < split_pitch)

        best = None
        # k = how many of the LOWEST notes go to the left hand
        for k in range(n + 1):
            lh = reduce_chord(pitches[:k], "bottom", max_notes, max_span) if k > 0 else []
            rh = reduce_chord(pitches[k:], "top", max_notes, max_span) if k < n else []

            dropped = n - len(lh) - len(rh)  # 0 whenever the split is fully playable
            move = 0.0
            if lh:
                move += abs(sum(lh) / len(lh) - prev["lh"])
            if rh:
                move += abs(sum(rh) / len(rh) - prev["rh"])

            cost = dropped * 1000 + move + 0.01 * abs(k - default_k)
            if best is None or cost < best[0]:
                best = (cost, lh, rh)

        _, lh, rh = best
        if lh:
            prev["lh"] = sum(lh) / len(lh)
            lh_raw.append((s, lh, max(cl[p] for p in lh)))
        if rh:
            prev["rh"] = sum(rh) / len(rh)
            rh_raw.append((s, rh, max(cl[p] for p in rh)))

    def build_events(raw):
        events = []
        for i, (s, pitches, end) in enumerate(raw):
            if i + 1 < len(raw):
                nxt = raw[i + 1][0]
                if nxt - end <= 1.0 or end > nxt:
                    end = nxt
            dur = max(grid, min(end - s, 4.0))
            events.append((s, sorted(pitches), dur))
        return events

    return build_events(rh_raw), build_events(lh_raw)

def save_clean_midi(rh_events, lh_events, bpm, output_midi_path):
    spb = 60.0 / bpm
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for name, events in (("Right Hand", rh_events), ("Left Hand", lh_events)):
        inst = pretty_midi.Instrument(program=0, name=name)
        for onset, pitches, dur in events:
            for p in pitches:
                inst.notes.append(pretty_midi.Note(
                    velocity=80, pitch=p, start=onset * spb, end=(onset + dur) * spb))
        midi.instruments.append(inst)
    midi.write(output_midi_path)
    print(f"      Clean 2-track MIDI saved: {output_midi_path}")


def build_grand_staff_score(rh_events, lh_events, bpm, output_xml_path):
    probe = stream.Stream()
    for _, pitches, _ in rh_events + lh_events:
        for p in pitches:
            probe.append(m21note.Note(p))
    detected = probe.analyze("key")
    sharps = detected.sharps

    def spell(midi_pitch):
        n = m21note.Note(midi_pitch)
        p = n.pitch
        if p.accidental is not None:
            if sharps < 0 and p.accidental.alter == 1:
                p = p.getEnharmonic()
            elif sharps >= 0 and p.accidental.alter == -1:
                p = p.getEnharmonic()
        return p

    def make_part(events, the_clef, first):
        part = stream.PartStaff()
        part.insert(0, the_clef)
        part.insert(0, m21key.KeySignature(sharps))
        part.insert(0, meter.TimeSignature("4/4"))
        if first:
            part.insert(0, m21tempo.MetronomeMark(number=bpm))
        for onset, pitches, dur in events:
            if len(pitches) == 1:
                el = m21note.Note()
                el.pitch = spell(pitches[0])
            else:
                el = m21chord.Chord([spell(p) for p in pitches])
            el.quarterLength = dur
            part.insert(onset, el)
        part.makeNotation(inPlace=True)
        return part

    rh_part = make_part(rh_events, clef.TrebleClef(), first=True)
    lh_part = make_part(lh_events, clef.BassClef(), first=False)

    score = stream.Score()
    score.insert(0, rh_part)
    score.insert(0, lh_part)
    score.insert(0, layout.StaffGroup([rh_part, lh_part], name="Piano",
                                      abbreviation="Pno.", symbol="brace",
                                      barTogether=True))
    score.write("musicxml", fp=output_xml_path)
    print(f"      MusicXML score saved: {output_xml_path}")


def run_pipeline(youtube_url: str, output_basename: str = "piano_transcription",
                 bpm: int | None = None):
    temp_wav = "temp_audio.wav"
    raw_midi = f"{output_basename}_raw.mid"
    clean_midi = f"{output_basename}.mid"
    xml_sheet = f"{output_basename}.musicxml"

    try:
        download_youtube_audio(youtube_url, output_wav_path=temp_wav)
        transcribe_audio_to_midi(temp_wav, raw_midi_path=raw_midi)

        print("[3/4] Detecting tempo & building one line per hand...")
        if bpm is None:
            bpm = estimate_bpm(temp_wav)
        print(f"      Using tempo: {bpm} BPM")
        rh, lh = extract_hand_events(raw_midi, temp_wav, bpm)
        save_clean_midi(rh, lh, bpm, clean_midi)

        print("[4/4] Exporting MusicXML Grand Staff score...")
        build_grand_staff_score(rh, lh, bpm, xml_sheet)

        print("\n[✓] Pipeline Finished Successfully!")
        print(f"    - Clean MIDI File: {Path(clean_midi).resolve()}")
        print(f"    - Sheet Music XML: {Path(xml_sheet).resolve()}")
    finally:
        for temp_file in [temp_wav, raw_midi]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter YouTube Piano Video Link: ").strip()
    manual_bpm = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if url:
        run_pipeline(url, bpm=manual_bpm)
    else:
        print("No URL provided. Exiting.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube Piano Video Link: ").strip()

    if url:
        run_pipeline(url)
    else:
        print("No URL provided. Exiting.")