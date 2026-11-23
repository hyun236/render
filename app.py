import os, subprocess, tempfile
import numpy as np
import librosa
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Tab Studio AI Chord Recognizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def build_chord_templates():
    names, mat = [], []
    for r in range(12):
        # Major
        v = np.zeros(12); v[r] = 1.0; v[(r+4)%12] = 0.8; v[(r+7)%12] = 0.85
        names.append(NOTE_NAMES[r]); mat.append(v / np.linalg.norm(v))
        # Minor
        v = np.zeros(12); v[r] = 1.0; v[(r+3)%12] = 0.8; v[(r+7)%12] = 0.85
        names.append(NOTE_NAMES[r] + 'm'); mat.append(v / np.linalg.norm(v))
        # Dom7
        v = np.zeros(12); v[r] = 1.0; v[(r+4)%12] = 0.75; v[(r+7)%12] = 0.75; v[(r+10)%12] = 0.65
        names.append(NOTE_NAMES[r] + '7'); mat.append(v / np.linalg.norm(v))
        # Maj7
        v = np.zeros(12); v[r] = 1.0; v[(r+4)%12] = 0.75; v[(r+7)%12] = 0.75; v[(r+11)%12] = 0.65
        names.append(NOTE_NAMES[r] + 'M7'); mat.append(v / np.linalg.norm(v))
        # Min7
        v = np.zeros(12); v[r] = 1.0; v[(r+3)%12] = 0.75; v[(r+7)%12] = 0.75; v[(r+10)%12] = 0.65
        names.append(NOTE_NAMES[r] + 'm7'); mat.append(v / np.linalg.norm(v))
    return names, np.array(mat)

CHORD_NAMES, CHORD_MAT = build_chord_templates()

class YouTubeReq(BaseModel):
    url: str

@app.get("/")
def health():
    return {"status": "ok", "service": "Tab Studio AI Chord Recognizer"}

@app.post("/analyze-youtube")
def analyze_youtube(req: YouTubeReq):
    with tempfile.TemporaryDirectory() as tmp:
        out_wav = os.path.join(tmp, "audio.wav")
        cmd = [
            "yt-dlp", "-x", "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 22050 -t 90",
            "-o", os.path.join(tmp, "raw.%(ext)s"),
            req.url
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        wav_files = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".wav")]
        if not wav_files:
            return {"error": "유튜브 오디오 추출 실패: " + (res.stderr[-200:] if res.stderr else "unknown")}

        y, sr = librosa.load(wav_files[0], sr=22050, mono=True)
        y_harm = librosa.effects.harmonic(y, margin=3.0)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else 100.0

        chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr)
        if len(beat_frames) < 4:
            beat_frames = np.arange(0, chroma.shape[1], max(1, int(sr * 0.5 / 512)))

        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        # Group every 2 beats (half-measure) for clean chord progression
        chords = []
        for i in range(0, min(64, beat_chroma.shape[1]), 2):
            vec = np.mean(beat_chroma[:, i:i+2], axis=1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            scores = CHORD_MAT @ vec
            chords.append(CHORD_NAMES[int(np.argmax(scores))])

        return {"bpm": round(bpm), "chords": chords}
