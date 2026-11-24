import base64
import http.cookiejar
import json
import os
import re
import subprocess
import tempfile
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
        v = np.zeros(12); v[r] = 1.0; v[(r+4)%12] = 0.8; v[(r+7)%12] = 0.85
        names.append(NOTE_NAMES[r]); mat.append(v / np.linalg.norm(v))
        v = np.zeros(12); v[r] = 1.0; v[(r+3)%12] = 0.8; v[(r+7)%12] = 0.85
        names.append(NOTE_NAMES[r] + 'm'); mat.append(v / np.linalg.norm(v))
        v = np.zeros(12); v[r] = 1.0; v[(r+4)%12] = 0.75; v[(r+7)%12] = 0.75; v[(r+10)%12] = 0.65
        names.append(NOTE_NAMES[r] + '7'); mat.append(v / np.linalg.norm(v))
        v = np.zeros(12); v[r] = 1.0; v[(r+4)%12] = 0.75; v[(r+7)%12] = 0.75; v[(r+11)%12] = 0.65
        names.append(NOTE_NAMES[r] + 'M7'); mat.append(v / np.linalg.norm(v))
        v = np.zeros(12); v[r] = 1.0; v[(r+3)%12] = 0.75; v[(r+7)%12] = 0.75; v[(r+10)%12] = 0.65
        names.append(NOTE_NAMES[r] + 'm7'); mat.append(v / np.linalg.norm(v))
    return names, np.array(mat)

CHORD_NAMES, CHORD_MAT = build_chord_templates()

def validate_netscape(content: str) -> str:
    if not content:
        return ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write(content)
            tpath = tf.name
        try:
            cj = http.cookiejar.MozillaCookieJar(tpath)
            cj.load(ignore_discard=True, ignore_expires=True)
            if len(cj) > 0:
                return content
        finally:
            if os.path.exists(tpath):
                os.unlink(tpath)
    except Exception:
        pass
    return ""

def normalize_to_netscape(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")

    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "cookies" in data:
                data = data["cookies"]
            if isinstance(data, list):
                lines = ["# Netscape HTTP Cookie File"]
                for c in data:
                    if not isinstance(c, dict) or "name" not in c:
                        continue
                    domain = str(c.get("domain", ".youtube.com")).strip()
                    flag = "TRUE" if domain.startswith(".") else "FALSE"
                    path = str(c.get("path", "/")).strip()
                    secure = "TRUE" if c.get("secure", True) else "FALSE"
                    exp = str(int(c.get("expirationDate", 0) or 0))
                    name = re.sub(r"\s+", "", str(c["name"]))
                    val = re.sub(r"[\r\n]+", "", str(c.get("value", "")))
                    if name:
                        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{exp}\t{name}\t{val}")
                res = validate_netscape("\n".join(lines) + "\n")
                if res:
                    return res
        except Exception:
            pass

    valid_tab_lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) == 7 and all(p != "" for p in parts[:6]):
            valid_tab_lines.append("\t".join(parts))
    if valid_tab_lines:
        res = validate_netscape("# Netscape HTTP Cookie File\n" + "\n".join(valid_tab_lines) + "\n")
        if res:
            return res

    if ".youtube.com" in text or ".google.com" in text:
        pattern = r'(\.?[a-zA-Z0-9.-]*youtube\.com|\.?google\.com|\.?googlevideo\.com)\s+(TRUE|FALSE)\s+(\S+)\s+(TRUE|FALSE)\s+(\d+)\s+(\S+)\s+(\S+)'
        matches = re.findall(pattern, text)
        if matches:
            lines = ["# Netscape HTTP Cookie File"] + ["\t".join(m) for m in matches]
            res = validate_netscape("\n".join(lines) + "\n")
            if res:
                return res

    collapsed = re.sub(r"[\r\n]+", "", text)
    if "=" in collapsed:
        lines = ["# Netscape HTTP Cookie File"]
        for part in collapsed.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k = re.sub(r"\s+", "", k)
                v = v.strip()
                if k and not any(ch in k for ch in [" ", "\t", ";"]):
                    lines.append(f".youtube.com\tTRUE\t/\tTRUE\t0\t{k}\t{v}")
        if len(lines) > 1:
            res = validate_netscape("\n".join(lines) + "\n")
            if res:
                return res

    return ""

def get_cookies_text() -> str:
    for secret_path in ["/etc/secrets/cookies.txt", "cookies.txt"]:
        if os.path.exists(secret_path):
            try:
                with open(secret_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = normalize_to_netscape(f.read())
                    if content:
                        return content
            except Exception:
                pass

    b64 = os.environ.get("YT_COOKIES_B64", "").strip()
    if b64:
        try:
            decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
            content = normalize_to_netscape(decoded)
            if content:
                return content
        except Exception:
            pass

    raw = os.environ.get("YT_COOKIES", "").strip()
    if raw:
        return normalize_to_netscape(raw)
    return ""

def count_cookies(netscape_text: str) -> int:
    if not netscape_text:
        return 0
    return sum(1 for ln in netscape_text.splitlines() if ln.strip() and not ln.startswith("#"))

class YouTubeReq(BaseModel):
    url: str

@app.get("/")
def health():
    c = get_cookies_text()
    return {
        "status": "ok",
        "service": "Tab Studio AI Chord Recognizer",
        "cookies_configured": bool(c),
        "cookie_count": count_cookies(c)
    }

@app.post("/analyze-youtube")
def analyze_youtube(req: YouTubeReq):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "yt-dlp", "--no-playlist", "--remote-components", "ejs:github", "-x", "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 22050 -t 90",
            "-o", os.path.join(tmp, "raw.%(ext)s"),
        ]
        cookies_text = get_cookies_text()
        if cookies_text:
            cookie_path = os.path.join(tmp, "cookies.txt")
            with open(cookie_path, "w", encoding="utf-8") as cf:
                cf.write(cookies_text)
            cmd.extend(["--cookies", cookie_path])

        cmd.append(req.url)
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        wav_files = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".wav")]
        if not wav_files:
            return {"error": "유튜브 오디오 추출 실패: " + (res.stderr[-250:] if res.stderr else "unknown")}

        y, sr = librosa.load(wav_files[0], sr=22050, mono=True)
        y_harm = librosa.effects.harmonic(y, margin=3.0)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else 100.0

        chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr)
        if len(beat_frames) < 4:
            beat_frames = np.arange(0, chroma.shape[1], max(1, int(sr * 0.5 / 512)))

        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        chords = []
        for i in range(0, min(64, beat_chroma.shape[1]), 2):
            vec = np.mean(beat_chroma[:, i:i+2], axis=1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            scores = CHORD_MAT @ vec
            chords.append(CHORD_NAMES[int(np.argmax(scores))])

        return {"bpm": round(bpm), "chords": chords}
