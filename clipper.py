import os
import json
import textwrap
import subprocess
import requests
import cv2
import numpy as np

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
VIDEO_URL = os.environ["VIDEO_URL"]
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "3"))

os.makedirs("work", exist_ok=True)
os.makedirs("output", exist_ok=True)

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def download_video():
    print("1. Mengunduh video sumber...")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", "work/source.mp4",
        VIDEO_URL
    ]
    if os.path.exists("cookies.txt"):
        cmd.extend(["--cookies", "cookies.txt"])
    run_cmd(cmd)

    run_cmd([
        "ffmpeg", "-y", "-i", "work/source.mp4",
        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k",
        "work/audio.mp3"
    ])

def transcribe_audio():
    print("2. Membuat transkrip presisi per kata dengan Groq Whisper...")
    with open("work/audio.mp3", "rb") as f:
        res = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            data={
                "model": "whisper-large-v3",
                "language": "id",
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"]
            },
            timeout=300
        )
    res.raise_for_status()
    data = res.json()
    with open("work/transcript.json", "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
    return data

def curate_viral_clips(transcript_data):
    print("3. Menganalisis momen paling viral dengan Gemini AI...")
    segments = transcript_data.get("segments", [])
    compact_transcript = "\n".join(
        [f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text'].strip()}" for s in segments]
    )

    prompt = f"""
Kamu adalah editor dan ahli strategi viral TikTok Indonesia untuk niche podcast, berita, dan perkembangan negara.
Analisis transkrip video berikut dan pilih {NUM_CLIPS} potongan paling berpotensi FYP (kontroversial, emosional, fakta mengejutkan, atau perdebatan tajam).

Aturan ketat:
1. Durasi tiap klip wajib antara 30 sampai 60 detik.
2. Potongan harus dimulai tepat di awal kalimat yang memancing rasa penasaran (strong hook) dan berakhir di kalimat penutup yang tuntas.
3. Buat 'hook_title' maksimal 6 kata huruf kapital untuk dipasang di bagian atas layar video.
4. Buat 'caption' TikTok yang memancing komentar beserta 5 hashtag relevan.

Keluarkan HANYA JSON valid dengan struktur:
{{
  "clips": [
    {{
      "start": 120.5,
      "end": 165.0,
      "viral_score": 92,
      "hook_title": "FAKTA MENGEJUTKAN DI BALIK KEBIJAKAN INI",
      "caption": "Setuju atau nggak sama pernyataan ini? Tulis pendapat kalian di bawah! #podcastindonesia #beritaterkini #fyp",
      "reason": "Alasan singkat kenapa segmen ini berpotensi viral"
    }}
  ]
}}

Transkrip:
{compact_transcript[:95000]}
"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    res = requests.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}
        },
        timeout=120
    )
    res.raise_for_status()
    raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw_text)["clips"]

def detect_speaker_crop_x(video_path, start_s, end_s):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = int(height * 9 / 16)

    if crop_w >= width:
        cap.release()
        return 0, width, width, height

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    centers_x = []
    t = start_s
    while t < end_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            centers_x.append(x + w // 2)
        t += 0.7

    cap.release()
    target_x = int(np.median(centers_x)) if centers_x else (width // 2)
    crop_x = max(0, min(width - crop_w, target_x - crop_w // 2))
    return crop_x, crop_w, width, height

def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def build_ass_subtitle(words, start_s, end_s, hook_title, ass_path):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,DejaVu Sans,54,&H00FFFFFF,&H000000FF,&H00000000,&HCC000000,-1,0,0,0,100,100,0,0,3,14,0,8,60,60,180,1
Style: Sub,DejaVu Sans,68,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,70,70,430,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    duration = end_s - start_s
    wrapped_hook = r"\N".join(textwrap.wrap(hook_title.upper(), width=24))
    events = [f"Dialogue: 1,0:00:00.00,{format_ass_time(duration)},Hook,,0,0,0,,{wrapped_hook}"]

    clip_words = [w for w in words if w["end"] >= start_s and w["start"] <= end_s]
    chunk_size = 3
    for i in range(0, len(clip_words), chunk_size):
        chunk = clip_words[i:i + chunk_size]
        w_start = max(0.0, chunk[0]["start"] - start_s)
        w_end = min(duration, chunk[-1]["end"] - start_s)
        if w_end <= w_start:
            continue
        text = " ".join([w["word"].strip().upper() for w in chunk])
        events.append(f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},Sub,,0,0,0,,{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

def render_clips(clips, transcript_data):
    print("4. Merender klip vertikal 9:16 + subtitle...")
    words = transcript_data.get("words", [])
    results_meta = []

    for idx, clip in enumerate(clips, start=1):
        start_s = float(clip["start"])
        end_s = float(clip["end"])
        duration = end_s - start_s
        out_mp4 = f"output/klip_{idx}.mp4"
        ass_file = f"work/sub_{idx}.ass"

        crop_x, crop_w, _, height = detect_speaker_crop_x("work/source.mp4", start_s, end_s)
        build_ass_subtitle(words, start_s, end_s, clip["hook_title"], ass_file)

        vf = f"crop={crop_w}:{height}:{crop_x}:0,scale=1080:1920,ass={ass_file}"
        run_cmd([
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", "work/source.mp4",
            "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            out_mp4
        ])

        clip["file"] = out_mp4
        results_meta.append(clip)
        print(f"Selesai merender: {out_mp4}")

    with open("output/hasil_kurasi.json", "w", encoding="utf-8") as f:
        json.dump(results_meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    download_video()
    transcript = transcribe_audio()
    clips = curate_viral_clips(transcript)
    render_clips(clips, transcript)
