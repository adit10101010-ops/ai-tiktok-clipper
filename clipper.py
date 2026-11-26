import os
import json
import shutil
import textwrap
import subprocess
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
VIDEO_URL = os.environ["VIDEO_URL"].strip()
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "3"))
CLIP_DURATION = os.environ.get("CLIP_DURATION", "60-90 detik").strip()

os.makedirs("work", exist_ok=True)
os.makedirs("output", exist_ok=True)

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def get_video_resolution(path):
    if not os.path.exists(path):
        return 0, 0
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def download_video():
    print("1. Mengunduh video sumber kualitas HD (target 1080p)...")
    if "drive.google.com" in VIDEO_URL:
        run_cmd(["gdown", "--fuzzy", VIDEO_URL, "-O", "work/source.mp4"])
    else:
        clean_url = VIDEO_URL.split("?si=")[0].split("&si=")[0]
        fmt = "bv*[height>=720][height<=1080]+ba/bv*[height<=1080]+ba/b[height>=720]/b"

        cmd1 = [
            "yt-dlp",
            "--remote-components", "ejs:github",
            "--extractor-args", "youtube:player_client=tv,web_safari,web",
            "-S", "res:1080,fps",
            "-f", fmt,
            "--merge-output-format", "mp4",
            "-o", "work/source.mp4",
            clean_url
        ]
        if os.path.exists("cookies.txt"):
            cmd1.extend(["--cookies", "cookies.txt"])
        run_cmd(cmd1)

        w, h = get_video_resolution("work/source.mp4")
        print(f"Resolusi percobaan 1: {w}x{h}")

        if h < 720:
            print("Resolusi masih di bawah 720p, mencoba jalur klien HD cadangan...")
            shutil.move("work/source.mp4", "work/source_fallback.mp4")
            cmd2 = [
                "yt-dlp",
                "--remote-components", "ejs:github",
                "--extractor-args", "youtube:player_client=android_vr,tv",
                "-S", "res:1080,fps",
                "-f", fmt,
                "--merge-output-format", "mp4",
                "-o", "work/source.mp4",
                clean_url
            ]
            try:
                run_cmd(cmd2)
                w2, h2 = get_video_resolution("work/source.mp4")
                print(f"Resolusi percobaan 2: {w2}x{h2}")
                if h2 <= h:
                    shutil.move("work/source_fallback.mp4", "work/source.mp4")
                else:
                    os.remove("work/source_fallback.mp4")
            except Exception as e:
                print(f"Jalur cadangan dilewati ({e}), memakai hasil pertama.")
                shutil.move("work/source_fallback.mp4", "work/source.mp4")

    w, h = get_video_resolution("work/source.mp4")
    print(f"Resolusi final video sumber: {w}x{h}")

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

def get_candidate_gemini_models():
    res = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}",
        timeout=30
    )
    res.raise_for_status()
    excluded = [
        "embedding", "aqa", "imagen", "veo", "tts", "audio",
        "image", "robotics", "computer-use", "customtools",
        "omni", "deep-research"
    ]
    available = [
        m["name"] for m in res.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and "gemini" in m["name"].lower()
        and not any(x in m["name"].lower() for x in excluded)
    ]

    stable_flash = sorted(
        [m for m in available if "flash" in m.lower() and "preview" not in m.lower() and "exp" not in m.lower()],
        reverse=True
    )
    preview_flash = sorted(
        [m for m in available if "flash" in m.lower() and m not in stable_flash],
        reverse=True
    )
    others = sorted([m for m in available if m not in stable_flash and m not in preview_flash], reverse=True)

    ordered = []
    for m in stable_flash + preview_flash + others:
        if m not in ordered:
            ordered.append(m)
    return ordered[:6]

def curate_viral_clips(transcript_data):
    print(f"3. Menganalisis momen paling viral (target {CLIP_DURATION}) dengan Gemini AI...")
    segments = transcript_data.get("segments", [])
    compact_transcript = "\n".join(
        [f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text'].strip()}" for s in segments]
    )

    prompt = f"""
Kamu adalah creative director & editor viral TikTok Indonesia (niche podcast, mindset, bisnis, berita).
Analisis transkrip video berikut dan pilih {NUM_CLIPS} potongan paling berpotensi FYP (kontroversial, tamparan realitas, fakta mengejutkan, atau perdebatan tajam).

Aturan ketat:
1. Durasi tiap klip wajib berada di kisaran {CLIP_DURATION}, dimulai tepat pada kalimat pembuka yang kuat dan ditutup tuntas pada akhir argumen/pembahasan.
2. 'hook_badge': label kategori singkat 2 KATA HURUF KAPITAL (contoh: "TAMPARAN KERAS", "FAKTA TERSEMBUNYI", "WAJIB TAHU", "MINDSET MAHAL").
3. 'hook_title': judul pemicu rasa penasaran/emosi 5 sampai 8 KATA HURUF KAPITAL. Hindari kalimat datar/klise; gunakan pola kontras atau pertanyaan menohok (contoh: "PANTAS HIDUPMU STAGNAN TERUS KALAU CIRCLE-NYA BEGINI").
4. 'highlight_words': daftar 2 sampai 3 kata kunci paling kuat dari 'hook_title' (HURUF KAPITAL persis seperti di hook_title) untuk diwarnai kuning neon.
5. 'caption': caption TikTok yang memancing komentar + 5 hashtag relevan.

Keluarkan HANYA JSON valid dengan struktur:
{{
  "clips": [
    {{
      "start": 120.5,
      "end": 205.0,
      "viral_score": 94,
      "hook_badge": "TAMPARAN KERAS",
      "hook_title": "PANTAS HIDUPMU STAGNAN TERUS KALAU CIRCLE-NYA BEGINI",
      "highlight_words": ["STAGNAN", "TERUS", "CIRCLE-NYA"],
      "caption": "Setuju atau nggak sama pernyataan ini? Tulis pendapat kalian di bawah! #podcastindonesia #mindset #fyp",
      "reason": "Alasan singkat kenapa segmen ini berpotensi viral"
    }}
  ]
}}

Transkrip:
{compact_transcript[:95000]}
"""
    candidates = get_candidate_gemini_models()
    last_error = None

    for model_name in candidates:
        print(f"Mencoba model Gemini: {model_name}")
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_API_KEY}"
        res = requests.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.5}
            },
            timeout=120
        )
        if res.status_code in (404, 429, 503):
            print(f"Model {model_name} dilewati (status {res.status_code}), lanjut ke cadangan...")
            last_error = f"{res.status_code}: {res.text[:200]}"
            continue
        res.raise_for_status()
        raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, list) else parsed["clips"]

    raise RuntimeError(f"Semua kandidat model Gemini gagal diakses. Error terakhir: {last_error}")

def get_bold_font_path():
    candidates = [
        "Poppins-ExtraBold.ttf",
        "work/Poppins-ExtraBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            return p
    try:
        r = requests.get(
            "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-ExtraBold.ttf",
            timeout=20
        )
        if r.ok and len(r.content) > 10000:
            with open("work/Poppins-ExtraBold.ttf", "wb") as f:
                f.write(r.content)
            return "work/Poppins-ExtraBold.ttf"
    except Exception:
        pass
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def create_hook_banner_png(hook_title, hook_badge, highlight_words, out_png_path):
    W, H = 1080, 390
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_path = get_bold_font_path()
    try:
        font_badge = ImageFont.truetype(font_path, 24)
        font_title = ImageFont.truetype(font_path, 44)
    except Exception:
        font_badge = ImageFont.load_default()
        font_title = ImageFont.load_default()

    badge_text = (hook_badge or "PODCAST HIGHLIGHT").upper().strip()
    title_text = (hook_title or "").upper().strip()
    hl_set = {w.upper().strip(".,!?\"'") for w in (highlight_words or [])}

    lines = textwrap.wrap(title_text, width=25)[:3] or ["SIMAK PENJELASAN INI"]
    line_height = 56
    box_pad_x = 40
    box_pad_y = 26

    max_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_title)
        max_w = max(max_w, bbox[2] - bbox[0])

    card_w = min(W - 96, max(520, max_w + box_pad_x * 2))
    card_h = len(lines) * line_height + box_pad_y * 2 + 8
    card_x1 = (W - card_w) // 2
    card_y1 = 44
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h

    draw.rounded_rectangle(
        [card_x1 + 5, card_y1 + 8, card_x2 + 5, card_y2 + 8],
        radius=24,
        fill=(0, 0, 0, 175)
    )
    draw.rounded_rectangle(
        [card_x1, card_y1, card_x2, card_y2],
        radius=24,
        fill=(11, 17, 32, 232),
        outline=(250, 204, 21, 255),
        width=4
    )

    b_bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
    b_w = (b_bbox[2] - b_bbox[0]) + 42
    b_h = 40
    b_x1 = (W - b_w) // 2
    b_y1 = card_y1 - 20
    draw.rounded_rectangle(
        [b_x1, b_y1, b_x1 + b_w, b_y1 + b_h],
        radius=20,
        fill=(220, 38, 38, 255)
    )
    draw.text(
        (b_x1 + 21, b_y1 + 6),
        badge_text,
        font=font_badge,
        fill=(255, 255, 255, 255)
    )

    curr_y = card_y1 + box_pad_y + 10
    for line in lines:
        words = line.split(" ")
        space_w = draw.textbbox((0, 0), " ", font=font_title)[2]
        word_widths = [draw.textbbox((0, 0), w, font=font_title)[2] for w in words]
        total_line_w = sum(word_widths) + space_w * (len(words) - 1)
        curr_x = (W - total_line_w) // 2

        for w, w_width in zip(words, word_widths):
            clean_w = w.strip(".,!?\"'")
            is_hl = clean_w in hl_set
            color = (250, 204, 21, 255) if is_hl else (255, 255, 255, 255)
            draw.text((curr_x + 2, curr_y + 2), w, font=font_title, fill=(0, 0, 0, 220))
            draw.text((curr_x, curr_y), w, font=font_title, fill=color)
            curr_x += w_width + space_w
        curr_y += line_height

    img.save(out_png_path)
    return out_png_path

def load_dnn_face_detector():
    if not hasattr(cv2, "dnn"):
        return None
    proto_path = "work/deploy.prototxt"
    model_path = "work/res10_300x300_ssd_iter_140000.caffemodel"
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    try:
        if not os.path.exists(proto_path):
            r = requests.get(proto_url, timeout=30)
            r.raise_for_status()
            with open(proto_path, "wb") as f:
                f.write(r.content)
        if not os.path.exists(model_path):
            r = requests.get(model_url, timeout=60)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)
        return cv2.dnn.readNetFromCaffe(proto_path, model_path)
    except Exception as e:
        print(f"DNN face detector dilewati ({e}), beralih ke pelacak gerak pembicara.")
        return None

def detect_faces_dnn(net, frame, width, height):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < 0.55:
            continue
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        w, h = x2 - x1, y2 - y1
        center_y = y1 + h * 0.5
        if w > width * 0.03 and h > height * 0.06 and (height * 0.06) < center_y < (height * 0.68):
            faces.append((x1, y1, w, h))
    return faces

def detect_active_speaker_expression(video_path, start_s, end_s):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = (int(height * 9 / 16) // 2) * 2
    duration = end_s - start_s

    if crop_w >= width:
        cap.release()
        return "0", width, width, height

    dnn_net = load_dnn_face_detector()
    step = 0.4
    samples = []
    last_center = width // 2
    t = start_s

    while t < end_s:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret1, frame1 = cap.read()
        if not ret1:
            break

        for _ in range(3):
            ret2, frame2 = cap.read()
            if not ret2:
                frame2 = frame1
                break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        faces = detect_faces_dnn(dnn_net, frame1, width, height) if dnn_net is not None else []

        if len(faces) == 1:
            x, y, w, h = faces[0]
            last_center = x + w // 2
        elif len(faces) >= 2:
            best_score = -1.0
            chosen_center = last_center
            for (x, y, w, h) in faces:
                m_y1 = y + int(h * 0.55)
                m_y2 = min(height, y + h)
                m_x1 = x + int(w * 0.20)
                m_x2 = min(width, x + int(w * 0.80))
                if m_y2 > m_y1 and m_x2 > m_x1:
                    roi1 = gray1[m_y1:m_y2, m_x1:m_x2]
                    roi2 = gray2[m_y1:m_y2, m_x1:m_x2]
                    motion = float(np.mean(cv2.absdiff(roi1, roi2)))
                else:
                    motion = 0.0
                score = motion * ((w * h) ** 0.25)
                if score > best_score:
                    best_score = score
                    chosen_center = x + w // 2
            last_center = chosen_center
        else:
            y_top, y_bot = int(height * 0.10), int(height * 0.62)
            diff_roi = cv2.absdiff(gray1[y_top:y_bot, :], gray2[y_top:y_bot, :])
            col_profile = np.mean(diff_roi, axis=0)
            kernel_w = max(15, crop_w // 2)
            smooth_profile = np.convolve(col_profile, np.ones(kernel_w) / kernel_w, mode="same")
            if float(np.max(smooth_profile)) > 2.5:
                last_center = int(np.argmax(smooth_profile))

        crop_x = max(0, min(width - crop_w, last_center - crop_w // 2))
        samples.append((round(t - start_s, 2), int(crop_x)))
        t += step

    cap.release()

    if not samples:
        return f"{(width - crop_w) // 2}", crop_w, width, height

    segments = []
    cur_start = 0.0
    cur_x = samples[0][1]
    pending_x = None
    pending_count = 0
    threshold = int(crop_w * 0.25)

    for rel_t, sx in samples[1:]:
        if abs(sx - cur_x) > threshold:
            if pending_x is not None and abs(sx - pending_x) <= threshold:
                pending_count += 1
            else:
                pending_x = sx
                pending_count = 1

            if pending_count >= 3:
                switch_t = max(cur_start + 0.5, round(rel_t - (step * 2), 2))
                segments.append((cur_start, switch_t, cur_x))
                cur_start = switch_t
                cur_x = pending_x
                pending_x = None
                pending_count = 0
        else:
            cur_x = int(0.7 * cur_x + 0.3 * sx)
            pending_x = None
            pending_count = 0

    segments.append((cur_start, round(duration + 0.5, 2), cur_x))
    print(f"Pergerakan kamera terdeteksi ({len(segments)} segmen fokus pembicara).")

    if len(segments) == 1:
        x_expr = str(segments[0][2])
    else:
        terms = []
        for idx, (s_t, e_t, x_val) in enumerate(segments):
            op = "lte" if idx == len(segments) - 1 else "lt"
            terms.append(f"gte(t,{s_t:.2f})*{op}(t,{e_t:.2f})*{x_val}")
        x_expr = "+".join(terms)

    return x_expr, crop_w, width, height

def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def build_ass_subtitle(words, start_s, end_s, ass_path):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Poppins ExtraBold,66,&H0000FFFF,&H00FFFFFF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,410,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    duration = end_s - start_s
    events = []

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
    print("4. Merender klip vertikal 9:16 Full HD + kartu judul modern...")
    words = transcript_data.get("words", [])
    results_meta = []

    for idx, clip in enumerate(clips, start=1):
        start_s = float(clip["start"])
        end_s = float(clip["end"])
        duration = end_s - start_s
        out_mp4 = f"output/klip_{idx}.mp4"
        ass_file = f"work/sub_{idx}.ass"
        banner_png = f"work/banner_{idx}.png"

        x_expr, crop_w, _, height = detect_active_speaker_expression("work/source.mp4", start_s, end_s)
        build_ass_subtitle(words, start_s, end_s, ass_file)
        create_hook_banner_png(
            clip.get("hook_title", ""),
            clip.get("hook_badge", "PODCAST HIGHLIGHT"),
            clip.get("highlight_words", []),
            banner_png
        )

        vf = (
            f"crop={crop_w}:{height}:'{x_expr}':0,"
            f"scale=1080:1920:flags=lanczos,"
            f"unsharp=5:5:0.7:3:3:0.3,"
            f"ass={ass_file}[base];"
            f"movie={banner_png}[banner];"
            f"[base][banner]overlay=(W-w)/2:145"
        )
        run_cmd([
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", "work/source.mp4",
            "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            out_mp4
        ])

        clip["file"] = out_mp4
        results_meta.append(clip)
        print(f"Selesai merender HD ({duration:.1f} detik): {out_mp4}")

    with open("output/hasil_kurasi.json", "w", encoding="utf-8") as f:
        json.dump(results_meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    download_video()
    transcript = transcribe_audio()
    clips = curate_viral_clips(transcript)
    render_clips(clips, transcript)
