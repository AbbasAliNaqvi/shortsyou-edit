import sys
import shutil
import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

import subprocess
import tempfile
import httpx
import json
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .cropper import crop_to_vertical
from .captions import add_captions
from .audio_mixer import mix_mastered_audio, normalize_dialogue
from .ai_director import create_execution_blueprint
from .pacing import apply_camera_punch_ins
from .backgrounds import apply_background
from .grader import apply_color_grade
from .cleaner import remove_silences_and_fillers
from services.sfx import add_sfx, auto_sfx_for_emotion


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

STYLE_CONFIGS = {
    "clean": {
        "background": "crop",
        "caption_style": "white_bold_bottom",
        "color_grade": "warm",
        "music_mood": "calm",
        "font_size": 52,
        "text_color": "white",
        "highlight_color": "#FFEE00",
    },
    "bold": {
        "background": "crop",
        "caption_style": "karaoke_yellow",
        "color_grade": "vibrant",
        "music_mood": "energetic",
        "font_size": 64,
        "text_color": "#FFFF00",
        "highlight_color": "#FFEE00",
    },
    "minimal": {
        "background": "crop",
        "caption_style": "small_white",
        "color_grade": "natural",
        "music_mood": None,
        "font_size": 40,
        "text_color": "white",
        "highlight_color": "#FFEE00",
    }
}

async def create_short(req) -> dict:
    config = STYLE_CONFIGS.get(req.style, STYLE_CONFIGS["clean"])
    # Explicit editor controls win over the style preset. This keeps the API
    # backwards compatible while allowing creators to mix a custom look.
    config = {**config}
    if getattr(req, "background_style", None):
        config["background"] = req.background_style
    if getattr(req, "color_grade", None):
        config["color_grade"] = req.color_grade
    if getattr(req, "caption_style", None):
        apply_caption_style(config, req.caption_style)

    t_job_start = time.time()
    clip_len = max(0.1, req.end_time - req.start_time)
    print(
        f"[renderer] >>> Starting job #{req.job_id} (clip #{req.clip_id}) | "
        f"style={req.style} layout={req.layout} window={req.start_time:.2f}s-{req.end_time:.2f}s ({clip_len:.1f}s)"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Step 1: fetch source video
        t_fetch = time.time()
        print(f"[fetch] downloading video segment from source ({req.start_time:.1f}s to {req.end_time:.1f}s)...")
        raw_path = tmp / "raw.mp4"
        source_was_clipped = await download_file(
            req.video_url, raw_path, req.start_time, req.end_time
        )
        file_size_mb = raw_path.stat().st_size / (1024 * 1024) if raw_path.is_file() else 0.0
        print(f"[fetch] source video ready in {time.time() - t_fetch:.1f}s ({file_size_mb:.1f} MB)")

        # Step 2: Extract the clip segment
        t_extract = time.time()
        clip_path = tmp / "clip.mp4"
        if source_was_clipped:
            extract_clip(raw_path, clip_path, 0, req.end_time - req.start_time)
        else:
            extract_clip(raw_path, clip_path, req.start_time, req.end_time)
        print(f"[renderer] clip segment extracted in {time.time() - t_extract:.1f}s")

        # Step 3: Silence removal
        clean_path = tmp / "clean.mp4"
        if req.remove_silences and not req.caption_words:
            t_silence = time.time()
            print("[silence] analyzing and trimming dead silences...")
            remove_silences_and_fillers(clip_path, clean_path)
            print(f"[silence] silence removal complete in {time.time() - t_silence:.1f}s")
        else:
            if req.remove_silences and req.caption_words:
                print("[renderer] preserving source timing for word-perfect captions")
            clean_path = clip_path

        clip_duration = get_video_duration(clean_path)
        caption_data = [word.model_dump() for word in req.caption_words]

        # Step 4: Auto-transcription if captions missing
        if not caption_data:
            t_asr = time.time()
            print("[transcription] no captions supplied -> auto-transcribing with Groq whisper-large-v3...")
            from services.transcription import auto_transcribe_clip
            transcribed_words, transcript_text = auto_transcribe_clip(clean_path)
            if transcribed_words:
                caption_data = transcribed_words
                print(f"[transcription] auto-transcription complete: {len(caption_data)} words transcribed in {time.time() - t_asr:.1f}s")
            else:
                transcript_text = ""
        else:
            transcript_text = " ".join(item["word"] for item in caption_data)

        # Step 5: Creative AI Director
        t_dir = time.time()
        print("[director] consulting AI Director for edit blueprint...")
        director_plan = create_execution_blueprint(
            transcript_text, caption_data, clip_duration, req.style, req.emotion_type or "excited"
        )
        print(f"[director] edit blueprint finalized in {time.time() - t_dir:.1f}s")

        if req.layout in {"two_frame", "multi_face", "auto_face"}:
            t_layout = time.time()
            print(f"[layout] formatting adaptive face layout: {req.layout}...")

            from services.two_frame import create_adaptive_face_short

            two_frame_path = tmp / "face_layout.mp4"
            fallback_path = tmp / "vertical.mp4"

            success = create_adaptive_face_short(
                clean_path,
                two_frame_path,
                fallback_path,
                mode=req.layout,
            )

            if success:
                working_path = two_frame_path
            else:
                working_path = fallback_path
            print(f"[layout] adaptive face layout complete in {time.time() - t_layout:.1f}s")

            # Apply color grade
            t_grade = time.time()
            graded_path = tmp / "graded.mp4"
            print(f"[color] applying color grade: {config['color_grade']}...")
            apply_color_grade(
                working_path,
                graded_path,
                config["color_grade"],
            )
            print(f"[color] color grade applied in {time.time() - t_grade:.1f}s")

            t_pace = time.time()
            paced_path = tmp / "paced.mp4"
            print(f"[pacing] applying {len(director_plan['camera_punch_ins'])} camera punch-ins...")
            apply_camera_punch_ins(graded_path, paced_path, clip_duration, director_plan["camera_punch_ins"])
            print(f"[pacing] camera punch-ins complete in {time.time() - t_pace:.1f}s")

            # Generate and add captions
            t_cap = time.time()
            captioned_path = tmp / "captioned.mp4"
            print("[captions] burning studio-grade subtitles & hook card...")
            add_captions(
                paced_path,
                captioned_path,
                config,
                hook_text=req.hook_text,
                caption_words=caption_data,
                director_plan=director_plan,
            )
            print(f"[captions] subtitles & hook burned in {time.time() - t_cap:.1f}s")

            # Select and mix background music
            t_audio = time.time()
            final_path = tmp / "final.mp4"
            music = director_plan["bg_music"]
            mood = req.music_mood or music["mood"] or config["music_mood"]
            if mood and mood != "none":
                print(f"[audio] mixing background music ({mood}) with speech ducking...")
                mix_mastered_audio(captioned_path, final_path, mood, music["base_volume_db"], music["duck_during_speech_db"])
            else:
                normalize_dialogue(captioned_path, final_path)
            print(f"[audio] music mastering complete in {time.time() - t_audio:.1f}s")

        else:
            # Step 4: Fast vertical formatting
            t_layout = time.time()
            bg_mode = config.get("background", "blur")
            bg_path = tmp / "background.mp4"

            print(f"[layout] cropping to 9:16 vertical (mode={bg_mode})...")
            if bg_mode in ("blur", "crop", "original"):
                crop_to_vertical(clean_path, bg_path)
            elif bg_mode in ("dark_gradient", "brand_color"):
                apply_background(clean_path, bg_path, bg_mode)
            else:
                crop_to_vertical(clean_path, bg_path)
            print(f"[layout] vertical formatting complete in {time.time() - t_layout:.1f}s")

            # Step 5: Apply color grade (only if not natural/none)
            grade = config.get("color_grade")
            if grade and grade != "natural":
                t_grade = time.time()
                graded_path = tmp / "graded.mp4"
                print(f"[color] applying color grade ({grade})...")
                apply_color_grade(
                    bg_path,
                    graded_path,
                    grade,
                )
                print(f"[color] color grade applied in {time.time() - t_grade:.1f}s")
            else:
                graded_path = bg_path

            t_pace = time.time()
            paced_path = tmp / "paced.mp4"
            print(f"[pacing] applying {len(director_plan['camera_punch_ins'])} camera punch-ins...")
            apply_camera_punch_ins(graded_path, paced_path, clip_duration, director_plan["camera_punch_ins"])
            print(f"[pacing] camera punch-ins complete in {time.time() - t_pace:.1f}s")

            # Step 6: Generate and add professional captions
            t_cap = time.time()
            captioned_path = tmp / "captioned.mp4"
            print("[captions] burning studio-grade subtitles & hook card...")
            add_captions(
                paced_path,
                captioned_path,
                config,
                hook_text=req.hook_text,
                caption_words=caption_data,
                director_plan=director_plan,
            )
            print(f"[captions] subtitles & hook burned in {time.time() - t_cap:.1f}s")

            # Step 7: Select and mix background music
            t_audio = time.time()
            final_path = tmp / "final.mp4"
            music = director_plan["bg_music"]
            mood = req.music_mood or music["mood"] or config.get("music_mood")
            if mood and mood != "none":
                print(f"[audio] mixing background music ({mood}) with speech ducking...")
                mix_mastered_audio(captioned_path, final_path, mood, music["base_volume_db"], music["duck_during_speech_db"])
            else:
                normalize_dialogue(captioned_path, final_path)
            print(f"[audio] music mastering complete in {time.time() - t_audio:.1f}s")

        # SFX is part of every output path
        t_sfx = time.time()
        sfx_path = tmp / "sfx.mp4"
        sfx_events = (
            [event.model_dump() for event in req.sfx_events]
            if req.sfx_events
            else director_plan["sfx_cues"] or auto_sfx_for_emotion(req.emotion_type or "excited", clip_duration)
        )
        print(f"[sfx] mixing {len(sfx_events)} sound effects into audio track...")
        add_sfx(final_path, sfx_path, sfx_events)
        final_path = sfx_path
        print(f"[sfx] sound effects mixed in {time.time() - t_sfx:.1f}s")

        # Generate thumbnail
        thumbnail_path = tmp / "thumbnail.jpg"
        extract_thumbnail(final_path, thumbnail_path)

        # Upload both to Supabase
        t_upload = time.time()
        print("[upload] uploading final short & thumbnail to Supabase...")
        video_key = f"processed-clips/{req.user_id}/{req.clip_id}.mp4"
        thumb_key = f"thumbnails/{req.user_id}/{req.clip_id}.jpg"

        video_url = await upload_to_supabase(
            final_path,
            "videos",
            video_key,
            "video/mp4",
        )

        thumb_url = await upload_to_supabase(
            thumbnail_path,
            "videos",
            thumb_key,
            "image/jpeg",
        )
        print(f"[upload] uploaded to Supabase in {time.time() - t_upload:.1f}s")

        file_size = os.path.getsize(final_path)
        duration = get_video_duration(final_path)

        total_elapsed = time.time() - t_job_start
        print(f"[renderer] >>> Job #{req.job_id} completely rendered in {total_elapsed:.1f}s! ({file_size / (1024*1024):.2f} MB)")

        file_size = os.path.getsize(final_path)
        duration = get_video_duration(final_path)

        return {
            "job_id": req.job_id,
            "clip_id": req.clip_id,
            "jobId": req.job_id,
            "clipId": req.clip_id,
            "output_url": video_url,
            "outputUrl": video_url,
            "thumbnail_url": thumb_url,
            "thumbnailUrl": thumb_url,
            "duration": duration,
            "resolution": "1080x1920",
            "file_size_bytes": file_size,
            "fileSizeBytes": file_size,
            "style_applied": req.style,
            "styleApplied": req.style,
        }


def apply_caption_style(config: dict, caption_style: str) -> None:
    """Map editor labels to the existing caption service configuration."""
    styles = {
        "bold": {"caption_style": "white_bold_bottom", "font_size": 64, "text_color": "white", "highlight_color": "#FFEE00"},
        "karaoke": {"caption_style": "karaoke_yellow", "font_size": 64, "text_color": "#FFFF00", "highlight_color": "#FFEE00"},
        "minimal": {"caption_style": "small_white", "font_size": 40, "text_color": "white", "highlight_color": "#FFEE00"},
    }
    config.update(styles.get(caption_style, {}))

def probe_video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration_raw = result.stdout.strip()
    if not duration_raw:
        raise RuntimeError(f"unable to read video duration from {path}")
    return float(duration_raw)


def _normalize_clip_bounds(
    start: float,
    end: float,
    source_duration: float,
) -> tuple[float, float]:
    if end <= start:
        raise RuntimeError(
            f"clip end must be after clip start: start={start:.3f}s end={end:.3f}s"
        )

    if source_duration <= 0:
        raise RuntimeError(f"source video has invalid duration: {source_duration:.3f}s")

    requested_start = max(0.0, start)
    requested_end = min(end, source_duration)

    if requested_start >= source_duration:
        print(
            "[renderer] clip range is outside source duration; "
            f"falling back to full source start={start:.3f}s end={end:.3f}s "
            f"duration={source_duration:.3f}s"
        )
        return 0.0, source_duration

    if requested_end <= requested_start:
        print(
            "[renderer] clip end is outside source duration; "
            f"using remaining source from {requested_start:.3f}s to "
            f"{source_duration:.3f}s"
        )
        requested_end = source_duration

    return requested_start, requested_end


def extract_clip(input_path: Path, output_path: Path, start: float, end: float):
    source_duration = probe_video_duration(input_path)
    clip_start, clip_end = _normalize_clip_bounds(start, end, source_duration)
    clip_duration = clip_end - clip_start

    print(
        "[renderer] extracting clip "
        f"start={clip_start:.3f}s end={clip_end:.3f}s "
        f"duration={clip_duration:.3f}s"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-ss",
            str(clip_start),
            "-i",
            str(input_path),
            "-t",
            str(clip_duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )


def extract_thumbnail(video_path: Path, thumb_path: Path):
    # Extract frame at 10% into the video
    subprocess.run([
        "ffmpeg", "-y", "-nostdin",
        "-ss", "0",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(thumb_path)
    ], check=True, capture_output=True, timeout=30)


def get_video_duration(path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path)
    ], capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


async def download_file(url: str, dest: Path, start: float = 0, end: float | None = None) -> bool:
    if Path(url).is_file():
        shutil.copy2(url, dest)
        return False

    if _is_youtube_url(url):
        was_clipped = await asyncio.to_thread(download_youtube_video, url, dest, start, end)
        return was_clipped

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, follow_redirects=True, timeout=300.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"download produced an empty source file: {dest}")
    return False


def _is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "youtu.be" or host.endswith(".youtube.com") or host == "youtube.com"


def download_youtube_video(url: str, dest: Path, start: float = 0, end: float | None = None) -> bool:
    proxy = os.getenv("YOUTUBE_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    failures = []
    has_sections = end is not None and end > start

    # Use the venv yt-dlp binary with the latest JS challenge solver
    venv_yt_dlp = Path(sys.executable).parent / "yt-dlp"
    yt_bin = str(venv_yt_dlp) if venv_yt_dlp.is_file() else "yt-dlp"

    cookie_browser = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER", "brave")
    cookie_file = os.getenv("YOUTUBE_COOKIES_FILE")
    if not cookie_file and Path("cookies.txt").is_file():
        cookie_file = "cookies.txt"

    # Strategy 1: Browser cookies (default web client with ejs challenge solver)
    cookie_attempts = []
    if cookie_file and Path(cookie_file).is_file():
        cookie_attempts.append(["--cookies", str(cookie_file)])
    elif cookie_browser and cookie_browser.lower() != "none":
        cookie_attempts.append(["--cookies-from-browser", cookie_browser])
        if cookie_browser.lower() != "chrome":
            cookie_attempts.append(["--cookies-from-browser", "chrome"])
    cookie_attempts.append([])

    format_str = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"

    for c_args in cookie_attempts:
        if has_sections:
            args = [
                yt_bin,
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "--remote-components", "ejs:github",
                "--format", format_str,
                "--merge-output-format", "mp4",
                "--download-sections", f"*{max(0, start):.3f}-{end:.3f}",
                "--output", str(dest),
            ] + c_args
            if proxy:
                args.extend(["--proxy", proxy])
            dest.unlink(missing_ok=True)
            try:
                res = subprocess.run(args + [url], capture_output=True, text=True, timeout=90)
                if res.returncode == 0 and dest.is_file() and dest.stat().st_size >= 1024:
                    return True
                detail = (res.stderr or res.stdout or "no output").strip()
            except subprocess.TimeoutExpired:
                detail = "download section timed out after 90s"
            failures.append(f"cookie_section ({c_args[-1] if c_args else 'none'}): {detail[-250:]}")

        args_full = [
            yt_bin,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--remote-components", "ejs:github",
            "--format", format_str,
            "--merge-output-format", "mp4",
            "--output", str(dest),
        ] + c_args
        if proxy:
            args_full.extend(["--proxy", proxy])
        dest.unlink(missing_ok=True)
        try:
            res_full = subprocess.run(args_full + [url], capture_output=True, text=True, timeout=120)
            if res_full.returncode == 0 and dest.is_file() and dest.stat().st_size >= 1024:
                return False
            detail = (res_full.stderr or res_full.stdout or "no output").strip()
        except subprocess.TimeoutExpired:
            detail = "download full timed out after 120s"
        failures.append(f"cookie_full ({c_args[-1] if c_args else 'none'}): {detail[-250:]}")

    # Strategy 2: Fallback to multi-client player extractor args
    for player_client in ("mweb", "android", "web_safari", "tv_embedded"):
        if has_sections:
            args = [
                yt_bin,
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "--remote-components", "ejs:github",
                "--extractor-args", f"youtube:player_client={player_client}",
                "--format", format_str,
                "--merge-output-format", "mp4",
                "--download-sections", f"*{max(0, start):.3f}-{end:.3f}",
                "--output", str(dest),
            ]
            if proxy:
                args.extend(["--proxy", proxy])
            dest.unlink(missing_ok=True)
            try:
                res = subprocess.run(args + [url], capture_output=True, text=True, timeout=90)
                if res.returncode == 0 and dest.is_file() and dest.stat().st_size >= 1024:
                    return True
                detail = (res.stderr or res.stdout or "no output").strip()
            except subprocess.TimeoutExpired:
                detail = f"{player_client} section timed out after 90s"
            failures.append(f"{player_client} (section): {detail[-250:]}")

        args_full = [
            yt_bin,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--remote-components", "ejs:github",
            "--extractor-args", f"youtube:player_client={player_client}",
            "--format", format_str,
            "--merge-output-format", "mp4",
            "--output", str(dest),
        ]
        if proxy:
            args_full.extend(["--proxy", proxy])
        dest.unlink(missing_ok=True)
        try:
            res_full = subprocess.run(args_full + [url], capture_output=True, text=True, timeout=120)
            if res_full.returncode == 0 and dest.is_file() and dest.stat().st_size >= 1024:
                return False
            detail = (res_full.stderr or res_full.stdout or "no output").strip()
        except subprocess.TimeoutExpired:
            detail = f"{player_client} full timed out after 120s"
        failures.append(f"{player_client} (full): {detail[-250:]}")

    combined = " | ".join(failures).lower()
    if "private video" in combined or "video is private" in combined:
        raise RuntimeError("This YouTube video is private and cannot be processed.")
    if "this video is unavailable" in combined:
        raise RuntimeError("This YouTube video is unavailable or has been deleted.")
    if "sign in to confirm you're not a bot" in combined:
        raise RuntimeError("YouTube datacenter bot challenge encountered; configure YOUTUBE_PROXY or open YouTube in your browser.")

    raise RuntimeError("yt-dlp could not download YouTube video: " + " | ".join(failures))

async def upload_to_supabase(
    file_path: Path,
    bucket: str,
    object_key: str,
    content_type: str,
) -> str:
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{object_key}"

    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            resp = await client.post(
                url,
                content=f.read(),
                headers={
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "apikey": SUPABASE_KEY,
                    "Content-Type": content_type,
                    "x-upsert": "true",
                },
                timeout=300.0,
            )

        if resp.status_code >= 400:
            print("=== SUPABASE UPLOAD FAILED ===")
            print("Status:", resp.status_code)
            print("Response:", resp.text)   
            print("URL:", url)
            print("==============================")

        resp.raise_for_status()

    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{object_key}"
