import os
from dotenv import load_dotenv

load_dotenv()

import subprocess
import tempfile
import httpx
from pathlib import Path
from typing import Optional

from .cropper import crop_to_vertical
from .captions import add_captions
from .music import select_and_mix_music
from .backgrounds import apply_background
from .grader import apply_color_grade
from .cleaner import remove_silences_and_fillers

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

STYLE_CONFIGS = {
    "clean": {
        "background": "blur",
        "caption_style": "white_bold_bottom",
        "color_grade": "warm",
        "music_mood": "calm",
        "font_size": 52,
        "text_color": "white",
        "highlight_color": "#FFD700",
    },
    "bold": {
        "background": "dark_gradient",
        "caption_style": "karaoke_yellow",
        "color_grade": "vibrant",
        "music_mood": "energetic",
        "font_size": 64,
        "text_color": "#FFFF00",
        "highlight_color": "#FF4444",
    },
    "minimal": {
        "background": "original",
        "caption_style": "small_white",
        "color_grade": "natural",
        "music_mood": None,
        "font_size": 40,
        "text_color": "white",
        "highlight_color": "white",
    }
}


async def create_short(req) -> dict:
    config = STYLE_CONFIGS.get(req.style, STYLE_CONFIGS["clean"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Step 1: Download video from Supabase signed URL
        raw_path = tmp / "raw.mp4"
        await download_file(req.video_url, raw_path)

        # Step 2: Extract the clip segment
        clip_path = tmp / "clip.mp4"
        extract_clip(raw_path, clip_path, req.start_time, req.end_time)

        # Step 3: Remove silences and fillers if requested
        clean_path = tmp / "clean.mp4"
        if req.remove_silences or req.remove_fillers:
            remove_silences_and_fillers(clip_path, clean_path)
        else:
            clean_path = clip_path

        # Step 4: Crop to 9:16 vertical format
        vertical_path = tmp / "vertical.mp4"
        crop_to_vertical(clean_path, vertical_path)

        # Step 5: Apply background style
        bg_path = tmp / "background.mp4"
        apply_background(vertical_path, bg_path, config["background"])

        # Step 6: Apply color grade
        graded_path = tmp / "graded.mp4"
        apply_color_grade(bg_path, graded_path, config["color_grade"])

        # Step 7: Generate and add captions
        captioned_path = tmp / "captioned.mp4"
        await add_captions(graded_path, captioned_path, config, req.hook_text)

        # Step 8: Select and mix background music
        final_path = tmp / "final.mp4"
        mood = req.music_mood or config["music_mood"]
        if mood:
            select_and_mix_music(captioned_path, final_path, mood)
        else:
            final_path = captioned_path

        # Step 9: Generate thumbnail (best frame)
        thumbnail_path = tmp / "thumbnail.jpg"
        extract_thumbnail(final_path, thumbnail_path)

        # Step 10: Upload both to Supabase
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

        file_size = os.path.getsize(final_path)
        duration = get_video_duration(final_path)

        return {
            "job_id": req.job_id,
            "clip_id": req.clip_id,
            "output_url": video_url,
            "thumbnail_url": thumb_url,
            "duration": duration,
            "resolution": "1080x1920",
            "file_size_bytes": file_size,
            "style_applied": req.style,
        }


def extract_clip(input_path: Path, output_path: Path, start: float, end: float):
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(input_path),
        "-c", "copy",
        str(output_path)
    ], check=True, capture_output=True)


def extract_thumbnail(video_path: Path, thumb_path: Path):
    # Extract frame at 10% into the video
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(thumb_path)
    ], check=True, capture_output=True)


def get_video_duration(path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path)
    ], capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


async def download_file(url: str, dest: Path):
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, follow_redirects=True, timeout=300.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


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
