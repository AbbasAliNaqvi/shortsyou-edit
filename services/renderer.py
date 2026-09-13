import asyncio
import base64
import os
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
from .music import select_and_mix_music
from .backgrounds import apply_background
from .grader import apply_color_grade
from .cleaner import remove_silences_and_fillers
from services.sfx import add_sfx, auto_sfx_for_emotion


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
    # Explicit editor controls win over the style preset. This keeps the API
    # backwards compatible while allowing creators to mix a custom look.
    config = {**config}
    if getattr(req, "background_style", None):
        config["background"] = req.background_style
    if getattr(req, "color_grade", None):
        config["color_grade"] = req.color_grade
    if getattr(req, "caption_style", None):
        apply_caption_style(config, req.caption_style)

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

        if req.layout in {"two_frame", "multi_face", "auto_face"}:
            print(f"[renderer] adaptive face layout requested: {req.layout}")

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
                # Two-frame layout worked
                working_path = two_frame_path
            else:
                # Fall back to normal vertical crop
                working_path = fallback_path

            # Apply color grade
            graded_path = tmp / "graded.mp4"
            apply_color_grade(
                working_path,
                graded_path,
                config["color_grade"],
            )

            # Generate and add captions
            captioned_path = tmp / "captioned.mp4"
            # await add_captions(
            #     graded_path,
            #     captioned_path,
            #     config,
            #     req.hook_text,
            # )
            add_captions(
                graded_path,
                captioned_path,
                config,
                req.hook_text,
            )

            # Select and mix background music
            final_path = tmp / "final.mp4"
            mood = req.music_mood or config["music_mood"]

            if mood and mood != "none":
                select_and_mix_music(
                    captioned_path,
                    final_path,
                    mood,
                )
            else:
                final_path = captioned_path

        else:
            # Step 4: Crop to 9:16 vertical format
            vertical_path = tmp / "vertical.mp4"
            crop_to_vertical(clean_path, vertical_path)

            # Step 5: Apply background style
            bg_path = tmp / "background.mp4"
            apply_background(
                vertical_path,
                bg_path,
                config["background"],
            )

            # Step 6: Apply color grade
            graded_path = tmp / "graded.mp4"
            apply_color_grade(
                bg_path,
                graded_path,
                config["color_grade"],
            )

            # Step 7: Generate and add captions
            captioned_path = tmp / "captioned.mp4"
            add_captions(
                graded_path,
                captioned_path,
                config,
                req.hook_text,
            )

            # Step 8: Select and mix background music
            final_path = tmp / "final.mp4"
            mood = req.music_mood or config["music_mood"]

            if mood and mood != "none":
                select_and_mix_music(
                    captioned_path,
                    final_path,
                    mood,
                )
            else:
                final_path = captioned_path

        # SFX is part of every output path, including face layouts. Apply it
        # last so it is mixed with the final caption/music audio consistently.
        sfx_path = tmp / "sfx.mp4"
        sfx_events = (
            [event.model_dump() for event in req.sfx_events]
            if req.sfx_events
            else auto_sfx_for_emotion(req.emotion_type or "excited", req.end_time - req.start_time)
        )
        add_sfx(final_path, sfx_path, sfx_events)
        final_path = sfx_path


        # Generate thumbnail
        thumbnail_path = tmp / "thumbnail.jpg"
        extract_thumbnail(final_path, thumbnail_path)

        # Upload both to Supabase
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
        "bold": {"caption_style": "white_bold_bottom", "font_size": 64, "text_color": "white", "highlight_color": "#FFD700"},
        "karaoke": {"caption_style": "karaoke_yellow", "font_size": 64, "text_color": "#FFFF00", "highlight_color": "#FF4444"},
        "minimal": {"caption_style": "small_white", "font_size": 40, "text_color": "white", "highlight_color": "white"},
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
            "-ss",
            str(clip_start),
            "-i",
            str(input_path),
            "-t",
            str(clip_duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


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
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


async def download_file(url: str, dest: Path):
    if _is_youtube_url(url):
        # Raw source videos are intentionally not stored in Supabase: the
        # Free plan rejects uploads above 50 MB. Download them only for the
        # lifetime of this render, then upload the much smaller final short.
        await asyncio.to_thread(download_youtube_video, url, dest)
        return

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, follow_redirects=True, timeout=300.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


def _is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "youtu.be" or host.endswith(".youtube.com") or host == "youtube.com"


def download_youtube_video(url: str, dest: Path):
    args = [
        "yt-dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--format",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--output",
        str(dest),
    ]

    cookie_path = None
    encoded_cookies = os.getenv("YOUTUBE_COOKIES_BASE64", "").strip()
    if encoded_cookies:
        try:
            cookie_path = dest.parent / "youtube-cookies.txt"
            cookie_path.write_bytes(base64.b64decode(encoded_cookies, validate=True))
            args.extend(["--cookies", str(cookie_path)])
        except ValueError as exc:
            raise RuntimeError("invalid YOUTUBE_COOKIES_BASE64") from exc

    try:
        result = subprocess.run(args + [url], capture_output=True, text=True)
    finally:
        if cookie_path:
            cookie_path.unlink(missing_ok=True)

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown yt-dlp error"
        raise RuntimeError(f"yt-dlp download failed: {message}")
    if not dest.is_file():
        raise RuntimeError(f"yt-dlp did not create expected source file: {dest}")


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
