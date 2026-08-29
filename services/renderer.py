import json
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .cropper import crop_to_vertical
from .captions import add_captions
from .music import select_and_mix_music
from .backgrounds import apply_background
from .grader import apply_color_grade
from .cleaner import remove_silences_and_fillers


load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

RAW_VIDEO_BUCKET = "raw-videos"
PROCESSED_CLIPS_BUCKET = "processed-clips"


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
    },
}


async def create_short(req) -> dict:
    config = STYLE_CONFIGS.get(req.style, STYLE_CONFIGS["clean"])

    validate_environment()

    with tempfile.TemporaryDirectory(prefix="shortsyou-render-") as tmp_dir:
        tmp = Path(tmp_dir)

        print(
            f"[renderer] starting job={req.job_id} "
            f"clip={req.clip_id} user={req.user_id}"
        )

        raw_path = tmp / "raw.mp4"

        print(
            f"[renderer] downloading source video "
            f"clip={req.clip_id}"
        )

        await download_file(req.video_url, raw_path)

        print(
            f"[renderer] source downloaded "
            f"bytes={raw_path.stat().st_size}"
        )

        validate_video_file(raw_path, "raw video")

        raw_info = get_video_info(raw_path)

        print(
            f"[renderer] raw video "
            f"codec={raw_info['codec_name']} "
            f"size={raw_info['width']}x{raw_info['height']} "
            f"duration={raw_info['duration']:.3f}s "
            f"audio={raw_info['has_audio']}"
        )

        clip_path = tmp / "clip.mp4"

        extract_clip(
            raw_path,
            clip_path,
            float(req.start_time),
            float(req.end_time),
        )

        validate_video_file(clip_path, "extracted clip")

        if req.remove_silences or req.remove_fillers:
            clean_path = tmp / "clean.mp4"

            remove_silences_and_fillers(
                clip_path,
                clean_path,
            )

            validate_video_file(clean_path, "cleaned clip")
        else:
            clean_path = clip_path

        vertical_path = tmp / "vertical.mp4"

        crop_to_vertical(
            clean_path,
            vertical_path,
        )

        validate_video_file(vertical_path, "vertical video")

        bg_path = tmp / "background.mp4"

        apply_background(
            vertical_path,
            bg_path,
            config["background"],
        )

        validate_video_file(bg_path, "background video")

        graded_path = tmp / "graded.mp4"

        apply_color_grade(
            bg_path,
            graded_path,
            config["color_grade"],
        )

        validate_video_file(graded_path, "graded video")

        captioned_path = tmp / "captioned.mp4"

        await add_captions(
            graded_path,
            captioned_path,
            config,
            req.hook_text,
        )

        validate_video_file(captioned_path, "captioned video")

        final_path = tmp / "final.mp4"

        mood = req.music_mood or config["music_mood"]

        if mood:
            select_and_mix_music(
                captioned_path,
                final_path,
                mood,
            )
        else:
            final_path = captioned_path

        validate_video_file(final_path, "final video")

        if not final_path.exists():
            raise RuntimeError(
                f"final render file does not exist: {final_path}"
            )

        file_size = final_path.stat().st_size

        if file_size <= 0:
            raise RuntimeError(
                f"final render file is empty: {final_path}"
            )

        duration = get_video_duration(final_path)

        if duration <= 0:
            raise RuntimeError(
                f"final render has invalid duration: {duration}"
            )

        final_info = get_video_info(final_path)

        thumbnail_path = tmp / "thumbnail.jpg"

        extract_thumbnail(
            final_path,
            thumbnail_path,
        )

        if not thumbnail_path.exists():
            raise RuntimeError(
                "thumbnail generation failed"
            )

        if thumbnail_path.stat().st_size <= 0:
            raise RuntimeError(
                "thumbnail file is empty"
            )

        video_key = (
            f"{req.user_id}/"
            f"{req.clip_id}.mp4"
        )

        print(
            f"[renderer] uploading rendered clip "
            f"bucket={PROCESSED_CLIPS_BUCKET} "
            f"object={video_key}"
        )

        video_url = await upload_to_supabase(
            file_path=final_path,
            bucket=PROCESSED_CLIPS_BUCKET,
            object_key=video_key,
            content_type="video/mp4",
        )

        thumbnail_key = (
            f"{req.user_id}/"
            f"{req.clip_id}.jpg"
        )

        print(
            f"[renderer] uploading thumbnail "
            f"bucket={PROCESSED_CLIPS_BUCKET} "
            f"object={thumbnail_key}"
        )

        thumbnail_url = await upload_to_supabase(
            file_path=thumbnail_path,
            bucket=PROCESSED_CLIPS_BUCKET,
            object_key=thumbnail_key,
            content_type="image/jpeg",
        )

        print(
            f"[renderer] render complete "
            f"clip={req.clip_id} "
            f"bytes={file_size} "
            f"duration={duration:.2f}s "
            f"resolution={final_info['width']}x{final_info['height']}"
        )

        return {
            "job_id": req.job_id,
            "clipId": req.clip_id,
            "outputUrl": video_url,
            "thumbnailUrl": thumbnail_url,
            "duration": duration,
            "resolution": f"{final_info['width']}x{final_info['height']}",
            "fileSizeBytes": file_size,
            "styleApplied": req.style,
        }


def validate_environment():
    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured"
        )

    if not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_KEY is not configured"
        )


def extract_clip(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
):
    if not input_path.exists():
        raise RuntimeError(
            f"source video does not exist: {input_path}"
        )

    if start < 0:
        raise ValueError(
            f"start time cannot be negative: {start}"
        )

    if end <= start:
        raise ValueError(
            f"end time must be greater than start time: "
            f"start={start} end={end}"
        )

    source_info = get_video_info(input_path)
    source_duration = source_info["duration"]

    if source_duration <= 0:
        raise RuntimeError(
            f"source video has invalid duration: {source_duration}"
        )

    if start >= source_duration:
        raise RuntimeError(
            f"clip start is outside source video: "
            f"start={start:.3f}s "
            f"duration={source_duration:.3f}s"
        )

    actual_end = min(end, source_duration)

    if actual_end <= start:
        raise RuntimeError(
            f"clip range is outside source video: "
            f"start={start:.3f}s "
            f"end={end:.3f}s "
            f"duration={source_duration:.3f}s"
        )

    actual_duration = actual_end - start

    if actual_duration < 0.05:
        raise RuntimeError(
            f"clip duration is too short: "
            f"{actual_duration:.3f}s"
        )

    print(
        f"[renderer] extracting clip "
        f"start={start:.3f}s "
        f"end={actual_end:.3f}s "
        f"duration={actual_duration:.3f}s "
        f"source_codec={source_info['codec_name']}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{actual_duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed while extracting clip:\n"
            f"{result.stderr[-5000:]}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"FFmpeg did not create clip: {output_path}"
        )

    if output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"FFmpeg created an empty clip: {output_path}"
        )


def extract_thumbnail(
    video_path: Path,
    thumb_path: Path,
):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "0",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(thumb_path),
        ],
        check=True,
        capture_output=True,
    )


def get_video_duration(path: Path) -> float:
    if not path.exists():
        raise RuntimeError(
            f"video does not exist: {path}"
        )

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
        capture_output=True,
        text=True,
        check=True,
    )

    value = result.stdout.strip()

    if not value:
        raise RuntimeError(
            f"ffprobe returned no duration for {path}"
        )

    try:
        duration = float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"invalid video duration returned by ffprobe: {value}"
        ) from exc

    if duration <= 0:
        raise RuntimeError(
            f"video has invalid duration {duration}: {path}"
        )

    return duration


def get_video_info(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"video does not exist: {path}"
        )

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ffprobe returned invalid JSON for {path}"
        ) from exc

    video_stream = None
    audio_stream = None

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")

        if codec_type == "video" and video_stream is None:
            video_stream = stream

        if codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError(
            f"video contains no video stream: {path}"
        )

    format_data = data.get("format", {})

    duration_value = format_data.get("duration")

    if duration_value is None:
        duration_value = video_stream.get("duration")

    if duration_value is None:
        raise RuntimeError(
            f"video contains no readable duration: {path}"
        )

    try:
        duration = float(duration_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid video duration: {duration_value}"
        ) from exc

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"video has invalid dimensions: "
            f"{width}x{height} path={path}"
        )

    return {
        "codec_name": video_stream.get("codec_name", ""),
        "width": width,
        "height": height,
        "pix_fmt": video_stream.get("pix_fmt", ""),
        "duration": duration,
        "has_audio": audio_stream is not None,
        "audio_codec": (
            audio_stream.get("codec_name", "")
            if audio_stream
            else None
        ),
    }


def validate_video_file(
    path: Path,
    label: str,
):
    if not path.exists():
        raise RuntimeError(
            f"{label} does not exist: {path}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"{label} is empty: {path}"
        )

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    print(
        f"[renderer] {label} ffprobe:\n"
        f"{result.stdout}"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed ffprobe:\n"
            f"{result.stderr}"
        )

    has_video = False

    for line in result.stdout.splitlines():
        if line.startswith("codec_type=video"):
            has_video = True
            break

    if not has_video:
        raise RuntimeError(
            f"{label} contains no video stream: {path}\n"
            f"ffprobe output:\n{result.stdout}\n"
            f"ffprobe error:\n{result.stderr}"
        )


async def download_file(
    url: str,
    dest: Path,
):
    if not url:
        raise RuntimeError(
            "video URL is empty"
        )

    print(
        f"[renderer] downloading URL "
        f"length={len(url)}"
    )

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=300.0,
    ) as client:
        async with client.stream(
            "GET",
            url,
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()

                raise RuntimeError(
                    f"video download failed: "
                    f"status={resp.status_code} "
                    f"body={body[:1000]!r}"
                )

            with open(dest, "wb") as output:
                async for chunk in resp.aiter_bytes(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        output.write(chunk)

    if not dest.exists():
        raise RuntimeError(
            f"download destination does not exist: {dest}"
        )

    if dest.stat().st_size == 0:
        raise RuntimeError(
            f"downloaded video is empty: {dest}"
        )


async def upload_to_supabase(
    file_path: Path,
    bucket: str,
    object_key: str,
    content_type: str,
) -> str:
    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured"
        )

    if not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_KEY is not configured"
        )

    if not file_path.exists():
        raise RuntimeError(
            f"upload file does not exist: {file_path}"
        )

    object_key = object_key.lstrip("/")

    url = (
        f"{SUPABASE_URL}"
        f"/storage/v1/object/"
        f"{bucket}/"
        f"{object_key}"
    )

    print(
        f"[renderer] Supabase upload "
        f"bucket={bucket} "
        f"object={object_key} "
        f"bytes={file_path.stat().st_size}"
    )

    async with httpx.AsyncClient(
        timeout=300.0,
    ) as client:
        with open(file_path, "rb") as f:
            file_data = f.read()

        response = await client.post(
            url,
            content=file_data,
            headers={
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "apikey": SUPABASE_KEY,
                "Content-Type": content_type,
                "x-upsert": "true",
            },
        )

    if response.status_code >= 400:
        print("=== SUPABASE UPLOAD FAILED ===")
        print("Status:", response.status_code)
        print("Response:", response.text)
        print("URL:", url)
        print("==============================")

        raise RuntimeError(
            f"Supabase upload failed: "
            f"status={response.status_code} "
            f"body={response.text}"
        )

    return (
        f"{SUPABASE_URL}"
        f"/storage/v1/object/"
        f"{bucket}/"
        f"{object_key}"
    )
