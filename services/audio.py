"""Audio extraction for the transcription pipeline.

This runs beside the renderer so YouTube fetches use the media worker's egress
instead of the public API server's datacenter IP.
"""

import asyncio
import base64
import os
import subprocess
from pathlib import Path


def _download_youtube_audio(url: str, destination: Path) -> None:
    args = [
        "yt-dlp", "--no-playlist", "--quiet", "--no-warnings",
        "--format", "bestaudio[ext=m4a]/bestaudio",
        "--extract-audio", "--audio-format", "m4a", "--audio-quality", "5",
        "--output", str(destination),
    ]
    cookie_path: Path | None = None
    encoded_cookies = os.getenv("YOUTUBE_COOKIES_BASE64", "").strip()
    if encoded_cookies:
        try:
            cookie_path = destination.parent / "youtube-cookies.txt"
            cookie_path.write_bytes(base64.b64decode(encoded_cookies, validate=True))
            args.extend(["--cookies", str(cookie_path)])
        except ValueError as exc:
            raise RuntimeError("invalid YOUTUBE_COOKIES_BASE64") from exc
    try:
        result = subprocess.run(args + [url], capture_output=True, text=True)
    finally:
        if cookie_path:
            cookie_path.unlink(missing_ok=True)
    if result.returncode != 0 or not destination.is_file():
        detail = result.stderr.strip() or result.stdout.strip() or "yt-dlp produced no audio"
        raise RuntimeError(f"yt-dlp audio download failed: {detail}")


async def extract_youtube_audio(url: str, destination: Path) -> None:
    await asyncio.to_thread(_download_youtube_audio, url, destination)
