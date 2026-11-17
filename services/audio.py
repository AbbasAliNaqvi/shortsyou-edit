import asyncio
import os
import subprocess
import sys
from pathlib import Path


def _download_youtube_audio(url: str, destination: Path) -> None:
    proxy = os.getenv("YOUTUBE_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    venv_yt_dlp = Path(sys.executable).parent / "yt-dlp"
    yt_bin = str(venv_yt_dlp) if venv_yt_dlp.is_file() else "yt-dlp"

    cookie_browser = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER", "brave")
    cookie_file = os.getenv("YOUTUBE_COOKIES_FILE")
    if not cookie_file and Path("cookies.txt").is_file():
        cookie_file = "cookies.txt"

    cookie_attempts = []
    if cookie_file and Path(cookie_file).is_file():
        cookie_attempts.append(["--cookies", str(cookie_file)])
    elif cookie_browser and cookie_browser.lower() != "none":
        cookie_attempts.append(["--cookies-from-browser", cookie_browser])
        if cookie_browser.lower() != "chrome":
            cookie_attempts.append(["--cookies-from-browser", "chrome"])
    cookie_attempts.append([])

    for c_args in cookie_attempts:
        cmd = [
            yt_bin,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--format",
            "bestaudio[ext=m4a]/bestaudio/best",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "4",
            "-o",
            str(destination),
        ] + c_args
        if proxy:
            cmd.extend(["--proxy", proxy])
        destination.unlink(missing_ok=True)
        try:
            res = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=90)
            if res.returncode == 0 and destination.is_file() and destination.stat().st_size > 1024:
                return
        except subprocess.TimeoutExpired:
            pass

    for player_client in ("mweb", "android_vr", "ios", "tv", "web_safari"):
        cmd = [
            yt_bin,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--extractor-retries",
            "2",
            "--extractor-args",
            f"youtube:player_client={player_client}",
            "--format",
            "bestaudio[ext=m4a]/bestaudio/best",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "5",
            "-o",
            str(destination),
        ]
        if proxy:
            cmd.extend(["--proxy", proxy])
        destination.unlink(missing_ok=True)
        try:
            res = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=45)
            if res.returncode == 0 and destination.is_file() and destination.stat().st_size > 1024:
                return
        except subprocess.TimeoutExpired:
            pass

    raise RuntimeError("yt-dlp could not download YouTube audio")


async def extract_youtube_audio(url: str, destination: Path) -> None:
    await asyncio.to_thread(_download_youtube_audio, url, destination)
