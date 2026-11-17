"""High-speed AI audio transcription engine for short-form video.

Uses Groq whisper-large-v3 for ~0.5s cloud transcription with millisecond word timestamps.
Automatically rotates across multiple Groq API keys with graceful local fallback.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

from config import GROQ_WHISPER_MODEL, get_groq_keys

logger = logging.getLogger(__name__)


def extract_audio_for_transcription(video_path: Path, output_audio_path: Path) -> bool:
    """Extract lightweight, speech-optimized mono MP3 from a video segment."""
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(output_audio_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return res.returncode == 0 and output_audio_path.is_file() and output_audio_path.stat().st_size > 500
    except Exception as exc:
        logger.warning("Failed to extract audio track: %s", exc)
        return False


def transcribe_audio_groq(audio_path: Path) -> list[dict[str, Any]]:
    """Transcribe audio using Groq whisper-large-v3 with word-level timestamps.
    Rotates automatically across configured GROQ keys on rate limit (429) or error.
    """
    keys = get_groq_keys()
    if not keys:
        logger.warning("[transcription] No GROQ keys found in environment")
        return []

    if not audio_path.is_file() or audio_path.stat().st_size < 100:
        logger.warning("[transcription] Audio file missing or empty: %s", audio_path)
        return []

    audio_bytes = audio_path.read_bytes()

    for idx, key in enumerate(keys):
        masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "key"
        try:
            logger.info("[transcription] Calling Groq whisper-large-v3 with key %d (%s)...", idx + 1, masked_key)
            headers = {"Authorization": f"Bearer {key}"}
            files = {"file": ("audio.mp3", audio_bytes, "audio/mpeg")}
            data = {
                "model": GROQ_WHISPER_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            }

            with httpx.Client(timeout=25.0) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                )

            if resp.status_code == 429:
                logger.warning("[transcription] Groq key %d rate limited (429), trying next key...", idx + 1)
                continue

            if resp.status_code != 200:
                logger.warning(
                    "[transcription] Groq returned HTTP %d: %s, rotating key...",
                    resp.status_code,
                    resp.text[:200],
                )
                continue

            payload = resp.json()
            raw_words = payload.get("words", [])
            if not raw_words and payload.get("segments"):
                # Some API formats place words inside segments
                for seg in payload["segments"]:
                    raw_words.extend(seg.get("words", []))

            words = []
            for w in raw_words:
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                words.append({
                    "word": text,
                    "start": round(float(w.get("start", 0.0)), 3),
                    "end": round(float(w.get("end", 0.0)), 3),
                })

            logger.info("[transcription] Groq whisper transcribed %d words successfully", len(words))
            return words

        except Exception as exc:
            logger.warning("[transcription] Groq transcription attempt with key %d failed: %s", idx + 1, exc)
            continue

    logger.warning("[transcription] All Groq keys exhausted for whisper transcription")
    return []


_LOCAL_WHISPER_MODEL = None


def get_local_whisper_model():
    """Lazy load and cache open-source Whisper model in memory."""
    global _LOCAL_WHISPER_MODEL
    if _LOCAL_WHISPER_MODEL is None:
        import whisper

        logger.info("[transcription] Loading local open-source Whisper 'base' model into memory...")
        _LOCAL_WHISPER_MODEL = whisper.load_model("base")
        logger.info("[transcription] Local open-source Whisper model loaded successfully!")
    return _LOCAL_WHISPER_MODEL


def transcribe_audio_local(audio_path: Path) -> list[dict[str, Any]]:
    """Transcribe audio locally on Mac using open-source OpenAI Whisper.
    100% free, runs completely offline, zero API limits.
    """
    if not audio_path.is_file() or audio_path.stat().st_size < 100:
        logger.warning("[transcription] Local audio file missing or empty: %s", audio_path)
        return []

    try:
        model = get_local_whisper_model()
        logger.info("[transcription] Running local open-source Whisper transcription on CPU...")
        result = model.transcribe(str(audio_path), word_timestamps=True, fp16=False)

        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                words.append({
                    "word": text,
                    "start": round(float(w.get("start", 0.0)), 3),
                    "end": round(float(w.get("end", 0.0)), 3),
                })

        logger.info("[transcription] Local Whisper transcribed %d words successfully", len(words))
        return words
    except Exception as exc:
        logger.warning("[transcription] Local Whisper transcription failed: %s", exc)
        return []


def transcribe_audio(audio_path: Path) -> list[dict[str, Any]]:
    """Master hybrid transcription function.
    1. Runs Groq whisper-large-v3 first (ultra fast: ~0.7s, highest accuracy).
    2. If Groq encounters limits, quotas, or network issues, seamlessly
       falls back to local open-source Whisper on CPU (100% free, zero limits).
    """
    import os

    mode = os.getenv("TRANSCRIPTION_PROVIDER", "auto").lower()

    if mode == "local":
        logger.info("[transcription] Local-only mode active")
        return transcribe_audio_local(audio_path)

    # 1. Fast cloud transcription (0.7s)
    words = transcribe_audio_groq(audio_path)
    if words:
        return words

    # 2. Local fallback
    logger.info("[transcription] Groq unavailable or rate-limited -> falling back to local open-source Whisper")
    return transcribe_audio_local(audio_path)


def auto_transcribe_clip(video_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Extract audio from video segment and auto-transcribe with word timestamps.
    Returns (word_list, full_transcript).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
        tmp_audio_path = Path(tmp_audio.name)

    try:
        extracted = extract_audio_for_transcription(video_path, tmp_audio_path)
        if not extracted:
            return [], ""

        words = transcribe_audio(tmp_audio_path)
        transcript = " ".join(w["word"] for w in words).strip()
        return words, transcript
    finally:
        try:
            tmp_audio_path.unlink(missing_ok=True)
        except Exception:
            pass

