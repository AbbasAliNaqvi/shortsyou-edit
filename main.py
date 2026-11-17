import asyncio
import json
import logging
import os
import tempfile
from urllib.parse import urlparse
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles


from services.renderer import create_short
from services.audio import extract_youtube_audio
from services.renderer import upload_to_supabase
# from services.preview import create_preview


from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("shortsyou-edit")

# FFmpeg and the media helpers are synchronous, CPU-heavy operations. Keep one
# render active at a time, but run it away from Uvicorn's event loop so new
# create-short requests can still be accepted immediately.
render_slots = asyncio.Semaphore(1)


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

app = FastAPI(
    title="ShortsYou Edit Service",
    description="Video editing and rendering service for ShortsYou",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SFXEvent(BaseModel):
    type:      str
    at_second: float

class CaptionWord(BaseModel):
    word: str
    start: float
    end: float

class ExtractAudioRequest(BaseModel):
    video_url: str
    user_id: str
    video_id: str

class ExtractAudioResponse(BaseModel):
    audio_url: str
    bytes: int

class TranscribeRequest(BaseModel):
    job_id: Optional[str] = None
    videoId: Optional[str] = None
    video_id: Optional[str] = None
    userId: Optional[str] = None
    user_id: Optional[str] = None
    audioUrl: Optional[str] = None
    audio_url: Optional[str] = None
    videoUrl: Optional[str] = None
    video_url: Optional[str] = None
    language: Optional[str] = "en"
    callbackUrl: Optional[str] = None
    callback_url: Optional[str] = None
    internalKey: Optional[str] = None
    internal_key: Optional[str] = None

class TranscribeResponse(BaseModel):
    job_id: str
    accepted: bool

class AnalyzeSegment(BaseModel):
    index: int
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    words: list[dict] = []

class AnalyzeRequest(BaseModel):
    job_id: Optional[str] = None
    video_id: Optional[str] = None
    videoId: Optional[str] = None
    user_id: Optional[str] = None
    userId: Optional[str] = None
    language: Optional[str] = "en"
    segments: list[AnalyzeSegment] = []
    fillerWords: list[dict] = []
    silenceGaps: list[dict] = []
    callbackUrl: Optional[str] = None
    callback_url: Optional[str] = None
    internalKey: Optional[str] = None
    internal_key: Optional[str] = None

class AnalyzeResponse(BaseModel):
    job_id: str
    accepted: bool

# ---------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------

class CreateShortRequest(BaseModel):
    job_id: str
    clip_id: str
    user_id: str

    video_url: str

    start_time: float
    end_time: float

    style: str = "clean"

    hook_text: Optional[str] = None
    # These timestamps must be supplied by ShortsYou_Server's transcript.
    # Never infer them locally: rendering has no ASR fallback.
    caption_words: list[CaptionWord] = []
    music_mood: Optional[str] = None

    remove_silences: bool = True
    remove_fillers: bool = True

    callback_url: str
    callback_key: str

    layout: str = "standard"
    background_style: Optional[str] = None
    color_grade: Optional[str] = None
    caption_style: Optional[str] = None

    emotion_type:    str   = "excited"
    sfx_events:      list[SFXEvent] = []


class CreateShortResponse(BaseModel):
    job_id: str
    accepted: bool
    eta_seconds: int


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "shortsyou-edit",
        "version": "1.0.0",
    }


# ---------------------------------------------------------
# Create Short
# ---------------------------------------------------------

@app.post(
    "/create-short",
    response_model=CreateShortResponse,
)
async def create_short_endpoint(req: CreateShortRequest):

    logger.info(
        "Received render job | job_id=%s clip_id=%s style=%s caption_source=%s caption_words=%d",
        req.job_id,
        req.clip_id,
        req.style,
        "transcription_service" if req.caption_words else "none",
        len(req.caption_words),
    )

    # Don't block the HTTP request while FFmpeg runs.
    asyncio.create_task(
        process_in_background(req)
    )

    return CreateShortResponse(
        job_id=req.job_id,
        accepted=True,
        eta_seconds=120,
    )


@app.post("/extract-audio", response_model=ExtractAudioResponse)
async def extract_audio_endpoint(
    req: ExtractAudioRequest,
    internal_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
):
    """Fetch and publish audio from the media worker for the ASR service."""
    expected_key = os.getenv("INTERNAL_API_KEY", "")
    if expected_key and internal_key != expected_key:
        raise HTTPException(status_code=401, detail="invalid internal service key")
    logger.info("Extracting transcription audio | video_id=%s", req.video_id)
    with tempfile.TemporaryDirectory(prefix="shortsyou-audio-") as directory:
        audio_path = Path(directory) / "audio.m4a"
        try:
            await extract_youtube_audio(req.video_url, audio_path)
        except RuntimeError as exc:
            # Downloader failures are expected external-source failures, not
            # internal server crashes. Returning 422 lets the Go worker report
            # the real cause and avoids an opaque FastAPI 500.
            logger.warning("Audio extraction failed | video_id=%s reason=%s", req.video_id, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        size = audio_path.stat().st_size
        if size > 45 << 20:
            raise ValueError("audio track exceeds the 45 MB transcription upload limit")
        audio_url = await upload_to_supabase(
            audio_path,
            "videos",
            f"transcription-audio/{req.user_id}/{req.video_id}.m4a",
            "audio/mp4",
        )
    logger.info("Transcription audio ready | video_id=%s bytes=%d", req.video_id, size)
    return ExtractAudioResponse(audio_url=audio_url, bytes=size)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(req: TranscribeRequest):
    job_id = req.job_id or req.videoId or req.video_id or "job"
    v_id = req.videoId or req.video_id or ""
    logger.info("Received transcription job | job_id=%s video_id=%s", job_id, v_id)
    asyncio.create_task(process_transcription_background(req))
    return TranscribeResponse(job_id=job_id, accepted=True)


async def process_transcription_background(req: TranscribeRequest):
    video_id = req.videoId or req.video_id or ""
    user_id = req.userId or req.user_id or ""
    callback_url = req.callbackUrl or req.callback_url or ""
    if not callback_url:
        default_base = os.getenv("BACKEND_URL", "https://shortsyou-server.onrender.com").rstrip("/")
        callback_url = f"{default_base}/api/internal/transcription/done"
    internal_key = req.internalKey or req.internal_key or os.getenv("INTERNAL_API_KEY", "")
    audio_url = req.audioUrl or req.audio_url
    video_url = req.videoUrl or req.video_url

    logger.info("Starting background transcription | video_id=%s", video_id)
    try:
        with tempfile.TemporaryDirectory(prefix="transcribe-") as td:
            tmp_dir = Path(td)
            audio_file = tmp_dir / "audio.mp3"

            # Determine if input is a YouTube URL or direct audio file
            yt_url = None
            for u in (audio_url, video_url):
                if u and ("youtube.com" in str(u).lower() or "youtu.be" in str(u).lower()):
                    yt_url = u
                    break

            if yt_url:
                logger.info("Extracting audio from YouTube for transcription: %s", yt_url)
                raw_audio = tmp_dir / "yt_audio.mp3"
                await extract_youtube_audio(yt_url, raw_audio)
                if not raw_audio.is_file() or raw_audio.stat().st_size < 1000:
                    raise RuntimeError("Failed to extract valid audio from YouTube URL")
                audio_file = raw_audio
            elif audio_url:
                logger.info("Downloading audio from direct URL: %s", audio_url)
                async with httpx.AsyncClient() as client:
                    resp = await client.get(audio_url, follow_redirects=True, timeout=180.0)
                    resp.raise_for_status()
                    downloaded = tmp_dir / "downloaded.bin"
                    downloaded.write_bytes(resp.content)
                    from services.transcription import extract_audio_for_transcription
                    extract_audio_for_transcription(downloaded, audio_file)
            elif video_url:
                logger.info("Extracting audio from video: %s", video_url)
                raw_audio = tmp_dir / "video_audio.mp3"
                await extract_youtube_audio(video_url, raw_audio)
                audio_file = raw_audio
            else:
                raise ValueError("Neither audioUrl nor videoUrl provided for transcription")

            from services.transcription import transcribe_audio
            words = transcribe_audio(audio_file)

            segments = []
            if words:
                cur_words = []
                cur_start = words[0]["start"]
                for i, w in enumerate(words):
                    cur_words.append({
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "probability": 0.99,
                    })
                    dur = cur_words[-1]["end"] - cur_start
                    is_last = (i == len(words) - 1)
                    is_punct = str(w["word"]).rstrip().endswith((".", "!", "?", "।", "॥"))
                    is_gap = (not is_last and (words[i + 1]["start"] - w["end"] > 1.0))

                    # Target viral Short duration: 22s to 45s.
                    # Break when sentence ends or gap occurs after at least 22s,
                    # or force break if duration reaches 45s, or on the last word.
                    should_split = is_last or (dur >= 22.0 and (is_punct or is_gap)) or (dur >= 45.0)

                    if should_split:
                        seg_text = " ".join(cw["word"] for cw in cur_words).strip()
                        if seg_text:
                            segments.append({
                                "index": len(segments),
                                "start": cur_start,
                                "end": cur_words[-1]["end"],
                                "text": seg_text,
                                "words": cur_words,
                            })
                        if not is_last:
                            cur_words = []
                            cur_start = words[i + 1]["start"]

            payload = {
                "videoId": video_id,
                "userId": user_id,
                "segments": segments,
                "fillerWords": [],
                "silenceGaps": [],
                "language": req.language or "en",
            }

            if callback_url and callback_url.startswith(("http://", "https://")):
                headers = {
                    "X-Internal-API-Key": internal_key,
                    "X-Internal-Key": internal_key,
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient() as client:
                    res = await client.post(callback_url, json=payload, headers=headers, timeout=30.0)
                    res.raise_for_status()
                    logger.info(
                        "Transcription callback delivered successfully | video_id=%s segments=%d words=%d",
                        video_id,
                        len(segments),
                        len(words),
                    )
            else:
                logger.warning("No valid http/https callback URL provided for transcription; skipping callback")

    except Exception as exc:
        logger.exception("Transcription processing failed | video_id=%s: %s", video_id, exc)
        if callback_url and callback_url.startswith(("http://", "https://")):
            error_payload = {
                "videoId": video_id,
                "userId": user_id,
                "segments": [],
                "fillerWords": [],
                "silenceGaps": [],
                "language": req.language or "en",
                "error": str(exc),
            }
            try:
                headers = {
                    "X-Internal-API-Key": internal_key,
                    "X-Internal-Key": internal_key,
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient() as client:
                    await client.post(callback_url, json=error_payload, headers=headers, timeout=15.0)
            except Exception:
                pass


# ---------------------------------------------------------
# Analyze (Viral Clip Detection & Hook Generation)
# ---------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(req: AnalyzeRequest):
    job_id = req.job_id or req.video_id or req.videoId or "analysis_job"
    logger.info("Received analyze job | job_id=%s segments=%d", job_id, len(req.segments))
    asyncio.create_task(process_analysis_background(req))
    return AnalyzeResponse(job_id=job_id, accepted=True)


async def process_analysis_background(req: AnalyzeRequest):
    video_id = req.videoId or req.video_id or req.job_id or ""
    user_id = req.userId or req.user_id or ""
    callback_url = req.callbackUrl or req.callback_url or ""
    if not callback_url:
        default_base = os.getenv("BACKEND_URL", "https://shortsyou-server.onrender.com").rstrip("/")
        callback_url = f"{default_base}/api/internal/analysis/done"
    internal_key = req.internalKey or req.internal_key or os.getenv("INTERNAL_API_KEY", "")

    logger.info("Starting background analysis | video_id=%s segments=%d callback=%s", video_id, len(req.segments), callback_url)
    try:
        from services.ai_director import analyze_transcript_segments

        raw_segs = [s.model_dump() for s in req.segments]
        scored_segs, topic_dist = await asyncio.to_thread(
            lambda: analyze_transcript_segments(raw_segs, req.language or "en")
        )

        payload = {
            "videoId": video_id,
            "userId": user_id,
            "segments": scored_segs,
            "topicDistribution": topic_dist,
            "error": "",
        }

        if callback_url and callback_url.startswith(("http://", "https://")):
            headers = {
                "X-Internal-API-Key": internal_key,
                "X-Internal-Key": internal_key,
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient() as client:
                res = await client.post(callback_url, json=payload, headers=headers, timeout=30.0)
                res.raise_for_status()
                logger.info(
                    "Analysis callback delivered successfully | video_id=%s scored_segments=%d",
                    video_id,
                    len(scored_segs),
                )
        else:
            logger.warning("No valid http/https callback URL provided for analysis; skipping callback")
    except Exception as exc:
        logger.exception("Analysis processing failed | video_id=%s: %s", video_id, exc)
        if callback_url and callback_url.startswith(("http://", "https://")):
            error_payload = {
                "videoId": video_id,
                "userId": user_id,
                "segments": [],
                "topicDistribution": [],
                "error": str(exc),
            }
            try:
                headers = {
                    "X-Internal-API-Key": internal_key,
                    "X-Internal-Key": internal_key,
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient() as client:
                    await client.post(callback_url, json=error_payload, headers=headers, timeout=15.0)
            except Exception:
                pass


# ---------------------------------------------------------
# Background Processing
# ---------------------------------------------------------

async def process_in_background(req: CreateShortRequest):

    try:

        logger.info(
            "Starting render | job_id=%s clip_id=%s",
            req.job_id,
            req.clip_id,
        )

        async with render_slots:
            result = await asyncio.to_thread(
                lambda: asyncio.run(create_short(req))
            )

        logger.info(
            "Render completed | job_id=%s clip_id=%s",
            req.job_id,
            req.clip_id,
        )

        await send_callback(
            callback_url=req.callback_url,
            callback_key=req.callback_key,
            payload=result,
        )

    except Exception as exc:

        logger.exception(
            "Render failed | job_id=%s clip_id=%s",
            req.job_id,
            req.clip_id,
        )

        error_payload = {
            "job_id": req.job_id,
            "clip_id": req.clip_id,
            "jobId": req.job_id,
            "clipId": req.clip_id,
            "error": str(exc),
        }

        try:
            await send_callback(
                callback_url=req.callback_url,
                callback_key=req.callback_key,
                payload=error_payload,
            )

        except Exception:
            logger.exception(
                "Failed to send error callback | job_id=%s",
                req.job_id,
            )


# ---------------------------------------------------------
# Callback
# ---------------------------------------------------------

async def send_callback(
    callback_url: str,
    callback_key: str,
    payload: dict,
):

    local_fallback = "http://127.0.0.1:5050/api/internal/short/done"
    is_ngrok = "ngrok" in (urlparse(callback_url).hostname or "").lower()
    headers = {"X-Internal-API-Key": callback_key, "X-Internal-Key": callback_key, "Content-Type": "application/json"}
    urls = [callback_url]

    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            for url in list(urls):
                try:
                    response = await client.post(url, json=payload, headers=headers, timeout=15.0)
                    if response.status_code == 404 and is_ngrok and local_fallback not in urls:
                        logger.warning("ngrok callback returned 404; falling back to local Go server")
                        urls.append(local_fallback)
                        continue
                    response.raise_for_status()
                    return
                except httpx.HTTPError as exc:
                    logger.warning("Callback attempt %d/3 failed for %s: %s", attempt + 1, url, exc)
            if attempt < 2:
                await asyncio.sleep(.5 * (2 ** attempt))
    raise RuntimeError(f"callback failed after 3 attempts: {urls}")


test_assets_dir = Path(__file__).parent / "test-assets"
if test_assets_dir.is_dir():
    app.mount("/test-assets", StaticFiles(directory=str(test_assets_dir)), name="test-assets")

# ---------------------------------------------------------
# Development Server
# ---------------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
