import asyncio
import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles


from services.renderer import create_short
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
        "Received render job | job_id=%s clip_id=%s style=%s",
        req.job_id,
        req.clip_id,
        req.style,
    )

    # Don't block the HTTP request while FFmpeg/Whisper runs.
    asyncio.create_task(
        process_in_background(req)
    )

    return CreateShortResponse(
        job_id=req.job_id,
        accepted=True,
        eta_seconds=120,
    )


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

    async with httpx.AsyncClient() as client:

        response = await client.post(
            callback_url,
            json=payload,
            headers={
                "X-Internal-API-Key": callback_key,
                "X-Internal-Key": callback_key,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    if response.status_code >= 400:
        logger.error(
            "Callback failed | status=%s body=%s payload=%s",
            response.status_code,
            response.text,
            json.dumps(payload, default=str),
    )

    response.raise_for_status()


app.mount("/test-assets", StaticFiles(directory="test-assets"), name="test-assets")

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
