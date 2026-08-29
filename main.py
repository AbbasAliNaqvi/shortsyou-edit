import asyncio
import json
import logging
from typing import Optional

import httpx
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from services.renderer import create_short


load_dotenv()


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("shortsyou-edit")


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

    # Go backend callback endpoint.
    callback_url: str

    # Internal API key supplied by the Go backend.
    callback_key: str


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
async def create_short_endpoint(
    req: CreateShortRequest,
):
    logger.info(
        "Received render job | "
        "job_id=%s clip_id=%s user_id=%s style=%s",
        req.job_id,
        req.clip_id,
        req.user_id,
        req.style,
    )

    # Rendering happens in the background so that
    # the Go backend receives an immediate 200 response.
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

async def process_in_background(
    req: CreateShortRequest,
):
    try:
        logger.info(
            "Starting render | "
            "job_id=%s clip_id=%s",
            req.job_id,
            req.clip_id,
        )

        # -------------------------------------------------
        # Render video
        # -------------------------------------------------

        result = await create_short(req)

        logger.info(
            "Render completed | "
            "job_id=%s clip_id=%s",
            req.job_id,
            req.clip_id,
        )

        # -------------------------------------------------
        # Success callback
        # -------------------------------------------------

        await send_callback(
            req.callback_url,
            req.callback_key,
            result,
        )

        logger.info(
            "Success callback sent | "
            "job_id=%s clip_id=%s",
            req.job_id,
            req.clip_id,
        )

    except Exception as exc:
        logger.exception(
            "Render failed | "
            "job_id=%s clip_id=%s",
            req.job_id,
            req.clip_id,
        )

        # Always notify Go about failures.
        error_payload = {
            "job_id": req.job_id,
            "clipId": req.clip_id,
            "error": str(exc),
        }

        try:
            await send_callback(
                req.callback_url,
                req.callback_key,
                error_payload,
            )

            logger.info(
                "Error callback sent | "
                "job_id=%s clip_id=%s",
                req.job_id,
                req.clip_id,
            )

        except Exception:
            logger.exception(
                "Failed to send error callback | "
                "job_id=%s",
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
    """
    Send the renderer result back to the Go backend.

    IMPORTANT:
    The Go middleware expects:

        X-Internal-API-Key

    NOT:

        X-Internal-Key
    """

    if not callback_url:
        raise RuntimeError(
            "callback_url is empty"
        )

    if not callback_key:
        raise RuntimeError(
            "callback_key is empty"
        )

    logger.info(
        "Sending callback | "
        "url=%s job_id=%s clip_id=%s",
        callback_url,
        payload.get("job_id"),
        payload.get("clipId"),
    )

    headers = {
        "X-Internal-API-Key": callback_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(
        timeout=30.0,
    ) as client:

        response = await client.post(
            callback_url,
            json=payload,
            headers=headers,
        )

    if response.status_code >= 400:
        logger.error(
            "Callback failed | "
            "status=%s body=%s job_id=%s clip_id=%s",
            response.status_code,
            response.text,
            payload.get("job_id"),
            payload.get("clipId"),
        )

        response.raise_for_status()

    logger.info(
        "Callback successful | "
        "status=%s job_id=%s clip_id=%s",
        response.status_code,
        payload.get("job_id"),
        payload.get("clipId"),
    )


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