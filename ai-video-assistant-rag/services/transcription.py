from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st

from services.models import TranscriptSegment


ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpga"}


def persist_upload(uploaded_file, destination: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"upload{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def download_youtube(url: str, destination: Path) -> Path:
    """Download best available audio using yt-dlp. FFmpeg must be installed."""
    destination.mkdir(parents=True, exist_ok=True)
    template = str(destination / "source.%(ext)s")
    command = [
        "yt-dlp", "--no-playlist", "--js-runtimes", "node", "-f", "bestaudio/best", "-x", "--audio-format", "mp3",
        "-o", template, url,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Unable to download the YouTube media.")
    files = sorted(destination.glob("source.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError("yt-dlp finished but did not create an audio file.")
    return files[0]


@st.cache_resource(show_spinner=False)
def _load_whisper(model_name: str):
    import whisper

    return whisper.load_model(model_name)


def transcribe_whisper(media_path: Path, model_name: str) -> tuple[str, list[TranscriptSegment]]:
    model = _load_whisper(model_name)
    result = model.transcribe(str(media_path), fp16=False, language="en", verbose=False)
    segments = [
        TranscriptSegment(start=float(item["start"]), end=float(item["end"]), text=item["text"].strip())
        for item in result.get("segments", [])
    ]
    return result.get("text", "").strip(), segments


def transcribe_sarvam(media_path: Path, language_code: str) -> tuple[str, list[TranscriptSegment]]:
    """Submit media to Sarvam STT and accept the common response shapes.

    The endpoint is an environment setting to make provider-side API changes non-breaking.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is required for Hindi/Hinglish transcription.")
    endpoint = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
    with media_path.open("rb") as media:
        response = requests.post(
            endpoint,
            headers={"api-subscription-key": api_key},
            files={"file": (media_path.name, media)},
            data={"language_code": language_code, "model": "saarika:v2.5"},
            timeout=600,
        )
    if not response.ok:
        raise RuntimeError(f"Sarvam transcription failed ({response.status_code}): {response.text[:400]}")
    payload = response.json()
    transcript = payload.get("transcript") or payload.get("text") or payload.get("transcription") or ""
    raw_segments = payload.get("segments") or []
    segments = [
        TranscriptSegment(float(s.get("start", 0)), float(s.get("end", 0)), str(s.get("text", "")).strip())
        for s in raw_segments
        if s.get("text")
    ]
    if not transcript:
        raise RuntimeError(f"Sarvam returned no transcript. Response: {json.dumps(payload)[:400]}")
    return transcript.strip(), segments
