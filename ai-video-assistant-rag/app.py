from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.export import report_as_pdf, report_as_text
from services.llm import answer_question, create_report
from services.rag import build_collection, retrieve
from services.transcription import download_youtube, persist_upload, transcribe_sarvam, transcribe_whisper


load_dotenv()
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="AI Video Assistant", page_icon="🎥", layout="wide")
st.title("🎥 AI Video Assistant with RAG")
st.caption("Transcribe a meeting, extract what matters, then ask questions grounded in its transcript.")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("1. Add a meeting")
    source_type = st.radio("Input type", ["Upload audio/video", "YouTube URL"])
    language = st.selectbox("Meeting language", ["English (Whisper local)", "Hindi (Sarvam)", "Hinglish (Sarvam)"])
    whisper_model = st.selectbox("Whisper model", ["base", "small", "medium"], index=0)
    uploaded = None
    url = None
    if source_type == "Upload audio/video":
        uploaded = st.file_uploader("Audio or video", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm", "mpeg", "mpga"])
    else:
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    process = st.button("Process meeting", type="primary", use_container_width=True)
    st.divider()
    st.caption("Keys are read from `.env`; they are never stored in the app session.")

if process:
    if not uploaded and not url:
        st.error("Add an audio/video file or a YouTube URL first.")
    else:
        try:
            fingerprint = hashlib.sha256((uploaded.getvalue() if uploaded else url).encode() if isinstance(url, str) and not uploaded else uploaded.getvalue()).hexdigest()[:16]
            meeting_dir = DATA_DIR / fingerprint
            with st.status("Processing meeting…", expanded=True) as status:
                if uploaded:
                    st.write("Saving upload…")
                    media = persist_upload(uploaded, meeting_dir)
                    title = uploaded.name
                else:
                    st.write("Downloading audio from YouTube…")
                    media = download_youtube(url, meeting_dir)
                    title = "YouTube meeting"
                st.write("Transcribing…")
                if language.startswith("English"):
                    transcript, segments = transcribe_whisper(media, whisper_model)
                else:
                    transcript, segments = transcribe_sarvam(media, "hi-IN")
                if not transcript:
                    raise RuntimeError("No speech was detected in the supplied media.")
                st.write("Building local RAG index…")
                collection = build_collection(fingerprint, transcript, segments, meeting_dir / "chroma")
                st.write("Extracting meeting report…")
                report = create_report(transcript)
                status.update(label="Meeting ready", state="complete", expanded=False)
            st.session_state.meeting = {"id": fingerprint, "title": title, "transcript": transcript, "segments": segments, "report": report, "collection": collection}
            st.session_state.chat = []
            st.rerun()
        except Exception as exc:
            st.exception(exc)

meeting = st.session_state.get("meeting")
if not meeting:
    st.info("Upload a recording or provide a YouTube URL to begin.")
    st.stop()

report = meeting["report"]
st.header(meeting["title"])
tab_report, tab_chat, tab_transcript = st.tabs(["Meeting report", "Chat with meeting", "Transcript"])

with tab_report:
    left, right = st.columns(2)
    with left:
        st.subheader("Summary")
        for item in report.summary:
            st.markdown(f"- {item}")
        st.subheader("Decisions")
        for item in report.decisions:
            st.markdown(f"- {item}")
    with right:
        st.subheader("Action items")
        st.dataframe(report.action_items, use_container_width=True, hide_index=True)
        st.subheader("Open questions & follow-ups")
        for item in report.open_questions:
            st.markdown(f"- {item}")
    text_export = report_as_text(meeting["title"], report, meeting["transcript"])
    c1, c2 = st.columns(2)
    c1.download_button("Download TXT", text_export, "meeting-report.txt", "text/plain", use_container_width=True)
    c2.download_button("Download PDF", report_as_pdf(meeting["title"], report, meeting["transcript"]), "meeting-report.pdf", "application/pdf", use_container_width=True)

with tab_chat:
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if question := st.chat_input("Ask about decisions, owners, deadlines, or discussion points…"):
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript…"):
                context, sources = retrieve(meeting["collection"], question)
                answer = answer_question(question, context, st.session_state.chat)
                if sources:
                    labels = ", ".join(sorted({s.get("timestamp", "00:00") for s in sources}))
                    answer += f"\n\n*Retrieved transcript timestamps: {labels}*"
                st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

with tab_transcript:
    st.download_button("Download transcript", meeting["transcript"], "transcript.txt", "text/plain")
    if meeting["segments"]:
        st.dataframe([{"Time": segment.label, "Text": segment.text} for segment in meeting["segments"]], use_container_width=True, hide_index=True)
    else:
        st.text_area("Transcript", meeting["transcript"], height=550)
