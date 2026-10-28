# AI Video Assistant with RAG

Turn a YouTube meeting or local audio/video file into a transcript, a meeting report, and a searchable chat experience.

## Features

- YouTube URLs and common audio/video uploads
- Local Whisper transcription for English
- Sarvam API adapter for Hindi and Hinglish
- Mistral-powered summary, action items, decisions, and open questions through a LangChain LCEL pipeline
- Local ChromaDB + HuggingFace embeddings retrieval
- Grounded meeting chat with source chunk references
- Downloadable TXT and PDF reports

## Setup

1. Install FFmpeg and Node.js LTS, and make both available on your `PATH`. FFmpeg is required by Whisper and YouTube audio extraction; Node.js lets yt-dlp complete YouTube's JavaScript verification.
2. Create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your `MISTRAL_API_KEY`. Add `SARVAM_API_KEY` only if you use Hindi/Hinglish.
4. Run the app:

   ```powershell
   streamlit run app.py
   ```

The first Whisper and embedding run downloads its selected model. Meeting data is stored locally in `data/` and is excluded from Git.

## Notes

- Choose a language explicitly for predictable transcription routing.
- The Sarvam endpoint is configurable through `SARVAM_STT_URL`, since providers may revise API routes or payload requirements.
- The assistant only answers from retrieved transcript context; if the information is absent, it says so.
