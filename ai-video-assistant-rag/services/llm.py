from __future__ import annotations

import json
import os
import re
from typing import Iterable

from mistralai.client import Mistral
from langchain_core.runnables import RunnableLambda

from services.models import MeetingReport


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(item, "text", None) or item.get("text", "") for item in content)
    return str(content or "")


def ask_mistral(system: str, user: str, temperature: float = 0.2) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is required for reports and chat.")
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return _content_to_text(response.choices[0].message.content).strip()


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The model did not return JSON.")
    return json.loads(match.group())


def create_report(transcript: str) -> MeetingReport:
    # LCEL keeps the prompt -> model -> structured output stages composable and testable.
    def build_prompt(text: str) -> str:
        return """Analyze the following meeting transcript. Return ONLY valid JSON with this exact schema:
{
  "summary": ["concise factual bullet"],
  "action_items": [{"task": "", "owner": "Unknown if not stated", "deadline": "Not stated if not stated"}],
  "decisions": ["decision"],
  "open_questions": ["question or follow-up"]
}
Do not invent names, deadlines, decisions, or answers. Include empty arrays when no evidence exists.

TRANSCRIPT:
""" + text[:100_000]

    report_chain = (
        RunnableLambda(build_prompt)
        | RunnableLambda(lambda prompt: ask_mistral("You are a precise meeting analyst.", prompt))
        | RunnableLambda(_extract_json)
    )
    data = report_chain.invoke(transcript)
    return MeetingReport(
        summary=[str(x) for x in data.get("summary", [])],
        action_items=[{str(k): str(v) for k, v in x.items()} for x in data.get("action_items", []) if isinstance(x, dict)],
        decisions=[str(x) for x in data.get("decisions", [])],
        open_questions=[str(x) for x in data.get("open_questions", [])],
    )


def answer_question(question: str, context: str, chat_history: Iterable[dict[str, str]]) -> str:
    history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in list(chat_history)[-6:])
    prompt = f"""Answer the question only from the transcript excerpts below. If the answer is not supported,
say that it was not discussed in the provided excerpts. Be concise and mention timestamps when relevant.

EXCERPTS:
{context}

RECENT CHAT:
{history}

QUESTION: {question}"""
    return ask_mistral("You are a grounded meeting assistant. Never invent meeting facts.", prompt)
