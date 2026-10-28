from __future__ import annotations

from html import escape
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from services.models import MeetingReport


def report_as_text(title: str, report: MeetingReport, transcript: str) -> str:
    lines = [title, "=" * len(title), "", "SUMMARY"]
    lines += [f"- {x}" for x in report.summary] or ["- None extracted"]
    lines += ["", "ACTION ITEMS"]
    lines += [f"- {x.get('task', '')} | Owner: {x.get('owner', 'Unknown')} | Deadline: {x.get('deadline', 'Not stated')}" for x in report.action_items] or ["- None extracted"]
    lines += ["", "DECISIONS"] + [f"- {x}" for x in report.decisions]
    lines += ["", "OPEN QUESTIONS & FOLLOW-UPS"] + [f"- {x}" for x in report.open_questions]
    lines += ["", "FULL TRANSCRIPT", transcript]
    return "\n".join(lines)


def report_as_pdf(title: str, report: MeetingReport, transcript: str) -> bytes:
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"])]
    sections = [("Summary", report.summary), ("Action items", [f"{x.get('task', '')} — Owner: {x.get('owner', 'Unknown')}; Deadline: {x.get('deadline', 'Not stated')}" for x in report.action_items]), ("Decisions", report.decisions), ("Open questions & follow-ups", report.open_questions), ("Full transcript", [transcript])]
    for heading, entries in sections:
        story += [Spacer(1, 10), Paragraph(heading, styles["Heading2"])]
        for entry in entries or ["None extracted."]:
            story.append(Paragraph(escape(str(entry)).replace("\n", "<br/>"), styles["BodyText"]))
    doc.build(story)
    return output.getvalue()
