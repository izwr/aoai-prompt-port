from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from prompt_migration.evaluation.types import ConversationMessage

DOCUMENT_HEADER = "Document Intelligence extraction:"


def resolve_image_data_url(source: str, *, base_dir: Path | None = None) -> str:
    """Resolve an image reference to something the chat API accepts.

    Accepts an existing ``data:`` URL, an ``http(s)`` URL, or a file path. File
    paths are read and inlined as a base64 ``data:`` URL. Relative paths resolve
    against ``base_dir`` (typically the golden file's directory).
    """
    text = source.strip()
    if text.startswith(("data:", "http://", "https://")):
        return text

    path = Path(text)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"Image not found: {path}")

    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def resolve_document_text(source: Any, *, base_dir: Path | None = None) -> str:
    """Resolve Azure Document Intelligence output to plain text.

    ``source`` may be a parsed object, an inline JSON/text string, or a path to a
    ``.json``/text file. Document Intelligence JSON is reduced to its extracted
    ``content`` text; anything else is returned verbatim (text) or pretty-printed
    (objects).
    """
    if isinstance(source, (dict, list)):
        return _document_text_from_payload(source)
    if not isinstance(source, str):
        raise ValueError("document_intelligence must be a string path/text or a JSON object.")

    text = source.strip()
    candidate = Path(text)
    if base_dir is not None and not candidate.is_absolute():
        candidate = base_dir / candidate

    if candidate.is_file():
        raw = candidate.read_text(encoding="utf-8")
        if candidate.suffix.lower() == ".json":
            try:
                return _document_text_from_payload(json.loads(raw))
            except json.JSONDecodeError:
                return raw.strip()
        return raw.strip()

    try:
        return _document_text_from_payload(json.loads(text))
    except json.JSONDecodeError:
        return text


def _document_text_from_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        analyze_result = payload.get("analyzeResult")
        if isinstance(analyze_result, dict):
            nested = analyze_result.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return json.dumps(payload, ensure_ascii=False, indent=2)


def message_api_content(message: ConversationMessage) -> str | list[dict[str, Any]]:
    """Build the chat-API ``content`` for a message.

    Plain text messages return a string (unchanged behavior). Messages carrying a
    document and/or images return a list of content parts: document text, then
    image parts, then the question text.
    """
    if not message.images and not message.document:
        return message.content

    parts: list[dict[str, Any]] = []
    if message.document:
        parts.append({"type": "text", "text": f"{DOCUMENT_HEADER}\n{message.document}"})
    for image_url in message.images:
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    if message.content:
        parts.append({"type": "text", "text": message.content})
    return parts


def message_display_text(message: ConversationMessage) -> str:
    """A base64-free text rendering of a message for reports and judging."""
    if not message.images and not message.document:
        return message.content

    segments: list[str] = []
    if message.document:
        segments.append(f"{DOCUMENT_HEADER}\n{message.document}")
    for index in range(1, len(message.images) + 1):
        segments.append(f"[image {index}]")
    if message.content:
        segments.append(message.content)
    return "\n\n".join(segments)


def message_text(content: str | list[dict[str, Any]] | None) -> str:
    """Extract concatenated text from a string or list-of-parts content."""
    if isinstance(content, list):
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(text for text in texts if text)
    return content or ""
