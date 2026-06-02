from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

PROMPT_GUIDE_BASE_URL = "https://developers.openai.com/api/docs/guides/prompt-guidance"
DEFAULT_PROMPT_GUIDE_MODEL = "gpt-4.1"
DEFAULT_TARGET_PROMPT_GUIDE_URL = f"{PROMPT_GUIDE_BASE_URL}?model={DEFAULT_PROMPT_GUIDE_MODEL}"
MAX_GUIDE_CHARS = 30000


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text and not self._skip_depth:
            self.parts.append(text)


class _PromptGuideSectionParser(HTMLParser):
    def __init__(self, guide_model: str) -> None:
        super().__init__(convert_charrefs=False)
        self.guide_model = guide_model
        self._capturing = False
        self._depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if not self._capturing and attrs_dict.get("data-prompting-guide-model") == self.guide_model:
            self._capturing = True
            self._depth = 1
            self.parts.append(self.get_starttag_text() or "")
            return

        if self._capturing:
            self._depth += 1
            self.parts.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        if not self._capturing:
            return
        self.parts.append(f"</{tag}>")
        self._depth -= 1
        if self._depth == 0:
            self._capturing = False

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._capturing:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._capturing:
            self.parts.append(f"&#{name};")


def fetch_prompt_guide(url: str = DEFAULT_TARGET_PROMPT_GUIDE_URL) -> str:
    request = Request(url, headers={"User-Agent": "aoai-prompt-port/0.1"})
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")

    guide_model = _guide_model_from_url(url)
    guide = _extract_prompt_guidance_section_from_html(html, guide_model=guide_model)
    if not guide:
        parser = _VisibleTextParser()
        parser.feed(html)
        text = "\n".join(parser.parts)
        guide = _extract_prompt_guidance_section(text, guide_model=guide_model)
    if not guide:
        raise ValueError(f"Could not extract prompt guidance from {url}")

    if len(guide) > MAX_GUIDE_CHARS:
        guide = guide[:MAX_GUIDE_CHARS].rsplit("\n", 1)[0].strip()
        guide += "\n\n[Truncated to keep GEPA reflection context concise.]"

    addendum = prompt_convention_addendum(guide_model)
    return f"Source: {url}\n\n{addendum}\n\n{guide}".strip()


def build_prompt_guide_url(model: str) -> str:
    guide_model = normalize_prompt_guide_model(model)
    return f"{PROMPT_GUIDE_BASE_URL}?{urlencode({'model': guide_model})}"


def _guide_model_from_url(url: str) -> str:
    values = parse_qs(urlparse(url).query).get("model")
    return values[0] if values else DEFAULT_PROMPT_GUIDE_MODEL


def prompt_convention_addendum(guide_model: str) -> str:
    if guide_model.startswith("gpt-5.4"):
        return (
            "Target convention addendum for GPT-5.4:\n"
            "- Prefer XML-style section tags for prompt scaffolding instead of Markdown headings.\n"
            "- Use tags such as <role>, <goal>, <personality>, <constraints>, "
            "<output_contract>, <decision_rules>, and <examples> when they clarify boundaries.\n"
            "- Keep the rewrite minimal, but convert section structure to tags when migrating from GPT-4o/4.1-style prompts."
        )
    return ""


def _extract_prompt_guidance_section_from_html(html: str, *, guide_model: str) -> str:
    parser = _PromptGuideSectionParser(guide_model)
    parser.feed(html)
    section_html = "".join(parser.parts).strip()
    if not section_html:
        return ""

    text_parser = _VisibleTextParser()
    text_parser.feed(section_html)
    return "\n".join(text_parser.parts).strip()


def normalize_prompt_guide_model(model: str) -> str:
    deployment = model.removeprefix("azure/").lower()
    match = re.search(r"gpt-(\d+(?:\.\d+)?|4o)(?:-[a-z]+)?(?:-\d{4}-\d{2}-\d{2})?", deployment)
    if not match:
        return DEFAULT_PROMPT_GUIDE_MODEL

    family = match.group(0)
    family = re.sub(r"-(?:mini|nano|turbo|preview)$", "", family)
    family = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", family)

    version = family.removeprefix("gpt-")
    if version == "4o":
        return DEFAULT_PROMPT_GUIDE_MODEL

    try:
        if float(version) < 4.1:
            return DEFAULT_PROMPT_GUIDE_MODEL
    except ValueError:
        return DEFAULT_PROMPT_GUIDE_MODEL

    return family


def _extract_prompt_guidance_section(text: str, *, guide_model: str = DEFAULT_PROMPT_GUIDE_MODEL) -> str:
    display_model = guide_model.replace("gpt-", "GPT-", 1)
    match = re.search(rf"{re.escape(display_model)}\s+prompting guide", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"GPT-[\w.]+ prompting guide", text)
    if not match:
        return ""

    section = text[match.start() :]
    next_section = re.search(r"\nGPT-[\w.]+(?:\s+Codex)?\s+prompting guide", section[len(match.group(0)) :])
    next_section_end = len(match.group(0)) + next_section.start() if next_section else len(section)
    next_section_markers = ["\nAppendix", "\nRelated guides", "\nWas this page useful?"]
    marker_positions = [section.find(marker) for marker in next_section_markers if section.find(marker) != -1]
    end = min([next_section_end, *marker_positions]) if marker_positions else next_section_end
    return section[:end].strip()
