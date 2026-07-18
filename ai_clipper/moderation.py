"""
Lightweight content moderation pass for selected segments.
An LLM optimizing purely for engagement can surface outrage/conflict clips
that weren't actually wanted. This flags them before rendering.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModerationResult(BaseModel):
    flagged: bool = False
    reason: str = ""
    categories: List[str] = []
    severity: float = 0.0  # 0-1


MODERATION_PROMPT = """You are a content safety reviewer. Check this transcript segment for problematic content.

Review for:
1. Hate speech, slurs, or discriminatory language
2. Graphic violence or gore descriptions
3. Sexual content
4. Harassment or targeted attacks
5. Dangerous misinformation
6. Self-harm or suicide content

Also flag:
- Outrage-bait: content designed purely to make people angry
- Toxic conflict: interpersonal drama with no redeeming value
- Clickbait that misrepresents the source material

TRANSCRIPT:
{text}

Respond ONLY with a JSON object:
{{
  "flagged": <true or false>,
  "reason": "<brief explanation, empty if not flagged>",
  "categories": ["<list of triggered categories>"],
  "severity": <0-1 float: how severe, 0=none, 1=extreme>
}}"""


def moderate_segment(
    text: str,
    llm_call_fn,
    threshold: float = 0.5,
) -> ModerationResult:
    """
    Run a moderation pass on a transcript segment.

    Args:
        text: Transcript text to check
        llm_call_fn: Function that takes a prompt string and returns a response string
        threshold: Severity threshold to flag (0-1, lower = stricter)

    Returns:
        ModerationResult with flagged status
    """
    if not text.strip() or len(text) < 30:
        return ModerationResult(flagged=False, reason="Too short to check")

    prompt = MODERATION_PROMPT.format(text=text[:1500])

    try:
        response = llm_call_fn(prompt)
        parsed = _parse_moderation(response)

        if parsed.get("flagged", False) and parsed.get("severity", 0) >= threshold:
            return ModerationResult(
                flagged=True,
                reason=parsed.get("reason", "Content flagged by moderation"),
                categories=parsed.get("categories", []),
                severity=parsed.get("severity", 0),
            )
        return ModerationResult(flagged=False)

    except Exception as e:
        logger.warning(f"Moderation check failed: {e}. Allowing segment through.")
        return ModerationResult(flagged=False)


def _parse_moderation(text: str) -> dict:
    """Parse moderation JSON from LLM response."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return {}
