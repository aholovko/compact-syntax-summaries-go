from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum


class ExtractionStatus(str, Enum):
    GO_BLOCK = "go_block"
    FENCED_BLOCK = "fenced_block"
    LARGEST_PARSEABLE = "largest_parseable"
    FAILED = "failed"


_GO_FENCE = re.compile(
    r"(?P<fence>`{3,})[ \t]*go[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?(?P=fence)",
    re.DOTALL | re.IGNORECASE,
)
_ANY_FENCE = re.compile(
    r"(?P<fence>`{3,})[ \t]*\w*[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?(?P=fence)",
    re.DOTALL,
)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_FENCE_LINE = re.compile(r"^\s*`{3,}(?:\w+)?\s*$", re.MULTILINE)


def _has_code(text: str) -> bool:
    without_fences = _FENCE_LINE.sub("", text)
    without_comments = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", without_fences))
    return bool(without_comments.strip())


def extract(
    text: str,
    *,
    is_parseable: Callable[[str], bool],
) -> tuple[str | None, ExtractionStatus]:
    for match in _GO_FENCE.finditer(text):
        body = match.group("body")
        if _has_code(body):
            return body, ExtractionStatus.GO_BLOCK
    for match in _ANY_FENCE.finditer(text):
        body = match.group("body")
        if _has_code(body):
            return body, ExtractionStatus.FENCED_BLOCK
    code = _largest_parseable(text, is_parseable)
    if code is not None:
        return code, ExtractionStatus.LARGEST_PARSEABLE
    return None, ExtractionStatus.FAILED


def _largest_parseable(
    text: str,
    is_parseable: Callable[[str], bool],
    *,
    max_probes: int = 32,
) -> str | None:
    lines = text.splitlines()
    probes = 0
    for size in range(len(lines), 0, -1):
        for start in range(0, len(lines) - size + 1):
            if probes >= max_probes:
                return None
            probes += 1
            candidate = "\n".join(lines[start : start + size])
            if _has_code(candidate) and is_parseable(candidate):
                return candidate
    return None
