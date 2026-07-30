from __future__ import annotations

import re


_BACKTICK_RUN = re.compile(r"`+")


def fence(body: str) -> str:
    longest = max((len(match.group()) for match in _BACKTICK_RUN.finditer(body)), default=0)
    return "`" * max(3, longest + 1)


def fenced_go(code: str) -> str:
    body = code if code.endswith("\n") else f"{code}\n"
    delimiter = fence(body)
    return f"{delimiter}go\n{body}{delimiter}"
