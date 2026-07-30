from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


_PACKAGE = re.compile(r"^\s*package\s+\w+", re.MULTILINE)
_TOP_LEVEL_DECL = re.compile(r"^\s*(func|type|var|const|import)\b")


@dataclass(frozen=True)
class Fixture:
    source: str
    strategy: Literal["file", "decl", "stmt"]


def synthesize(code: str) -> Fixture:
    """Apply one fixture policy to originals, references, and generated outputs."""
    if _PACKAGE.search(code):
        return Fixture(source=code, strategy="file")
    first = next((line for line in code.splitlines() if line.strip()), "")
    body = code if code.endswith("\n") else f"{code}\n"
    if _TOP_LEVEL_DECL.match(first):
        return Fixture(source=f"package p\n\n{body}", strategy="decl")
    indented = "".join(f"\t{line}\n" for line in code.splitlines())
    return Fixture(
        source=f"package p\n\nfunc _gocheck_body() {{\n{indented}}}\n",
        strategy="stmt",
    )
