from __future__ import annotations

import re

from go_ast_assistant.paper4.config import CHECK_NAMES


def _key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


_LOOKUP = {_key(check): check for check in CHECK_NAMES}
_LOOKUP.update(
    {
        _key("capitalized local"): "captLocal",
        _key("builtin shadow"): "builtinShadow",
        _key("builtin shadowing"): "builtinShadow",
        _key("else if"): "elseif",
        _key("if else chain"): "ifElseChain",
        _key("single case switch"): "singleCaseSwitch",
        _key("param type combine"): "paramTypeCombine",
        _key("parameter type combine"): "paramTypeCombine",
        _key("assignment operator"): "assignOp",
        _key("comment formatting"): "commentFormatting",
    }
)


def normalize(raw: str) -> str | None:
    return _LOOKUP.get(_key(raw))
