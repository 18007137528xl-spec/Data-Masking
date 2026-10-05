"""Shape-preserving dummy text for free-text columns.

The shape is what survives: length, word breaks, punctuation, where the
digits were, upper and lower case. The letters do not. A short value with
little room ('NA', 'Y') still gets a dummy, but a collision there is refused
by the vault rather than quietly issued twice.
"""

from __future__ import annotations

import secrets
import string
from typing import Any

import pandas as pd

from .vault import Vault

_UPPER = string.ascii_uppercase
_DIGITS = string.digits
# The common CJK block. Wide enough that a random draw reads as Chinese text
# and carries nothing of the original.
_CJK_LO, _CJK_HI = 0x4E00, 0x9FA5


def _is_cjk(ch: str) -> bool:
    return _CJK_LO <= ord(ch) <= _CJK_HI


def _draw(key: str) -> str:
    """A random string in the shape of ``key`` (which is upper case)."""
    out = []
    for ch in key:
        if "A" <= ch <= "Z":
            out.append(secrets.choice(_UPPER))
        elif ch.isdigit():
            out.append(secrets.choice(_DIGITS))
        elif _is_cjk(ch):
            out.append(chr(_CJK_LO + secrets.randbelow(_CJK_HI - _CJK_LO + 1)))
        elif ch.isalpha():
            # Accented and other-script letters: a plain letter in their place.
            out.append(secrets.choice(_UPPER))
        else:
            out.append(ch)
    return "".join(out)


def _recase(dummy: str, original: str) -> str:
    """Give the dummy the original's case, character by character."""
    return "".join(
        d.lower() if o.islower() else d for d, o in zip(dummy, original)
    )


def dummy_value(value: Any, vault: Vault) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return value
    text = str(value)
    core = text.strip()
    if not core:
        return value
    key = core.upper()
    if len(key) != len(core):
        # A character whose upper case is longer (German sharp s): key the
        # value as written instead, so the shape still lines up.
        key = core
    dummy = vault.text_dummy(key, lambda: _draw(key))
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    return lead + _recase(dummy, core) + trail


def dummy_column(col: pd.Series, vault: Vault) -> pd.Series:
    cache: dict[Any, Any] = {}
    out = []
    for v in col:
        k = v if isinstance(v, str) else repr(v)
        if k not in cache:
            cache[k] = dummy_value(v, vault)
        out.append(cache[k])
    return pd.Series(out, index=col.index, dtype=object)
