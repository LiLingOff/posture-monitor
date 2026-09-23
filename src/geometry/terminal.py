"""終端機輸出的寬度計算。

中日韓字元在終端機佔兩欄，但 `len()` 只算一個。用 f-string 的 `:<16` 排版，
或是用 `text[:width]` 裁切，都會以字元數為準，輸出的欄位對不齊、
或是裁完仍然超出畫面。
"""
from __future__ import annotations

import unicodedata


def display_width(text: str) -> int:
    """這段文字在終端機上佔幾欄。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def cell(text: str, width: int, align: str = "left") -> str:
    """照顯示寬度補空白，排成固定寬度的欄位。"""
    pad = " " * max(0, width - display_width(text))
    return text + pad if align == "left" else pad + text


def truncate(text: str, width: int) -> str:
    """裁到指定的顯示寬度。

    中文一個字佔兩欄，所以 `text[:width]` 留下來的字元數雖然對，
    佔用的欄數可能是兩倍，一行仍然會折行。
    """
    if display_width(text) <= width:
        return text
    used = 0
    out = []
    for char in text:
        w = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + w > width:
            break
        out.append(char)
        used += w
    return "".join(out)
