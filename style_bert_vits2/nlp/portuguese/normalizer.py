import re

_whitespace_re = re.compile(r"\s+")

def lowercase(text: str) -> str:
    return text.lower()

def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text)

def replace_punctuation(text: str) -> str:
    REPLACE_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "．": ".",
        "…": "...",
        "···": "...",
        "—": "-",
        "−": "-",
        "～": "-",
        "~": "-",
        "“": "'",
        "”": "'",
        '"': "'",
        "‘": "'",
        "’": "'",
        "（": "(",
        "）": ")",
        "「": "'",
        "」": "'",
    }
    pattern = re.compile("|".join(re.escape(p) for p in REPLACE_MAP))
    return pattern.sub(lambda m: REPLACE_MAP[m.group()], text)

def normalize_text(text: str) -> str:
    text = replace_punctuation(text)
    text = re.sub(r"([,;.!?])(\S)", r"\1 \2", text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
