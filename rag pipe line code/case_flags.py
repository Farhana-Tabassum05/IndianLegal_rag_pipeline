import re

def identify_case_flags(text: str) -> dict:
    text = text.lower()
    return {
        "bail": bool(re.search(r"\bbail\b", text)),
        "punishment": bool(re.search(r"\bpunishment\b|\bsentence\b", text)),
        "dismissal": bool(re.search(r"\bdismiss(al|ed|ing)\b", text))
    }
