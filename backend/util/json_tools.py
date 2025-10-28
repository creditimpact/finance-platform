import re


def try_fix_to_json(text: str) -> str:
    m = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        return m.group(1)
    return text
