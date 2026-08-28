import re


def preprocess_text(text: str) -> str:
    """Clean extracted resume or job-description text in a deterministic way.

    Normalizes line endings and whitespace, drops empty and consecutive
    duplicate lines, and lowercases the result so later skill matching can
    be case-insensitive. Punctuation is kept so tokens such as C++, C#,
    .NET, and Node.js stay intact.
    """
    if not text or not str(text).strip():
        return ""

    # PDF text may use Windows (\\r\\n) or old Mac (\\r) line breaks.
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []
    previous_line: str | None = None

    for raw_line in normalized.split("\n"):
        # Collapse runs of spaces/tabs; strip the ends of the line.
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue

        # Lowercase so "GitHub" and "github" are treated the same later.
        line = line.lower()

        if line == previous_line:
            continue

        cleaned_lines.append(line)
        previous_line = line

    return "\n".join(cleaned_lines)
