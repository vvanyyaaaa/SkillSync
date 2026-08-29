"""
skills.py
 
Step 5: Skill Extraction.
 
Given raw resume text or job description text, find which known technical
skills are mentioned in it. This uses a curated list of skills + regex
pattern matching. No ML, no embeddings — just pattern matching for now.
"""
 
import re
 
# ---------------------------------------------------------------------------
# 1. Curated skill list
# ---------------------------------------------------------------------------
# The "display" casing we want to see in results (extract_skills lowercases
# these before returning, but keeping proper casing here makes the list
# easier to read/maintain).
 
SKILLS = [
    "Python",
    "C",
    "C++",
    "C#",
    "Java",
    "JavaScript",
    "TypeScript",
    "SQL",
    "HTML",
    "CSS",
    "Git",
    "GitHub",
    "Docker",
    "Linux",
    "AWS",
    "Azure",
    "GCP",
    "React",
    "Node.js",
    "MongoDB",
    "MySQL",
    "PostgreSQL",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Data Structures",
    "Algorithms",
    "OOP",
    "FastAPI",
    "Flask",
    "Streamlit",
    "REST API",
    "Generative AI",
    "LLM",
    "Prompt Engineering",
    "Gemini API",
]
 
# ---------------------------------------------------------------------------
# 2. Build one compiled regex pattern per skill
# ---------------------------------------------------------------------------
# We can't just use Python's \b word-boundary escape hatch for everything.
# \b only fires at a transition between a "word" character (letters/digits/_)
# and a "non-word" character. That works fine for plain words like "python",
# but it breaks for things like "C++" and "C#":
#
#   "C++" -> \bC\+\+\b
#   The trailing \b needs a word char on one side and a non-word char on the
#   other. But the character right before that final boundary is "+", which
#   is ALREADY a non-word character. If "C++" is followed by a space (also
#   non-word), there's no transition there at all, so \b silently fails to
#   match and "C++" would never be detected at the end of a sentence like
#   "experience in C++."
#
# The fix: define our own "boundary" using lookaround assertions that check
# for alphanumeric characters specifically (not the broader \w definition).
# A match is only valid if the character immediately before and after it is
# NOT a letter or digit. This correctly treats spaces, commas, periods,
# parentheses, AND symbols like "+" or "#" as valid boundaries.
 
_BOUNDARY_CHARS = "a-z0-9"
 
 
def _build_pattern(skill: str) -> re.Pattern:
    """Turn a plain skill name into a safe, case-insensitive regex pattern."""
    lowered = skill.lower()
 
    # Escape regex special characters (the '+' in C++, the '#' in C#, etc.)
    escaped = re.escape(lowered)
 
    # We want a skill like "machine learning" to still match if the resume
    # has extra whitespace or a line break between the two words, so we
    # turn spaces into "one or more whitespace characters". Depending on
    # the Python version, re.escape() may turn a space into "\ "
    # (backslash + space) or leave it as a plain space — handle both.
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(" ", r"\s+")
 
    pattern = rf"(?<![{_BOUNDARY_CHARS}]){escaped}(?![{_BOUNDARY_CHARS}])"
    return re.compile(pattern)
 
 
# A few skills have common alternate spellings. We match any of these
# variants, but always report back the single canonical skill name.
_ALIASES = {
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Node.js": ["node.js", "node js", "nodejs"],
    "Natural Language Processing": [
        "natural language processing",
        "nlp",
    ],
}
 
 
def _build_patterns_for_skill(skill: str) -> list[re.Pattern]:
    variants = _ALIASES.get(skill, [skill])
    return [_build_pattern(variant) for variant in variants]
 
 
# Precompute once at import time: {canonical skill name: [compiled patterns]}
_SKILL_PATTERNS = {skill: _build_patterns_for_skill(skill) for skill in SKILLS}
 
 
# ---------------------------------------------------------------------------
# 3. Public function
# ---------------------------------------------------------------------------
 
def extract_skills(text: str) -> list[str]:
    """
    Scan `text` and return the list of known skills found in it.
 
    - Matching is case-insensitive.
    - Each skill appears at most once in the result.
    - Order follows the SKILLS list above (not order of appearance in text).
    - Returns [] for empty/None input.
    """
    if not text:
        return []
 
    text_lower = text.lower()
    found = []
 
    for skill in SKILLS:
        patterns = _SKILL_PATTERNS[skill]
        if any(pattern.search(text_lower) for pattern in patterns):
            found.append(skill.lower())
 
    return found