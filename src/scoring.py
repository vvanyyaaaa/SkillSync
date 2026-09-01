
"""
scoring.py

SkillSync scoring system.

Calculates:
1. Weighted Skill Match
2. Overall Compatibility Score
"""

from src.skills import get_category


# ---------------------------------------------------------
# CATEGORY WEIGHTS
# ---------------------------------------------------------

CATEGORY_WEIGHTS = {
    "AI/LLM": 1.3,
    "Data/ML": 1.2,
    "Frameworks": 1.0,
    "Programming Languages": 1.0,
    "Cloud/DevOps": 0.9,
    "CS Fundamentals": 0.8,
    "Other": 1.0,
}


# ---------------------------------------------------------
# WEIGHTED SKILL SCORE
# ---------------------------------------------------------

def calculate_weighted_skill_score(
    resume_skills: list[str],
    job_skills: list[str],
) -> float:
    """
    Calculate how well the candidate's skills match
    the skills required by the job.

    Skills belonging to more important categories receive
    higher weights.

    Returns:
        Weighted skill match percentage from 0 to 100.
    """

    if not job_skills:
        return 0.0

    resume_set = {
        skill.strip().lower()
        for skill in resume_skills
    }

    job_set = {
        skill.strip().lower()
        for skill in job_skills
    }

    total_weight = 0.0
    matched_weight = 0.0

    for skill in job_set:

        category = get_category(skill)

        weight = CATEGORY_WEIGHTS.get(
            category,
            1.0,
        )

        total_weight += weight

        if skill in resume_set:
            matched_weight += weight

    if total_weight == 0:
        return 0.0

    score = (
        matched_weight
        / total_weight
    ) * 100

    return round(score, 2)


# ---------------------------------------------------------
# OVERALL COMPATIBILITY SCORE
# ---------------------------------------------------------

def calculate_overall_score(
    weighted_skill_score: float,
    semantic_score: float,
    tfidf_score: float,
) -> float:
    """
    Calculate the final SkillSync compatibility score.

    Components:

    Weighted Skill Match → 50%
    Semantic Similarity → 30%
    TF-IDF Similarity   → 20%

    Returns:
        Overall compatibility percentage from 0 to 100.
    """

    overall_score = (
        weighted_skill_score * 0.50
        + semantic_score * 0.30
        + tfidf_score * 0.20
    )

    # Keep the score within a valid percentage range.
    overall_score = max(
        0.0,
        min(100.0, overall_score),
    )

    return round(overall_score, 2)