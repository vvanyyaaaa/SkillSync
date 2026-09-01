import os

from google import genai


def analyze_candidate(
    resume_text: str,
    job_text: str,
    matched_skills: list[str],
    missing_skills: list[str],
    basic_skill_score: float,
    weighted_skill_score: float,
    tfidf_score: float,
    semantic_score: float,
    overall_score: float,
) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Gemini API key is not configured."

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an AI career assistant inside a resume analysis application called SkillSync.

Analyze the candidate's resume against the job description using ONLY the information provided below.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_text}

MATCHED SKILLS:
{matched_skills}

MISSING SKILLS:
{missing_skills}

BASIC SKILL MATCH:
{basic_skill_score}%

WEIGHTED SKILL MATCH:
{weighted_skill_score}%

TF-IDF SIMILARITY:
{tfidf_score}%

SEMANTIC SIMILARITY:
{semantic_score}%

OVERALL COMPATIBILITY:
{overall_score}%

Provide a concise and practical career analysis.

Use exactly these sections:

1. Overall Assessment
Give a short assessment of how well the candidate fits the role.

2. Candidate Strengths
List 3-5 strengths that are directly supported by the resume.

3. Important Skill Gaps
List the most important missing skills from the job description.

4. Recommended Learning
Recommend what the candidate should learn first. Prioritize the most important gaps.

5. Resume Improvement Suggestions
Give 3-5 specific suggestions for improving the resume for this role.

6. Interview Preparation
Give 3-5 technical topics the candidate should prepare for an interview for this role.

Rules:
- Do not invent experience that is not present in the resume.
- Do not claim the candidate knows a technology unless it appears in the resume or matched skills.
- Focus on the specific job description.
- Keep the analysis concise.
- Be honest about weaknesses.
- Do not simply repeat the entire resume.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt,
        )

        return response.text

    except Exception as e:
        return f"Unable to generate AI analysis: {e}"