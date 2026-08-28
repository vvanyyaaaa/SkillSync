"""
matching.py

Step 6: Skill Matching and Skill Gap Analysis.

Given the skills found in a resume and the skills found in a job
description (both produced by extract_skills() in skills.py), figure out:

- which skills the resume already has that the job wants (matched)
- which skills the job wants that the resume is missing (missing)
- what percentage of the job's required skills are covered

This module is pure Python: no Streamlit, no ML, no external packages.
"""


def match_skills(resume_skills: list[str], job_skills: list[str]) -> dict:
    """
    Compare resume skills against job skills.

    Returns a dictionary:
    {
        "matched": [...],           # skills present in both, deterministic order
        "missing": [...],           # job skills missing from resume, deterministic order
        "match_percentage": 71.43,  # matched / total job skills * 100, rounded to 2dp
    }

    Duplicates in the input lists are safely ignored (sets handle that).
    If job_skills is empty, match_percentage is 0.0 to avoid dividing by zero.
    """
    # Sets are used here purely to compute the intersection/difference.
    # They're a natural fit because "does this skill appear in both lists"
    # and "does this skill appear in job but not resume" are exactly what
    # set intersection and set difference are built for.
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched_set = resume_set & job_set   # skills in both resume and job
    missing_set = job_set - resume_set   # job skills the resume doesn't have

    # Sets have no guaranteed order, so we can't return them directly —
    # two runs on the same input could print skills in a different order.
    # To keep results deterministic, we walk through job_skills (in its
    # original order) and pick out whichever ones landed in each set.
    matched = [skill for skill in job_skills if skill in matched_set]
    missing = [skill for skill in job_skills if skill in missing_set]

    # De-duplicate while preserving order (in case job_skills had repeats).
    matched = list(dict.fromkeys(matched))
    missing = list(dict.fromkeys(missing))

    if len(job_set) == 0:
        match_percentage = 0.0
    else:
        match_percentage = round(len(matched_set) / len(job_set) * 100, 2)

    return {
        "matched": matched,
        "missing": missing,
        "match_percentage": match_percentage,
    }