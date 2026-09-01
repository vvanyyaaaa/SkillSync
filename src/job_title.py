import re


def extract_job_title(job_description: str) -> str | None:
    """
    Try to extract the job title from a job description.

    First checks for explicit labels such as:
    Job Title: Machine Learning Engineer
    Position: Software Engineer
    Role: Data Scientist

    If no label is found, checks the first few lines for
    common job-title words.
    """

    if not job_description or not job_description.strip():
        return None

    lines = [
        line.strip()
        for line in job_description.splitlines()
        if line.strip()
    ]

    # 1. Look for explicitly labelled titles
    label_pattern = re.compile(
        r"^(job\s*title|position|role|title)\s*[:\-]\s*(.+)$",
        re.IGNORECASE
    )

    for line in lines[:10]:
        match = label_pattern.match(line)

        if match:
            return match.group(2).strip()

    # 2. Look for a likely title in the first few lines
    title_keywords = [
        "engineer",
        "developer",
        "scientist",
        "analyst",
        "manager",
        "designer",
        "architect",
        "consultant",
        "specialist",
        "intern",
        "lead",
        "director",
    ]

    for line in lines[:5]:

        word_count = len(line.split())

        if 1 <= word_count <= 8:

            line_lower = line.lower()

            if any(keyword in line_lower for keyword in title_keywords):
                return line

    return None