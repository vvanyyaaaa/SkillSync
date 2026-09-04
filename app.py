import streamlit as st
import pandas as pd

from src.recommendations import generate_recommendations
from src.scoring import (
    calculate_weighted_skill_score,
    calculate_overall_score,
)
from src.job_title import extract_job_title
from src.parser import extract_text_from_pdf
from src.preprocessing import preprocess_text
from src.skills import extract_skills, categorize_skills
from src.matching import match_skills
from src.embeddings import (
    calculate_similarity,
    calculate_semantic_similarity,
)
from src.llm import analyze_candidate


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="SkillSync",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* -------------------------------------------------
       GOOGLE FONT — SAIRA
    ------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Saira:ital,wght@0,100..900;1,100..900&display=swap');

    html, body, .stApp, .stApp *, 
    button, input, textarea, select, [data-baseweb="select"],
    .main-title, .subtitle, .section-title, .hero-badge,
    .hero-description, .score-number, .score-label,
    .metric-card, .metric-label, .metric-number,
    .category-title, .skill-text, .skill-chip,
    .recommendation-card, .recommendation-title,
    .recommendation-text, .improvement-card,
    .improvement-title, .improvement-text,
    .learning-card, .skill-name, .skill-description,
    .job-caption, .history-card, .history-score,
    .history-meta, .input-card-title, .input-card-text {
        font-family: 'Saira', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6,
    .main-title, .section-title, .score-number,
    .metric-number, .hero-badge {
        font-family: 'Saira', sans-serif !important;
        font-weight: 700 !important;
    }

    .stApp {
        overflow-x: hidden;
    }

    /* -------------------------------------------------
       HERO / LANDING HEADER
    ------------------------------------------------- */
    .hero {
        text-align: center;
        padding: 28px 20px 24px;
        margin: 4px auto 28px;
        max-width: 950px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.28);
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 14px;
    }

    .main-title {
        font-size: clamp(42px, 6vw, 68px);
        line-height: 1;
        letter-spacing: -1.5px;
        margin: 0 0 12px;
    }

    .hero-description {
        max-width: 680px;
        margin: 0 auto;
        color: rgba(255,255,255,.62);
        font-size: 17px;
        line-height: 1.6;
    }

    .subtitle {
        text-align: center;
        color: rgba(255,255,255,.48);
        font-size: 14px;
        margin-top: 10px;
    }

    /* -------------------------------------------------
       INPUT CARDS
    ------------------------------------------------- */
    .input-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.16);
        min-height: 120px;
        margin-bottom: 8px;
    }

    .input-card-title {
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .input-card-text {
        font-size: 13px;
        color: rgba(255,255,255,.52);
        line-height: 1.5;
    }

    .input-card + div {
        margin-top: 0;
    }

    /* -------------------------------------------------
       SECTION HEADINGS
    ------------------------------------------------- */
    .section-title {
        font-size: 25px;
        letter-spacing: -.3px;
        margin-top: 34px;
        margin-bottom: 16px;
    }

    .section-subtitle {
        color: rgba(255,255,255,.52);
        font-size: 14px;
        margin-top: -9px;
        margin-bottom: 18px;
    }

    /* -------------------------------------------------
       SCORE DASHBOARD
    ------------------------------------------------- */
    .score-box {
        width: 100%;
        margin: 12px auto 22px;
        padding: 28px 20px;
        border-radius: 22px;
        text-align: center;
        box-sizing: border-box;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 10px 35px rgba(0,0,0,.12);
    }

    .score-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255,255,255,.55);
        margin-bottom: 3px;
    }

    .score-number {
        font-size: 58px;
        line-height: 1.05;
        letter-spacing: -1px;
    }

    .job-caption {
        text-align: center;
        color: rgba(255,255,255,.60);
        font-size: 14px;
        margin: 4px 0 20px;
    }

    /* -------------------------------------------------
       METRIC CARDS
    ------------------------------------------------- */
    .metric-card {
        padding: 22px 15px;
        border-radius: 17px;
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.13);
        text-align: center;
        min-height: 105px;
        box-sizing: border-box;
    }

    .metric-label {
        font-size: 13px;
        color: rgba(255,255,255,.55);
        margin-bottom: 7px;
    }

    .metric-number {
        font-size: 31px;
        line-height: 1;
    }

    /* -------------------------------------------------
       SKILL CHIPS
    ------------------------------------------------- */
    .skill-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-bottom: 20px;
    }

    .skill-chip {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }

    .matched-chip {
        background: rgba(34,197,94,.12);
        border: 1px solid rgba(34,197,94,.28);
    }

    .missing-chip {
        background: rgba(239,68,68,.10);
        border: 1px solid rgba(239,68,68,.25);
    }

    .category-title {
        font-size: 15px;
        font-weight: 650;
        margin: 12px 0 7px;
    }

    .skill-text {
        font-size: 13px;
        color: rgba(255,255,255,.58);
        line-height: 1.6;
    }

    /* -------------------------------------------------
       CONTENT CARDS
    ------------------------------------------------- */
    .recommendation-card,
    .improvement-card,
    .learning-card {
        width: 100%;
        margin: 9px 0;
        padding: 17px 20px;
        border-radius: 15px;
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.12);
        box-sizing: border-box;
    }

    .recommendation-title,
    .improvement-title,
    .skill-name {
        font-size: 16px;
        font-weight: 650;
        margin-bottom: 5px;
    }

    .recommendation-text,
    .improvement-text,
    .skill-description {
        font-size: 13px;
        line-height: 1.55;
        color: rgba(255,255,255,.58);
    }

    /* -------------------------------------------------
       HISTORY
    ------------------------------------------------- */
    .history-card {
        padding: 15px 17px;
        border-radius: 14px;
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.12);
        margin: 8px 0;
    }

    .history-score {
        font-size: 20px;
        font-weight: 700;
    }

    .history-meta {
        font-size: 12px;
        color: rgba(255,255,255,.52);
        line-height: 1.5;
    }

    /* -------------------------------------------------
       BUTTONS / UPLOADER
    ------------------------------------------------- */
    div.stButton {
        display: flex;
        justify-content: center;
    }

    div.stButton > button {
        border-radius: 11px;
        padding: 10px 28px;
        font-family: 'Saira', sans-serif !important;
        font-weight: 650;
        min-height: 44px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 15px;
    }

    /* -------------------------------------------------
       RESPONSIVE
    ------------------------------------------------- */
    @media (max-width: 768px) {
        .hero {
            padding-top: 15px;
        }

        .hero-description {
            font-size: 15px;
        }

        .section-title {
            font-size: 22px;
        }

        .score-number {
            font-size: 48px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# ANALYSIS HISTORY
# ---------------------------------------------------------

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">✦ AI-POWERED CAREER MATCHING</div>
        <div class="main-title">SkillSync</div>
        <div class="hero-description">
            Know the gap. Build the skill. Get the job.
            <br>
            Analyze how closely your resume matches any job description
            and discover exactly what to improve.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

input_col1, input_col2 = st.columns(2, gap="large")

with input_col1:
    st.markdown(
        """
        <div class="input-card">
            <div class="input-card-title">📄 Your Resume</div>
            <div class="input-card-text">
                Upload your latest resume in PDF format. SkillSync will
                extract your technical skills and experience.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    resume_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"],
        label_visibility="collapsed",
    )

with input_col2:
    st.markdown(
        """
        <div class="input-card">
            <div class="input-card-title">💼 Job Description</div>
            <div class="input-card-text">
                Paste the complete job description to compare its
                requirements with your resume.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    job_description = st.text_area(
        "Paste the job description",
        height=180,
        label_visibility="collapsed",
        placeholder="Paste the job description here...",
    )


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

if st.button("✦ Analyze My Match", type="primary"):

    if resume_file is None or not job_description.strip():

        st.warning(
            "Please upload your resume and paste a job description."
        )

    else:

        # -------------------------------------------------
        # EXTRACT RESUME TEXT
        # -------------------------------------------------

        try:
            raw_resume_text = extract_text_from_pdf(resume_file)

        except Exception:

            st.error(
                "Could not read this PDF. Please upload a valid PDF."
            )

        else:

            # -------------------------------------------------
            # PREPROCESSING
            # -------------------------------------------------

            resume_text = preprocess_text(raw_resume_text)
            job_text = preprocess_text(job_description)

            # -------------------------------------------------
            # JOB TITLE
            # -------------------------------------------------

            job_title = extract_job_title(job_description)

            if job_title:
                st.markdown(
                    f'<div class="job-caption">'
                    f'Analyzing for: <strong>{job_title}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # SKILL EXTRACTION
            # -------------------------------------------------

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            resume_categories = categorize_skills(resume_skills)

            # -------------------------------------------------
            # CALCULATE ALL SCORES BEFORE DISPLAYING RESULTS
            # -------------------------------------------------
            result = match_skills(resume_skills, job_skills)

            weighted_score = calculate_weighted_skill_score(
                resume_skills,
                job_skills,
            )

            tfidf_score = calculate_similarity(
                resume_text,
                job_text,
            )

            semantic_score = calculate_semantic_similarity(
                resume_text,
                job_text,
            )

            overall_score = calculate_overall_score(
                weighted_score,
                semantic_score,
                tfidf_score,
            )

            # -------------------------------------------------
            # RESULTS DASHBOARD
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Your Match Dashboard</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">A quick breakdown of how your profile compares with this role.</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4, gap="medium")

            with col1:
                st.html(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Overall Match</div>
                        <div class="metric-number">{overall_score:.1f}%</div>
                    </div>
                    """
                )

            with col2:
                st.html(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Basic Skill Match</div>
                        <div class="metric-number">{result["match_percentage"]:.1f}%</div>
                    </div>
                    """
                )

            with col3:
                st.html(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Weighted Skill Match</div>
                        <div class="metric-number">{weighted_score:.1f}%</div>
                    </div>
                    """
                )

            with col4:
                st.html(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Skills Missing</div>
                        <div class="metric-number">{len(result["missing"])}</div>
                    </div>
                    """
                )

            # -------------------------------------------------
            # VISUAL SKILL MATCH CHART
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Skill Match Overview</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">Each required skill is scored as matched or missing based on your resume.</div>',
                unsafe_allow_html=True,
            )

            if job_skills:
                chart_rows = []
                matched_set = set(result["matched"])

                for skill in job_skills:
                    chart_rows.append({
                        "Skill": skill.title(),
                        "Match": 100 if skill in matched_set else 0,
                    })

                chart_df = pd.DataFrame(chart_rows).set_index("Skill")
                st.bar_chart(
                    chart_df,
                    y="Match",
                    height=340,
                    use_container_width=True,
                )
            else:
                st.info("No specific technical skills were detected in the job description.")

            # -------------------------------------------------
            # MATCHED + MISSING SKILLS
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Your Skill Breakdown</div>',
                unsafe_allow_html=True,
            )

            skills_col1, skills_col2 = st.columns(2, gap="large")

            with skills_col1:
                st.markdown("### ✓ Matched Skills")
                if result["matched"]:
                    chips = "".join(
                        f'<span class="skill-chip matched-chip">{skill.title()}</span>'
                        for skill in result["matched"]
                    )
                    st.markdown(
                        f'<div class="skill-grid">{chips}</div>',
                        unsafe_allow_html=True,
                    )

                    matched_categories = categorize_skills(result["matched"])
                    for category, skills in matched_categories.items():
                        st.markdown(
                            f'<div class="category-title">{category}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="skill-text">{", ".join(skills)}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No matching skills were detected.")

            with skills_col2:
                st.markdown("### ! Missing Skills")
                if result["missing"]:
                    chips = "".join(
                        f'<span class="skill-chip missing-chip">{skill.title()}</span>'
                        for skill in result["missing"]
                    )
                    st.markdown(
                        f'<div class="skill-grid">{chips}</div>',
                        unsafe_allow_html=True,
                    )

                    missing_categories = categorize_skills(result["missing"])
                    for category, skills in missing_categories.items():
                        st.markdown(
                            f'<div class="category-title">{category}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="skill-text">{", ".join(skills)}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("You cover all detected required skills.")

            # -------------------------------------------------
            # RESUME IMPROVEMENTS
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Resume Improvements</div>',
                unsafe_allow_html=True,
            )

            improvements = []

            if result["missing"]:

                improvements.append(
                    (
                        "Highlight relevant skills",
                        "Showcase projects that demonstrate skills required by the target role.",
                    )
                )

            if "AI/LLM" in resume_categories:

                improvements.append(
                    (
                        "Add measurable results",
                        "Include accuracy, latency, dataset size, or other meaningful project metrics.",
                    )
                )

            if "Data/ML" in resume_categories:

                improvements.append(
                    (
                        "Mention ML techniques",
                        "Clearly state the algorithms, techniques, and evaluation metrics used.",
                    )
                )

            improvements.append(
                (
                    "Strengthen project bullets",
                    "Keep descriptions concise and begin each bullet with a strong action verb.",
                )
            )

            for title, description in improvements:

                st.html(
                    f"""
                    <div class="improvement-card">

                        <div class="improvement-title">
                            {title}
                        </div>

                        <div class="improvement-text">
                            {description}
                        </div>

                    </div>
                    """
                )

            # -------------------------------------------------
            # RECOMMENDED PROJECTS
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Recommended Projects</div>',
                unsafe_allow_html=True,
            )

            missing = set(result["missing"])
            projects = []

            if "pytorch" in missing or "tensorflow" in missing:

                projects.append(
                    (
                        "Deep Learning Image Classifier",
                        "Build a CNN using PyTorch or TensorFlow and evaluate it on a real dataset.",
                    )
                )

            if "fastapi" in missing or "flask" in missing:

                projects.append(
                    (
                        "ML Model API",
                        "Serve a trained ML model through FastAPI or Flask with prediction endpoints.",
                    )
                )

            if "docker" in missing:

                projects.append(
                    (
                        "Containerized ML Application",
                        "Dockerize an ML project and manage its dependencies for deployment.",
                    )
                )

            if "aws" in missing:

                projects.append(
                    (
                        "Cloud ML Deployment",
                        "Deploy an ML inference application using AWS and learn basic cloud infrastructure.",
                    )
                )

            if "generative ai" in missing:

                projects.append(
                    (
                        "RAG Resume Assistant",
                        "Build a GenAI assistant using embeddings and retrieval-augmented generation.",
                    )
                )

            if not projects:

                projects.append(
                    (
                        "End-to-End ML Application",
                        "Build an ML application combining model training, an API, and deployment.",
                    )
                )

            for title, description in projects:

                st.html(
                    f"""
                    <div class="recommendation-card">

                        <div class="recommendation-title">
                            {title}
                        </div>

                        <div class="recommendation-text">
                            {description}
                        </div>

                    </div>
                    """
                )

            # -------------------------------------------------
            # RECOMMENDED LEARNING
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Recommended Learning</div>',
                unsafe_allow_html=True,
            )

            recommendations = generate_recommendations(
                result["missing"]
            )

            if recommendations:

                for item in recommendations:

                    st.html(
                        f"""
                        <div class="learning-card">

                            <div class="skill-name">
                                {item["skill"].title()}
                            </div>

                            <div class="skill-description">
                                {item["recommendation"]}
                            </div>

                        </div>
                        """
                    )

            else:

                st.html(
                    """
                    <div class="skill-text">
                        Your current skills cover the main requirements.
                    </div>
                    """
                )

            # -------------------------------------------------
            # DETAILED ANALYSIS
            # -------------------------------------------------

            with st.expander("View detailed analysis"):

                st.write(
                    f"TF-IDF Similarity: {tfidf_score:.2f}%"
                )

                st.write(
                    f"Semantic Similarity: {semantic_score:.2f}%"
                )

                st.write(
                    f"Weighted Skill Match: {weighted_score:.2f}%"
                )

                try:

                    ai_analysis = analyze_candidate(
                        resume_text,
                        job_text,
                        overall_score,
                    )

                    st.markdown("### AI Career Analysis")

                    st.markdown(ai_analysis)

                except Exception:

                    st.info(
                        "Detailed AI analysis is currently unavailable."
                    )

            # Save completed analysis in the current Streamlit session.
            st.session_state.analysis_history.insert(
                0,
                {
                    "score": overall_score,
                    "job_title": job_title or "Untitled Job",
                    "timestamp": pd.Timestamp.now().strftime("%d %b %Y, %I:%M %p"),
                    "matched": len(result["matched"]),
                    "missing": len(result["missing"]),
                },
            )

            # Keep only the 10 most recent analyses.
            st.session_state.analysis_history = (
                st.session_state.analysis_history[:10]
            )


# ---------------------------------------------------------
# ANALYSIS HISTORY
# ---------------------------------------------------------

st.markdown('<div class="section-title">Analysis History</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Your recent SkillSync analyses from this session.</div>',
    unsafe_allow_html=True,
)

if st.session_state.analysis_history:
    for entry in st.session_state.analysis_history:
        st.markdown(
            f"""
            <div class="history-card">
                <div class="history-score">{entry["score"]:.1f}% — {entry["job_title"]}</div>
                <div class="history-meta">
                    {entry["timestamp"]} &nbsp;•&nbsp;
                    ✓ {entry["matched"]} matched &nbsp;•&nbsp;
                    ! {entry["missing"]} missing
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Your completed analyses will appear here.")
