import streamlit as st

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
       GLOBAL PAGE
    ------------------------------------------------- */

    .stApp {
        overflow-x: hidden;
    }

    /* Use the new font only for our custom text */
    .main-title,
    .subtitle,
    .section-title,
    .score-number,
    .metric-card,
    .metric-label,
    .metric-number,
    .category-title,
    .skill-text,
    .recommendation-card,
    .recommendation-title,
    .recommendation-text,
    .improvement-card,
    .improvement-title,
    .improvement-text,
    .learning-card,
    .skill-name,
    .skill-description,
    .job-caption {
        font-family: "Segoe UI", Arial, sans-serif !important;
    }


    /* -------------------------------------------------
       HEADER
    ------------------------------------------------- */

    .main-title {
        font-size: 42px;
        font-weight: 600;
        letter-spacing: -1px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.55);
        font-size: 15px;
        font-weight: 400;
        margin-bottom: 32px;
    }


    /* -------------------------------------------------
       JOB TITLE
    ------------------------------------------------- */

    .job-caption {
        text-align: center;
        color: rgba(255, 255, 255, 0.60);
        font-size: 14px;
        margin-top: 5px;
        margin-bottom: 15px;
    }


    /* -------------------------------------------------
       OVERALL SCORE
    ------------------------------------------------- */

    .score-box {
        width: 55%;
        max-width: 650px;
        min-width: 280px;
        margin: 25px auto 35px auto;
        padding: 30px 20px;
        border-radius: 20px;
        text-align: center;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        box-sizing: border-box;
    }

    .score-number {
        font-size: 50px;
        font-weight: 600;
        letter-spacing: -1px;
    }


    /* -------------------------------------------------
       SECTION HEADINGS
    ------------------------------------------------- */

    .section-title {
        text-align: center;
        font-size: 22px;
        font-weight: 500;
        letter-spacing: -0.2px;
        margin-top: 30px;
        margin-bottom: 16px;
    }


    /* -------------------------------------------------
       SKILL MATCH CARDS
    ------------------------------------------------- */

    .metric-card {
        padding: 22px 15px;
        border-radius: 16px;
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.12);
        text-align: center;
        margin-bottom: 10px;
        box-sizing: border-box;
    }

    .metric-label {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.58);
        font-weight: 400;
        margin-bottom: 7px;
    }

    .metric-number {
        font-size: 31px;
        font-weight: 600;
    }


    /* -------------------------------------------------
       MATCHED SKILLS
    ------------------------------------------------- */

    .category-title {
        text-align: center;
        font-size: 14px;
        font-weight: 600;
        margin-top: 13px;
        margin-bottom: 4px;
    }

    .skill-text {
        text-align: center;
        font-size: 13px;
        color: rgba(255, 255, 255, 0.58);
        line-height: 1.6;
        margin-bottom: 8px;
    }


    /* -------------------------------------------------
       RECOMMENDATION CARDS
    ------------------------------------------------- */

    .recommendation-card {
        width: 82%;
        max-width: 800px;
        margin: 10px auto;
        padding: 16px 20px;
        border-radius: 14px;
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.12);
        text-align: center;
        box-sizing: border-box;
    }

    .recommendation-title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .recommendation-text {
        font-size: 13px;
        line-height: 1.55;
        color: rgba(255, 255, 255, 0.58);
    }


    /* -------------------------------------------------
       IMPROVEMENT CARDS
    ------------------------------------------------- */

    .improvement-card {
        width: 82%;
        max-width: 800px;
        margin: 10px auto;
        padding: 15px 20px;
        border-radius: 14px;
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.12);
        text-align: center;
        box-sizing: border-box;
    }

    .improvement-title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .improvement-text {
        font-size: 13px;
        color: rgba(255, 255, 255, 0.58);
        line-height: 1.55;
    }


    /* -------------------------------------------------
       LEARNING CARDS
    ------------------------------------------------- */

    .learning-card {
        width: 82%;
        max-width: 800px;
        margin: 10px auto;
        padding: 15px 20px;
        border-radius: 14px;
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.12);
        text-align: center;
        box-sizing: border-box;
    }

    .skill-name {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .skill-description {
        font-size: 13px;
        color: rgba(255, 255, 255, 0.58);
        line-height: 1.55;
    }


    /* -------------------------------------------------
       ANALYZE BUTTON
    ------------------------------------------------- */

    div.stButton {
        display: flex;
        justify-content: center;
    }

    div.stButton > button {
        border-radius: 10px;
        padding: 8px 25px;
        font-weight: 500;
    }


    /* -------------------------------------------------
       RESPONSIVE
    ------------------------------------------------- */

    @media (max-width: 768px) {

        .score-box {
            width: 90%;
        }

        .recommendation-card,
        .improvement-card,
        .learning-card {
            width: 95%;
        }

        .main-title {
            font-size: 34px;
        }

        .section-title {
            font-size: 20px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">SkillSync</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered resume and job description matching'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

resume_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"],
)

job_description = st.text_area(
    "Paste the job description",
    height=180,
)


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

if st.button("Analyze Resume", type="primary"):

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
            # SKILL MATCHING
            # -------------------------------------------------

            result = match_skills(
                resume_skills,
                job_skills,
            )

            weighted_score = calculate_weighted_skill_score(
                resume_skills,
                job_skills,
            )

            # -------------------------------------------------
            # TEXT SIMILARITY
            # -------------------------------------------------

            tfidf_score = calculate_similarity(
                resume_text,
                job_text,
            )

            semantic_score = calculate_semantic_similarity(
                resume_text,
                job_text,
            )

            # -------------------------------------------------
            # OVERALL SCORE
            # -------------------------------------------------

            overall_score = calculate_overall_score(
                weighted_score,
                semantic_score,
                tfidf_score,
            )

            # -------------------------------------------------
            # SCORE COLOR
            # -------------------------------------------------

            if overall_score >= 70:

                background = "rgba(34, 197, 94, 0.16)"
                border = "rgba(34, 197, 94, 0.45)"

            elif overall_score >= 40:

                background = "rgba(234, 179, 8, 0.16)"
                border = "rgba(234, 179, 8, 0.45)"

            else:

                background = "rgba(239, 68, 68, 0.16)"
                border = "rgba(239, 68, 68, 0.45)"

            # -------------------------------------------------
            # OVERALL COMPATIBILITY
            # -------------------------------------------------

            st.html(
                f"""
                <div class="score-box"
                     style="
                     background: {background};
                     border: 1px solid {border};
                     ">
                    <div class="score-number">
                        {overall_score:.1f}%
                    </div>
                </div>
                """
            )

            # -------------------------------------------------
            # SKILL MATCH
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">Skill Match</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:

                st.html(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Basic Skill Match
                        </div>

                        <div class="metric-number">
                            {result["match_percentage"]:.1f}%
                        </div>
                    </div>
                    """
                )

            with col2:

                st.html(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Weighted Skill Match
                        </div>

                        <div class="metric-number">
                            {weighted_score:.1f}%
                        </div>
                    </div>
                    """
                )

            # -------------------------------------------------
            # MATCHED SKILLS
            # -------------------------------------------------

            matched_categories = categorize_skills(
                result["matched"]
            )

            if matched_categories:

                st.markdown(
                    '<div class="section-title">Matched Skills</div>',
                    unsafe_allow_html=True,
                )

                for category, skills in matched_categories.items():

                    st.html(
                        f"""
                        <div class="category-title">
                            {category}
                        </div>

                        <div class="skill-text">
                            {", ".join(skills)}
                        </div>
                        """
                    )

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