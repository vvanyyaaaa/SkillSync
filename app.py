
import streamlit as st



from src.parser import extract_text_from_pdf
from src.preprocessing import preprocess_text
from src.skills import extract_skills
from src.matching import match_skills
from src.embeddings import calculate_similarity

st.title("SkillSync")

resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description")

if st.button("Analyze"):
    if resume_file is not None and job_description.strip():
        try:
            raw_resume_text = extract_text_from_pdf(resume_file)
        except Exception:
            st.error("Could not read this PDF. Please upload a valid PDF file.")
        else:
            resume_text = preprocess_text(raw_resume_text)
            job_text = preprocess_text(job_description)

            st.subheader("Extracted Resume Text")
            st.write(resume_text)

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            st.subheader("Detected Resume Skills")
            st.write(", ".join(resume_skills) if resume_skills else "No skills detected.")

            st.subheader("Detected Job Skills")
            st.write(", ".join(job_skills) if job_skills else "No skills detected.")

            result = match_skills(resume_skills, job_skills)

            st.subheader("Skill Match")
            st.write(f"{result['match_percentage']}%")

            st.subheader("Matched Skills")
            st.write(", ".join(result["matched"]) if result["matched"] else "None")

            st.subheader("Missing Skills")
            st.write(", ".join(result["missing"]) if result["missing"] else "None")

            similarity_score = calculate_similarity(resume_text, job_text)

            st.subheader("Resume–Job Similarity")
            st.write(f"{similarity_score}%")
    else:
        st.warning("Please upload a resume and paste a job description.")