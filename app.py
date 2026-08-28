import streamlit as st

from src.parser import extract_text_from_pdf
from src.preprocessing import preprocess_text


st.title("SkillSync")

st.caption(
    "Upload your resume and a target job description. "
    "SkillSync will compare your skills with the role and highlight gaps."
)

resume = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the job description",
    height=220
)


if st.button("Analyze"):

    if resume is None:
        st.warning("Please upload your resume.")

    elif not job_description.strip():
        st.warning("Please paste a job description.")

    else:
        try:
            extracted_text = extract_text_from_pdf(resume)

            if extracted_text:
                st.success("Resume parsed successfully.")

                cleaned_text = preprocess_text(extracted_text)

                if cleaned_text:
                    st.success("Resume text cleaned successfully.")

                    st.text_area(
                        "Cleaned resume text",
                        cleaned_text,
                        height=300,
                        disabled=True
                    )

                else:
                    st.warning(
                        "No usable text remained after cleaning this PDF."
                    )

            else:
                st.warning(
                    "No text could be extracted from this PDF."
                )

        except ValueError as exc:
            st.error(str(exc))