"""
embeddings.py

Step 7: TF-IDF + Cosine Similarity.
Step 8: Semantic Text Similarity using Sentence Transformers.

This module calculates two types of resume-job similarity:

1. TF-IDF similarity
   - Measures similarity based on shared vocabulary.

2. Semantic similarity
   - Uses sentence embeddings to compare the meaning
     of the resume and job description.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# SEMANTIC MODEL
# ---------------------------------------------------------

# Load the model once when the application starts.
# This prevents the model from being loaded every time
# calculate_semantic_similarity() is called.

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# TF-IDF SIMILARITY
# ---------------------------------------------------------

def calculate_similarity(
    resume_text: str,
    job_text: str,
) -> float:
    """
    Compare a resume and job description using
    TF-IDF + cosine similarity.

    Returns:
        Similarity score from 0 to 100.
    """

    if not resume_text or not job_text:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(
        [resume_text, job_text]
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2],
    )

    similarity_score = similarity_matrix[0][0]

    return round(
        similarity_score * 100,
        2,
    )


# ---------------------------------------------------------
# SEMANTIC SIMILARITY
# ---------------------------------------------------------

def calculate_semantic_similarity(
    resume_text: str,
    job_text: str,
) -> float:
    """
    Compare a resume and job description using
    sentence embeddings.

    Unlike TF-IDF, this measures semantic meaning
    rather than only shared words.

    Returns:
        Semantic similarity score from 0 to 100.
    """

    if not resume_text or not job_text:
        return 0.0

    resume_embedding = MODEL.encode(
        resume_text,
        normalize_embeddings=True,
    )

    job_embedding = MODEL.encode(
        job_text,
        normalize_embeddings=True,
    )

    similarity_score = cosine_similarity(
        [resume_embedding],
        [job_embedding],
    )[0][0]

    # Numerical safety.
    similarity_score = max(
        0.0,
        min(1.0, similarity_score),
    )

    return round(
        similarity_score * 100,
        2,
    )