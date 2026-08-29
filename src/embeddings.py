"""
embeddings.py

Step 7: TF-IDF + Cosine Similarity.
Step 8: Semantic Text Similarity using Sentence Transformers.

This module estimates how similar a resume and a job description are,
using two different approaches:

- calculate_similarity(): classic (non-ML, non-LLM) TF-IDF comparison.
  It only "sees" shared vocabulary — matching words, not matching meaning.

- calculate_semantic_similarity(): compares the *meaning* of the two
  texts using sentence embeddings from a local, free, pre-trained model
  (all-MiniLM-L6-v2 via sentence-transformers). This can recognize that
  "led a team of engineers" and "managed a software team" are related,
  even though they share almost no exact words.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def calculate_similarity(resume_text: str, job_text: str) -> float:
    """
    Compare a resume and a job description using TF-IDF + cosine similarity.

    Returns a similarity score from 0.0 to 100.0 (a percentage, rounded to
    2 decimal places). Returns 0.0 if either input is empty.
    """
    if not resume_text or not job_text:
        return 0.0

    # TfidfVectorizer turns each text into a TF-IDF weighted word vector.
    # fit_transform() learns the vocabulary from both documents together
    # and produces a matrix with one row per document.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

    # cosine_similarity compares the two document vectors (row 0 = resume,
    # row 1 = job) and returns a 2x2 matrix of similarities between every
    # pair of documents. We only need the similarity between the two
    # different documents, which is at position [0, 1].
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_score = similarity_matrix[0][0]

    return round(similarity_score * 100, 2)


# Loaded once, when this module is first imported, and reused by every
# call to calculate_semantic_similarity() below. See the explanation
# for why this is done at module level instead of inside the function.
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_semantic_similarity(resume_text: str, job_text: str) -> float:
    """
    Compare a resume and a job description using sentence embeddings.

    Unlike calculate_similarity() (which only compares matching words),
    this compares *meaning*: two sentences phrased very differently can
    still score highly here if they mean similar things.

    Returns a similarity score from 0.0 to 100.0 (a percentage, rounded to
    2 decimal places). Returns 0.0 if either input is empty.
    """
    if not resume_text or not job_text:
        return 0.0

    # encode() turns each text into a single fixed-length vector (an
    # "embedding") that captures its overall meaning.
    resume_embedding = MODEL.encode(resume_text)
    job_embedding = MODEL.encode(job_text)

    # cosine_similarity compares the two embedding vectors
# and returns their cosine similarity.
    # containing their cosine similarity.
    similarity_score = cosine_similarity(
    [resume_embedding],
    [job_embedding]
)[0][0]

    return round(similarity_score * 100, 2)