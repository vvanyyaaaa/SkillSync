"""
embeddings.py

Step 7: TF-IDF + Cosine Similarity.

This module estimates how textually similar a resume and a job description
are, using classic (non-ML, non-LLM) natural language processing:

- TF-IDF (Term Frequency - Inverse Document Frequency) turns each document
  into a vector of numbers, one per word. A word gets a high score in a
  document if it appears often in that document (term frequency) but
  doesn't appear in most other documents (inverse document frequency).
  This down-weights generic, common words and up-weights words that are
  distinctive to a document.

- Cosine similarity then measures how similar two vectors are by looking
  at the angle between them, not their length. A score of 1.0 means the
  vectors point in exactly the same direction (very similar wording),
  and 0.0 means they share no overlapping vocabulary at all.

This is purely statistical text comparison — no meaning/context awareness,
no ML model, no embeddings API. That's a later step.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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