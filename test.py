from src.embeddings import calculate_semantic_similarity

resume = """
I am a machine learning engineer experienced in Python and predictive modeling.
"""

job = """
We need a machine learning developer with Python experience who builds predictive models.
"""

score = calculate_semantic_similarity(resume, job)

print(score)
print(type(score))