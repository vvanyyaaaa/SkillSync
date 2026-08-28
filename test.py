from src.embeddings import calculate_similarity

resume = "Python SQL Pandas machine learning"

job = "Graphic design photography illustration painting"

score = calculate_similarity(resume, job)

print(f"Similarity: {score}%")