"""
Step 14: Skill-Gap Recommendations.

Provides deterministic learning recommendations
and priority levels for missing skills.
"""

SKILL_RECOMMENDATIONS = {
    "docker": (
        "Learn containerization, Dockerfiles, images, containers, "
        "volumes, networking, and Docker Compose."
    ),
    "aws": (
        "Learn AWS fundamentals, especially EC2, S3, IAM, "
        "and basic cloud deployment."
    ),
    "tensorflow": (
        "Learn TensorFlow basics, tensors, neural networks, "
        "model training, and model evaluation."
    ),
    "pytorch": (
        "Learn PyTorch tensors, datasets, neural networks, "
        "training loops, and model evaluation."
    ),
    "deep learning": (
        "Learn neural networks, backpropagation, CNNs, "
        "optimizers, loss functions, and model evaluation."
    ),
    "fastapi": (
        "Learn REST API development with FastAPI, including "
        "routes, request validation, responses, and deployment."
    ),
    "flask": (
        "Learn Flask fundamentals, routing, request handling, "
        "REST APIs, and deployment."
    ),
    "generative ai": (
        "Learn generative AI fundamentals, LLMs, prompting, "
        "embeddings, RAG, and API-based AI applications."
    ),
    "llm": (
        "Learn transformer-based language models, prompting, "
        "embeddings, context windows, and LLM APIs."
    ),
    "natural language processing": (
        "Learn text preprocessing, tokenization, vectorization, "
        "embeddings, sentiment analysis, and transformer models."
    ),
    "computer vision": (
        "Learn image preprocessing, CNNs, object detection, "
        "image classification, and OpenCV."
    ),
    "git": (
        "Learn Git fundamentals including branching, merging, "
        "rebasing, commits, and pull requests."
    ),
    "linux": (
        "Learn Linux commands, file permissions, processes, "
        "shell scripting, and basic system administration."
    ),
    "sql": (
        "Strengthen SQL with joins, subqueries, aggregations, "
        "CTEs, window functions, and query optimization."
    ),
    "scikit-learn": (
        "Practice supervised and unsupervised learning with "
        "scikit-learn, including preprocessing, pipelines, "
        "model selection, and evaluation."
    ),
    "machine learning": (
        "Strengthen machine learning fundamentals including "
        "regression, classification, feature engineering, "
        "model evaluation, and cross-validation."
    ),
}


# Higher number = higher learning priority.
SKILL_PRIORITIES = {
    "pytorch": 3,
    "tensorflow": 3,
    "deep learning": 3,
    "fastapi": 3,
    "flask": 3,
    "generative ai": 3,
    "llm": 3,
    "natural language processing": 3,
    "computer vision": 3,

    "docker": 2,
    "sql": 2,
    "scikit-learn": 2,
    "machine learning": 2,
    "git": 2,
    "linux": 2,

    "aws": 1,
}


def get_recommendation(skill: str) -> str:
    """
    Return a learning recommendation for a missing skill.
    """
    return SKILL_RECOMMENDATIONS.get(
        skill.lower(),
        f"Learn the fundamentals and build a small project using {skill}."
    )


def get_priority(skill: str) -> str:
    """
    Return the learning priority for a skill.

    High:
        Core skills that are highly relevant to ML/AI roles.

    Medium:
        Supporting technical skills.

    Low:
        Useful but generally secondary infrastructure skills.
    """
    priority = SKILL_PRIORITIES.get(skill.lower(), 2)

    if priority == 3:
        return "High"
    elif priority == 2:
        return "Medium"
    else:
        return "Low"


def generate_recommendations(missing_skills: list[str]) -> list[dict]:
    """
    Generate recommendations for all missing skills.

    Recommendations are sorted by learning priority:
    High → Medium → Low.

    Returns a list of dictionaries containing:
    - skill
    - priority
    - recommendation
    """
    recommendations = []

    for skill in missing_skills:
        recommendations.append({
            "skill": skill,
            "priority": get_priority(skill),
            "recommendation": get_recommendation(skill),
        })

    # Sort by priority while preserving the original order
    # for skills having the same priority.
    priority_order = {
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    recommendations.sort(
        key=lambda item: priority_order[item["priority"]],
        reverse=True,
    )

    return recommendations