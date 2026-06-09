import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    "Role": [
        "Data Scientist",
        "Machine Learning Engineer",
        "Backend Developer",
        "DevOps Engineer",
        "Cloud Architect",
        "Data Analyst",
        "Frontend Developer",
        "Cybersecurity Engineer"
    ],

    "Skills": [
        "python machine learning sql pandas numpy statistics",
        "python machine learning tensorflow pytorch deep learning",
        "java python sql api backend development",
        "aws docker kubernetes linux automation",
        "cloud aws azure docker networking",
        "sql excel power bi statistics data analysis",
        "html css javascript react ui ux",
        "security networking ethical hacking linux"
    ]
}

df = pd.DataFrame(data)

print("=== Tech Stack Recommender ===")

skill1 = input("Enter Skill 1: ")
skill2 = input("Enter Skill 2: ")
skill3 = input("Enter Skill 3: ")

user_profile = skill1 + " " + skill2 + " " + skill3

documents = df["Skills"].tolist()
documents.append(user_profile)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

user_vector = tfidf_matrix[-1]
job_vectors = tfidf_matrix[:-1]

scores = cosine_similarity(user_vector, job_vectors).flatten()


df["Similarity Score"] = scores

recommendations = df.sort_values(
    by="Similarity Score",
    ascending=False
)

print("\nTop 3 Recommended Roles:\n")

for i in range(3):
    print(
        f"{i+1}. {recommendations.iloc[i]['Role']} "
        f"({recommendations.iloc[i]['Similarity Score']:.2f})"
    )