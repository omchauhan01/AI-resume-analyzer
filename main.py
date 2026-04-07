import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# Load data
jobs = pd.read_csv("jobs.csv", encoding='utf-8')

# User input
user_skills = input("Enter your skills: ")
user_experience = int(input("Enter your experience (years): "))

# Combine skills
all_skills = jobs['skills'].tolist()
all_skills.append(user_skills)

jobs['skills'] = jobs['skills'].str.lower()
user_skills = user_skills.lower()

# Convert text to vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(all_skills)

# Calculate similarity
similarity = cosine_similarity(vectors[-1], vectors[:-1])
jobs['skill_score'] = similarity[0]

# Experience scoring
def experience_score(job_exp, user_exp):
    if user_exp >= job_exp:
        return 1
    else:
        return user_exp / job_exp

jobs['exp_score'] = jobs['experience'].apply(lambda x: experience_score(x, user_experience))

# Final score
jobs['final_score'] = jobs.apply(
    lambda row: 0 if row['skill_score'] < 0.2 else (
        0.6 * row['skill_score'] +
        0.3 * row['exp_score'] +
        0.1 * (row['skill_score'] * row['exp_score'])
    ),
    axis=1
)

# Sort jobs
recommended_jobs = jobs.sort_values(by='final_score', ascending=False)

print("\nTop Recommendations:\n")
print(recommended_jobs[['title', 'final_score']].head(2))
# Save to database
conn = sqlite3.connect("jobs.db")
jobs.to_sql("job_recommendations", conn, if_exists="replace", index=False)
conn.close()