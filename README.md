# AI Resume Analyzer

AI-powered resume analysis tool that extracts skills from resumes and recommends suitable job roles.

## Features

- Upload a resume (PDF)
- Extract skills automatically
- Match resume skills with job roles
- Show job match percentage
- Identify missing skills

## Tech Stack

- Python
- Streamlit
- NLP
- TF-IDF
- Cosine Similarity
- Pandas
- Scikit-learn

## How It Works

1. User uploads a resume
2. System extracts skills from the document
3. Resume skills are compared with job skill datasets
4. Similarity score is calculated
5. Best matching jobs are recommended

## Installation

```bash
pip install streamlit pandas scikit-learn PyPDF2

## run the app
streamlit run app.py