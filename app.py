import re
import time
import streamlit as st
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load AI model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Page setup
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("AI Resume Analyzer & Job Matching System")
st.caption("Analyze resumes, evaluate ATS compatibility, and match them with relevant technical roles.")
st.write("Upload your resume and get job recommendations")
st.markdown("""
### How it works
1. Upload your resume (PDF)
2. The system extracts skills using NLP
3. Resume and job skills are converted to BERT embeddings
4. Cosine similarity ranks the best matching roles
""")
st.divider()

st.subheader("What this tool analyzes")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧠 Skill Extraction")
    st.write("Detects technical skills from your resume using keyword mapping and pattern matching.")

with col2:
    st.markdown("### 📊 ATS Score")
    st.write("Evaluates how ATS-friendly your resume is based on skills, sections, and structure.")

with col3:
    st.markdown("### 🎯 Job Matching")
    st.write("Matches your resume with job roles using BERT embeddings and cosine similarity.")
# Load jobs
jobs = pd.read_csv("jobs.csv")
jobs['skills'] = jobs['skills'].str.lower()
@st.cache_resource
def get_job_embeddings():
    return model.encode(jobs['skills'].tolist())

# Skill keywords (important)
skill_map = {
    "python": ["python", "py"],
    "sql": ["sql", "mysql", "postgresql"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "machine learning": ["machine learning", "ml"],
    "data analysis": ["data analysis", "data analytics"],
    "flask": ["flask"],
    "django": ["django"],
    "api": ["api", "rest api"],
    "excel": ["excel"],
    "power bi": ["power bi", "powerbi"],
    "java": ["java"],
    "spring": ["spring", "spring boot"],
    "oop": ["oop", "object oriented programming"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs"],
    "bootstrap": ["bootstrap"],
    "testing": ["testing", "manual testing", "software testing"],
    "selenium": ["selenium"],
    "linux": ["linux"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "shell scripting": ["shell scripting", "bash"],
    "networking": ["networking", "computer networks"],
    "security": ["security", "cybersecurity"],
    "database optimization": ["database optimization", "query optimization"],
    "communication": ["communication"],
    "deep learning": ["deep learning", "dl"],
    "kotlin": ["kotlin"],
    "android": ["android"],
    "firebase": ["firebase"]
}


# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
st.markdown("#### Tips for better results")
st.write("• Include sections like Education, Skills, Projects, and Experience")
st.write("• Mention tools and technologies clearly")
st.write("• Add GitHub or LinkedIn links")
st.write("• Keep resume between 1–2 pages")

# Extract text from PDF
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def calculate_ats_score(resume_text, skill_names, required_skills):
    score = 0
    suggestions = []

    # 1. Skill match score (50 marks)
    matched_skills = [skill for skill in required_skills if skill in skill_names]
    skill_score = (len(matched_skills) / len(required_skills)) * 50 if len(required_skills) > 0 else 0
    score += skill_score

    # 2. Important sections check (20 marks)
    sections = ["education", "skills", "projects", "experience"]
    found_sections = [section for section in sections if section in resume_text]
    section_score = (len(found_sections) / len(sections)) * 20
    score += section_score

    # 3. Links check (15 marks)
    link_score = 0
    if "linkedin" in resume_text:
        link_score += 7.5
    else:
        suggestions.append("Add LinkedIn profile")

    if "github" in resume_text:
        link_score += 7.5
    else:
        suggestions.append("Add GitHub profile")

    score += link_score

    # 4. Resume length check (15 marks)
    word_count = len(resume_text.split())
    if 250 <= word_count <= 900:
        score += 15
    elif word_count < 250:
        suggestions.append("Resume is too short. Add more relevant details")
    else:
        suggestions.append("Resume is too long. Keep it concise and ATS-friendly")

    # Missing skills suggestion
    missing_skills = [skill for skill in required_skills if skill not in skill_names]
    if missing_skills:
        suggestions.append("Improve these skills for better match: " + ", ".join(missing_skills))

    return round(score, 2), suggestions

if uploaded_file is not None:

    resume_text = extract_text(uploaded_file).lower()

    # Extract skills (FIXED)
    found_skills = []

    for main_skill, variations in skill_map.items():
        count = 0

        for word in variations:
            pattern = r"\b" + word + r"\b"
            matches = re.findall(pattern, resume_text)
            count += len(matches)

        if count > 0:
            found_skills.append((main_skill, count))

    # Sort
    found_skills = sorted(found_skills, key=lambda x: x[1], reverse=True)

    skill_names = [skill[0] for skill in found_skills]
    user_skills = " ".join(skill_names)

    # Display
    st.subheader("Extracted Skills")

    if len(found_skills) == 0:
        st.error("No relevant skills found in resume")
    else:
        skill_html = ""
        for skill, count in found_skills:
            skill_html += f"""
            <span style="
                display:inline-block;
                background-color:#e6f3ff;
                color:#003366;
                padding:6px 12px;
                margin:4px;
                border-radius:16px;
                font-size:14px;
                font-weight:600;">
                {skill} ({count})
            </span>
            """

        st.markdown(skill_html, unsafe_allow_html=True)
    # Experience input
    user_experience = st.number_input("Enter your experience (years)", 0, 10)

    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume with AI model..."):
            progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        
        # Convert job skills and resume skills to embeddings

        job_embeddings = get_job_embeddings()
        resume_embedding = model.encode([resume_text])

        # Calculate similarity
        similarity = cosine_similarity(resume_embedding, job_embeddings)
        jobs['skill_score'] = similarity[0]

        # Experience score
        def experience_score(job_exp, user_exp):
            if user_exp >= job_exp:
                return 1
            else:
                return user_exp / job_exp

        jobs['exp_score'] = jobs['experience'].apply(lambda x: experience_score(x, user_experience))

        # Final score (advanced)
        jobs['final_score'] = (
            0.6 * jobs['skill_score'] +
            0.3 * jobs['exp_score'] +
            0.1 * (jobs['skill_score'] * jobs['exp_score'])
        )

        # Sort jobs
        recommended_jobs = jobs.sort_values(by='final_score', ascending=False)
        top_job_skills = jobs.loc[recommended_jobs.index[0], 'skills'].split()
        ats_score, ats_suggestions = calculate_ats_score(resume_text, skill_names, top_job_skills)
        # Resume Score (based on best job match)
        top_score = recommended_jobs.iloc[0]['final_score']
        resume_score = round(top_score * 100, 2)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Resume Score", f"{resume_score}/100")

        with col2:
            st.metric("ATS Score", f"{ats_score}/100")

            st.progress(ats_score / 100)

        if ats_score > 80:
            st.success("Your resume is highly ATS-friendly")
        elif ats_score > 60:
            st.warning("Your resume is moderately ATS-friendly")
        else:
            st.error("Your resume may struggle in ATS screening")
        if resume_score > 80:
            st.write("🔥 Strong profile!")
        elif resume_score > 50:
            st.write("⚡ Decent, but can improve")
        else:
            st.write("❗ Needs improvement")
        st.subheader("ATS Recommendations")

        if ats_suggestions:
            for suggestion in ats_suggestions:
                st.write(f"• {suggestion}")
        else:
            st.success("Your resume looks strong for ATS screening")

        st.divider()
        st.subheader("🎯 Best Matching Roles For Your Resume")

        top_jobs = recommended_jobs.head(3)
        for index, row in top_jobs.iterrows():
            with st.container():
                job_skills = jobs.loc[index, 'skills']
                matched = [skill for skill in skill_names if skill in job_skills]

                job_skills_list = job_skills.split()
                missing_skills = [skill for skill in job_skills_list if skill not in skill_names]

                total_required = len(job_skills_list)
                matched_count = len(matched)

                match_percent = int((matched_count / total_required) * 100) if total_required > 0 else 0

                st.markdown(f"""
                <div style="
                padding:15px;
                border-radius:10px;
                background-color:#1e1e1e;
                border-left:6px solid #4CAF50;
                margin-bottom:10px;
                color:white;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);
                ">

                <h3 style="color:white; margin-bottom:10px;">💼 {row['title']}</h3>

                <p style="color:white; margin:4px 0;"><b>Match Score:</b> {round(row['final_score']*100, 2)} %</p>
                <p style="color:white; margin:4px 0;"><b>Matched Skills:</b> {', '.join(matched)}</p>

                </div>
                """, unsafe_allow_html=True)

                st.progress(match_percent / 100)
                st.write(f"Skill Match: {match_percent}%")

                if match_percent > 75:
                    st.success("Strong skill match")
                elif match_percent > 40:
                    st.warning("Partial match - improve skills")
                else:
                    st.error("Low match - needs improvement")

                st.write("❌ Missing Skills:", ", ".join(missing_skills))
                st.divider()

        missing_skills = [skill for skill in top_job_skills if skill not in skill_names]
        if len(missing_skills) > 3:
            st.warning("You are missing several key skills for top roles")

        st.divider()

        st.markdown(
        """
        <center>
        Built with ❤️ using Python, Streamlit, and Machine Learning  
        AI Resume Analyzer Project
        </center>
        """,
        unsafe_allow_html=True
        )