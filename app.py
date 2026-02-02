# AI Resume Skill Validation & Job Matching System
# Semester End Term Project
# Language: Python

import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Predefined skill set
SKILLS = [
    "python", "java", "c++", "sql", "machine learning",
    "deep learning", "data science", "flask", "django",
    "html", "css", "javascript", "aws", "docker", "linux"
]

# Function to read PDF resume
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Skill extraction function
def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)
    return list(set(found_skills))

# Resume-job matching function
def calculate_match(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)

# ---------------- UI ---------------- #
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("AI Resume Skill Validation & Job Matching System")

st.write("Upload a resume and paste a job description to analyze skill match.")

resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here")

if st.button("Analyze Resume"):
    if resume_file and job_description.strip() != "":
        resume_text = read_pdf(resume_file)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)

        match_score = calculate_match(resume_text, job_description)
        missing_skills = list(set(job_skills) - set(resume_skills))

        st.subheader("Analysis Result")
        st.write(f"### Match Score: **{match_score}%**")

        st.write("#### Matched Skills")
        if resume_skills:
            st.success(", ".join(resume_skills))
        else:
            st.warning("No matching skills found")

        st.write("#### Missing Skills")
        if missing_skills:
            st.error(", ".join(missing_skills))
        else:
            st.success("No missing skills")

        if match_score >= 70:
            st.success("Recommendation: Candidate is suitable for interview.")
        else:
            st.warning("Recommendation: Candidate needs skill improvement.")

    else:
        st.error("Please upload a resume and enter a job description.")
