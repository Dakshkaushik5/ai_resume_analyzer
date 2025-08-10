import streamlit as st
import os
import fitz
from docx import Document
import spacy
import re
import shutil
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import base64

# --- Load Environment Variables and API Key ---
# No API key needed for Ollama, but we'll keep this to avoid errors
load_dotenv()

# ---
# CRITICAL FIX: The st.set_page_config() call must be the first Streamlit command.
# ---
st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load NLP Model ---
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# A simple list of skills for keyword matching (you can expand this list)
COMMON_SKILLS = [
    "Python", "Java", "C++", "JavaScript", "SQL", "MongoDB", "React", "Angular",
    "Node.js", "Django", "Flask", "AWS", "Azure", "GCP",
    "Machine Learning", "Deep Learning", "Data Science", "NLP",
    "Data Analysis", "Tableau", "Power BI", "Excel",
    "Agile", "Scrum", "Project Management", "Communication", "Leadership"
]

# --- Pre-defined Job Roles and Descriptions ---
JOB_DESCRIPTIONS = {
    "Data Scientist": {
        "description": "Analyzes complex data sets, builds predictive models, and creates visualizations. Requires strong programming in Python/R, knowledge of machine learning algorithms, and statistical analysis. Experience with SQL and cloud platforms (AWS, Azure, GCP) is a plus.",
        "keywords": ["Python", "R", "Machine Learning", "Deep Learning", "SQL", "Statistics", "Data Analysis", "Predictive Modeling", "Tableau", "AWS", "Azure", "GCP", "NLP"]
    },
    "Software Engineer (Backend)": {
        "description": "Designs, develops, and maintains scalable backend services and APIs. Requires expertise in Python, Java, or Node.js, database management (SQL/NoSQL), and cloud deployment. Knowledge of microservices and RESTful APIs is essential.",
        "keywords": ["Python", "Java", "Node.js", "Flask", "Django", "Spring Boot", "SQL", "MongoDB", "PostgreSQL", "RESTful APIs", "Microservices", "Docker", "Kubernetes", "AWS", "Azure"]
    },
    "Marketing Manager": {
        "description": "Develops and executes marketing strategies, manages campaigns, and analyzes market trends. Requires strong communication, project management, digital marketing, SEO/SEM, and content creation skills. Experience with marketing automation platforms is beneficial.",
        "keywords": ["Marketing Strategy", "Digital Marketing", "SEO", "SEM", "Content Creation", "Social Media", "Campaign Management", "Market Research", "Communication", "Project Management", "Analytics"]
    }
}

# --- Parsing Functions ---

def parse_pdf(file_path):
    """Parses a PDF file and extracts its text content."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error parsing PDF file: {e}")
        return None
    return text

def parse_docx(file_path):
    """Parses a DOCX file and extracts its text content."""
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error parsing DOCX file: {e}")
        return None
    return text

def parse_resume(file_path, file_type):
    """Master function to call the appropriate parser based on file type."""
    if file_type == "application/pdf":
        return parse_pdf(file_path)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(file_path)
    else:
        st.error("Unsupported file type.")
        return None

# --- Information Extraction Function ---

def extract_info(resume_text):
    """Extracts key information like skills, education, and experience from resume text."""
    doc = nlp(resume_text)
    
    found_skills = []
    text_lower = resume_text.lower()
    for skill in COMMON_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    found_skills = list(set(found_skills))

    education_patterns = [
        r"Bachelor's degree|Master's degree|Ph\.D\.|B\.S\.|M\.S\.|B\.A\.|M\.A\.",
        r"University of [A-Za-z\s]+",
        r"[A-Za-z\s]+ University",
        r"Institute of [A-Za-z\s]+"
    ]
    education = []
    for pattern in education_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        education.extend(matches)
    education = list(set(education))

    experience_text = "Not Found"
    experience_section_keywords = ["experience", "work history", "professional experience"]
    
    start_index = -1
    text_lower = resume_text.lower()
    for keyword in experience_section_keywords:
        start_index = text_lower.find(keyword)
        if start_index != -1:
            break
    
    if start_index != -1:
        end_keywords = ["education", "skills", "projects", "certifications"]
        end_index = len(resume_text)
        for keyword in end_keywords:
            temp_end_index = text_lower.find(keyword, start_index + 1)
            if temp_end_index != -1 and temp_end_index < end_index:
                end_index = temp_end_index
        
        experience_text = resume_text[start_index:end_index].strip()
        
    return {
        "skills": found_skills,
        "education": education,
        "experience_summary": experience_text
    }

# --- Matching Logic Function ---

def calculate_match_score(resume_text, job_description, job_keywords):
    """
    Calculates a match score between a resume and a job description.
    Combines TF-IDF cosine similarity with a keyword match percentage.
    """
    corpus = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    resume_lower = resume_text.lower()
    matched_keywords = [kw for kw in job_keywords if kw.lower() in resume_lower]
    keyword_match_percentage = (len(matched_keywords) / len(job_keywords)) * 100 if len(job_keywords) > 0 else 0
    
    overall_score = (cosine_sim * 0.7 + (keyword_match_percentage / 100) * 0.3) * 100
    
    return overall_score, matched_keywords

# --- LLM API Call Functions ---

def generate_tailored_resume_ollama(prompt_text):
    try:
        # Assumes Ollama is running and has the 'mistral' model pulled
        response = ollama.generate(model='mistral', prompt=prompt_text)
        return response['response']
    except Exception as e:
        st.error(f"Error calling local Ollama API. Make sure Ollama is running and the 'mistral' model is pulled: {e}")
        return None

# --- PDF Generation Function ---

def generate_pdf_from_text(resume_text, job_title="Tailored Resume"):
    """
    Generates a PDF from the given resume text using an HTML template.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ resume_title }}</title>
        <style>
            body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
            .header { text-align: center; margin-bottom: 20px; }
            .header h1 { margin: 0; font-size: 2em; }
            .section { margin-bottom: 20px; }
            .section h2 { border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-bottom: 10px; }
            ul { list-style-type: none; padding: 0; }
            li { margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{ resume_title }}</h1>
        </div>
        <div class="section">
            {{ resume_content | safe }}
        </div>
    </body>
    </html>
    """
    
    env = Environment(loader=FileSystemLoader('.'))
    template = env.from_string(html_template)
    
    # Simple replacement of newline characters with HTML breaks
    html_content = resume_text.replace('\n', '<br>')
    
    rendered_html = template.render(resume_title=job_title, resume_content=html_content)
    pdf = HTML(string=rendered_html).write_pdf()
    return pdf

# --- Tips Logic Function ---

def get_resume_improvement_tips(extracted_data, job_matches_sorted):
    """
    Generates a list of resume improvement tips based on analysis.
    """
    tips = []
    tips.append("1. **Quantify achievements:** Instead of 'Managed projects', try 'Managed 5 projects, reducing delivery time by 15%'.")
    tips.append("2. **Keywords:** Ensure your resume prominently features keywords from the job descriptions you're applying for.")
    tips.append("3. **Conciseness:** Keep bullet points concise and start them with action verbs.")
    tips.append("4. **Proofread:** Always double-check for typos and grammatical errors. A fresh pair of eyes helps!")
    tips.append("5. **Contact Info:** Ensure your contact details (email, phone, LinkedIn) are clear and up-to-date.")

    if job_matches_sorted:
        top_job = job_matches_sorted[0]
        if top_job['score'] < 50:
            tips.append(f"6. **Low Match Score:** Your resume has a low match ({top_job['score']:.2f}%) for the **{top_job['title']}** role. Consider a different job role or significantly tailoring your resume content.")
        
        job_keywords = set(JOB_DESCRIPTIONS[top_job['title']]['keywords'])
        resume_skills_lower = {s.lower() for s in extracted_data['skills']}
        missing_skills = [kw for kw in job_keywords if kw.lower() not in resume_skills_lower]

        if missing_skills:
            tips.append(f"7. **Skill Gap for '{top_job['title']}':** You might be missing some key skills for this role, such as: **{', '.join(missing_skills[:5])}** (and more). If you have these skills, make them more prominent on your resume!")
            tips.append(f"8. **Tailor for this role:** To improve your match for **{top_job['title']}**, tailor your experience and project bullet points to align more closely with the skills and requirements listed in its description.")

    if not extracted_data['skills']:
        tips.append("9. **Missing Skills Section:** We couldn't find a clear 'Skills' section. It's crucial to list your technical and soft skills to pass automated screening tools.")
    if extracted_data['experience_summary'] == "Not Found":
        tips.append("10. **Missing Experience:** Your resume appears to be missing a dedicated 'Experience' or 'Work History' section. This is a critical part of your resume.")
    if not extracted_data['education']:
        tips.append("11. **Missing Education:** Make sure your highest level of education and institution are clearly listed on your resume.")

    return tips

# --- Main Streamlit App UI ---

st.title("🚀 AI Resume Analyzer & Job Matcher")
st.markdown("---")
st.markdown(
    """
    ### 🧠 What This Project Does
    Upload your resume → Get instant job match % → Improve your resume using AI → Find tailored job suggestions.
    """
)
st.markdown("---")

st.subheader("Upload Your Resume")
uploaded_file = st.file_uploader(
    "Please upload a PDF or DOCX file:",
    type=["pdf", "docx"],
    help="The application will analyze this file to match you with job roles."
)

if uploaded_file is not None:
    upload_dir = "uploaded_resumes"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Resume '{uploaded_file.name}' uploaded successfully!")
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        st.stop()
    
    resume_text = parse_resume(file_path, uploaded_file.type)
    
    if resume_text:
        st.subheader("Extracted Resume Content (for debugging):")
        st.text(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))
        
        st.subheader("Analysis Results:")
        extracted_data = extract_info(resume_text)
        
        st.write(f"**Skills Found:** {', '.join(extracted_data['skills']) if extracted_data['skills'] else 'None'}")
        st.write(f"**Education Snippets:** {'; '.join(extracted_data['education']) if extracted_data['education'] else 'None'}")
        st.write(f"**Experience Summary:** {extracted_data['experience_summary'][:300] + '...' if len(extracted_data['experience_summary']) > 300 else extracted_data['experience_summary']}")
        
        st.subheader("Job Match Percentages:")
        job_matches = []
        for job_title, job_info in JOB_DESCRIPTIONS.items():
            score, matched_keywords = calculate_match_score(
                resume_text, 
                job_info['description'], 
                job_info['keywords']
            )
            job_matches.append({
                "title": job_title,
                "score": score,
                "matched_keywords": matched_keywords
            })

            st.markdown(f"**{job_title}:** {score:.2f}% Match")
            if matched_keywords:
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;*Matched Skills/Keywords:* {', '.join(matched_keywords)}")
            else:
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;*No specific keywords matched from the {job_title} description.*")

        job_matches_sorted = sorted(job_matches, key=lambda x: x['score'], reverse=True)
        
        st.subheader("Personalized Job Suggestions:")
        if job_matches_sorted:
            for job in job_matches_sorted[:3]:
                st.info(f"- **{job['title']}** (Match: {job['score']:.2f}%)")
        else:
            st.warning("No job matches found. Please upload a relevant resume.")

        st.subheader("AI-Powered Resume Improvement Tips:")
        with st.expander("Click here for personalized tips to improve your resume"):
            improvement_tips = get_resume_improvement_tips(extracted_data, job_matches_sorted)
            for tip in improvement_tips:
                st.markdown(f"- {tip}")
        
        # --- UI for the new 'Auto-Tailor Resume' feature ---
        st.markdown("---")
        st.subheader("✍️ Auto-Tailor Your Resume for a Specific Job")
        st.info("Paste a job description below and click 'Generate' to create a new resume tailored to that role.")
        
        job_description_input = st.text_area(
            "Paste the Full Job Description Here:",
            height=300,
            placeholder="e.g., 'As a Senior Software Engineer, you will be responsible for...'"
        )

        if st.button("Generate Tailored Resume", type="primary"):
            if not job_description_input:
                st.warning("Please paste a job description to get a tailored resume.")
            else:
                with st.spinner("🚀 Generating your personalized resume... this may take a moment."):
                    # Construct a detailed prompt for Ollama
                    prompt_template = f"""
                    You are an expert HR specialist and an ATS (Applicant Tracking System) optimization consultant.
                    Your task is to take the provided candidate's resume and a specific job description, and generate a new, tailored resume.

                    ### Instructions:
                    1.  **Analyze the job description:** Identify the most critical keywords, required skills, and key responsibilities.
                    2.  **Tailor the resume:** Rewrite the resume content to highlight the candidate's skills and experiences that are most relevant to the job description.
                    3.  **Focus on Quantifiable Achievements:** Rephrase bullet points from the original resume to start with strong action verbs and, where possible, add quantifiable metrics (e.g., "increased sales by 15%").
                    4.  **Rewrite the Summary:** Generate a professional summary at the top that is specifically tailored to this job role.
                    5.  **Integrity is key:** DO NOT invent new skills, work experience, or educational details that are not present in the original resume. Only use the information provided.
                    6.  **Maintain Structure:** Keep the original resume structure (e.g., sections for Contact, Summary, Experience, Education, Skills).
                    7.  **Final Output:** Present the final tailored resume as clean, formatted text.

                    ### Candidate's Original Resume:
                    {resume_text}

                    ### Target Job Description:
                    {job_description_input}

                    ### Tailored Resume:
                    """
                    tailored_resume_output = generate_tailored_resume_ollama(prompt_template)
                    
                    if tailored_resume_output:
                        # Calculate the ATS Score for the new resume
                        ats_score, ats_keywords = calculate_match_score(
                            tailored_resume_output,
                            job_description_input,
                            job_matches_sorted[0]['matched_keywords'] if job_matches_sorted else []
                        )
                        st.session_state.tailored_resume = tailored_resume_output
                        st.session_state.ats_score = ats_score
                        st.success(f"Generation complete! ATS Score: {ats_score:.2f}%")
                    else:
                        st.error("Failed to generate a tailored resume. Please try again.")

        st.markdown("---")
        if 'tailored_resume' in st.session_state and 'ats_score' in st.session_state:
            st.subheader(f"✨ Your Tailored Resume (ATS Score: {st.session_state.ats_score:.2f}%)")
            
            st.text_area(
                "Generated Resume Output",
                value=st.session_state.tailored_resume,
                height=500,
                key="tailored_output",
                help="Copy this content to update your resume!"
            )
            
            pdf_data = generate_pdf_from_text(st.session_state.tailored_resume, job_title="Tailored Resume")
            st.download_button(
                label="Download as PDF",
                data=pdf_data,
                file_name="tailored_resume.pdf",
                mime="application/pdf"
            )
        else:
            st.subheader("✨ Your Tailored Resume")
            st.text_area(
                "Generated Resume Output",
                value="[Your tailored resume will appear here]",
                height=500,
                key="tailored_output",
                help="Copy this content to update your resume!"
            )

        os.remove(file_path)
        if not os.listdir(upload_dir):
            os.rmdir(upload_dir)
        st.success("Resume text extracted and temporary file deleted.")
    else:
        st.warning("Could not extract text from resume.")
        if os.path.exists(file_path):
            os.remove(file_path)
            
else:
    st.info("Awaiting resume upload...")
    st.markdown("---")
    st.markdown(
        "**Note:** Your resume file will be temporarily stored for analysis and will be deleted after the process is complete."
    )