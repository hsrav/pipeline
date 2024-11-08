import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample job experience data (you can replace this with real data)
experience_data = [
    "Developed and maintained scalable web applications using Django and Flask. Led a team of 5 developers.",
    "Worked as a Data Analyst, analyzing data trends and building reports using SQL and Python. Developed dashboards.",
    "Managed project lifecycle from planning to deployment. Coordinated between clients and development teams.",
]

# Preprocess experience data: cleaning and tokenizing
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing to all experience entries
processed_experiences = [preprocess_text(exp) for exp in experience_data]

# Extract key skills/roles using CountVectorizer
vectorizer = CountVectorizer(max_features=10)
X = vectorizer.fit_transform(processed_experiences)
keywords = vectorizer.get_feature_names_out()

# Generate similarity matrix (optional, if you want to compare entries)
similarity_matrix = cosine_similarity(X)

# Summarize the experience (basic summarization for demonstration)
summary = {
    "total_experiences": len(experience_data),
    "skills_roles": keywords.tolist(),
    "similarity_matrix": similarity_matrix
}

# Build a sample resume
def build_resume(summary, name="John Doe", title="Software Engineer"):
    resume = f"Name: {name}\nTitle: {title}\n\nSummary of Experience:\n"
    resume += f"- Total Experiences: {summary['total_experiences']}\n"
    resume += f"- Key Skills and Roles: {', '.join(summary['skills_roles'])}\n"
    resume += "\nExperience Details:\n"
    
    for i, exp in enumerate(experience_data):
        resume += f"{i+1}. {exp}\n"
    
    return resume

# Generate the resume
resume = build_resume(summary)
print(resume)
