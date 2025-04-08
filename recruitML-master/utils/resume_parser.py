import re

# Basic list of common skills (you can expand this)
COMMON_SKILLS = [
    "python", "java", "c++", "sql", "javascript", "html", "css", "machine learning",
    "data science", "deep learning", "flask", "django", "react", "node.js",
    "git", "linux", "aws", "azure", "docker", "kubernetes", "nlp"
]

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    return match.group(0) if match else None

def extract_name(text):
    # Naive name extractor: first line with 2 capitalized words
    lines = text.strip().split("\n")
    for line in lines:
        words = line.strip().split()
        if len(words) >= 2 and all(w.istitle() for w in words[:2]):
            return " ".join(words[:2])
    return None

def extract_skills(text):
    found_skills = []
    text = text.lower()
    for skill in COMMON_SKILLS:
        if skill.lower() in text:
            found_skills.append(skill)
    return list(set(found_skills))  # Remove duplicates

def parse_resume(text):
    """Returns a dictionary of extracted info from resume text."""
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text)
    }
