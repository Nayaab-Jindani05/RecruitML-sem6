from flask import Flask, request, render_template, jsonify
import pickle
import os
import re
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

# --- Flask Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# --- Load Model and Preprocessing Tools ---
with open("classifier_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return ''.join(page.get_text() for page in doc)

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_resume(text):
    # Extract name
    lines = text.splitlines()
    name = "N/A"
    for line in lines:
        if line.strip() and len(line.split()) <= 4:
            name = line.strip()
            break

    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "N/A"

    # Extract phone number
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,6}', text)
    phone = phone_match.group(0) if phone_match else "N/A"

    # Extract skills
    skills_list = ['python', 'java', 'c++', 'sql', 'excel', 'machine learning', 'flask', 'django', 'html', 'css', 'javascript']
    found_skills = [skill for skill in skills_list if skill.lower() in text.lower()]

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": found_skills
    }

# --- Routes ---
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/recommender", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("resume")

        if not file or file.filename == "":
            return render_template("resume.html", error="Please upload a PDF or TXT resume.")

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                ext = filename.rsplit('.', 1)[1].lower()
                raw_text = extract_text_from_pdf(filepath) if ext == "pdf" else extract_text_from_txt(filepath)

                cleaned = clean_text(raw_text)
                vector = tfidf.transform([cleaned])
                prediction = model.predict(vector)
                predicted_role = label_encoder.inverse_transform(prediction)[0]

                return render_template("resume.html", prediction=predicted_role, resume_text=raw_text)

            except Exception as e:
                return render_template("resume.html", error=f"Error processing resume: {str(e)}")

        else:
            return render_template("resume.html", error="Allowed file types: PDF, TXT.")

    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def analyze_resume():
    file = request.files.get("resume")
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type."})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print("✅ File saved successfully:", filepath)

    # Extract text
    ext = filename.rsplit('.', 1)[1].lower()
    raw_text = extract_text_from_pdf(filepath) if ext == "pdf" else extract_text_from_txt(filepath)
    print("📄 Raw text extracted:", raw_text[:300])

    # Clean the text
    cleaned = clean_text(raw_text)
    print("🧼 Cleaned text:", cleaned[:300])

    try:
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)
        predicted_role = label_encoder.inverse_transform(prediction)[0]
        print("🎯 Predicted job role:", predicted_role)
    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"success": False, "error": "Prediction failed."})

    try:
        parsed_data = parse_resume(raw_text)
        print("📋 Parsed data:", parsed_data)
    except Exception as e:
        print("❌ Error during parsing:", e)
        parsed_data = {"name": "N/A", "email": "N/A", "phone": "N/A", "skills": []}

    response = {
        "success": True,
        "category": "Software Engineering",  # You can make this dynamic later
        "job": predicted_role,
        "name": parsed_data["name"],
        "email": parsed_data["email"],
        "phone": parsed_data["phone"],
        "skills": parsed_data["skills"],
        "text": raw_text
    }

    return jsonify(response)

@app.route('/screener', methods=["GET", "POST"])
def screener():
    if request.method == "POST":
        job_desc = clean_text(request.form.get("job_description", ""))

        resume_texts = []
        resume_names = []

        for file in request.files.getlist("resumes"):
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                ext = filename.rsplit('.', 1)[1].lower()
                text = extract_text_from_pdf(filepath) if ext == "pdf" else extract_text_from_txt(filepath)

                resume_texts.append(clean_text(text))
                resume_names.append(filename)

        all_docs = resume_texts + [job_desc]
        vectors = tfidf.transform(all_docs)
        job_vector = vectors[-1]
        resume_vectors = vectors[:-1]

        scores = cosine_similarity(job_vector, resume_vectors).flatten()
        results = [{'name': name, 'score': round(float(score) * 100, 2)} for name, score in zip(resume_names, scores)]
        results.sort(key=lambda x: x['score'], reverse=True)

        return render_template("resume_screener.html", results=results)

    return render_template("resume_screener.html")

if __name__ == "__main__":
    app.run(debug=True)
