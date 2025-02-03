from flask import Flask, render_template, request, jsonify
import pickle
import os
import spacy
from fuzzywuzzy import process

app = Flask(__name__)

# Load trained model and vectorizer
with open("models/skills_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/job_roles_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/valid_job_roles.pkl", "rb") as f:
    valid_job_roles = pickle.load(f)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Normalize job titles by removing stopwords, lemmatizing, and lowercasing."""
    doc = nlp(text.lower().strip())  # Convert to lowercase and process with spaCy
    return " ".join([token.lemma_ for token in doc if not token.is_stop])  # Remove stopwords & lemmatize

def find_best_match(user_input, job_titles):
    """Find the closest job title from the dataset using fuzzy matching."""
    best_match, score = process.extractOne(user_input, job_titles)
    return best_match if score > 80 else None  # Return match only if similarity >80%

@app.route('/')
def home():
    return render_template('index.html')  # Load the web page

@app.route('/predict', methods=['POST'])
def predict():
    job_title = request.form['job_title'].strip().lower()  # Normalize user input
    processed_title = preprocess(job_title)  # Use NLU to process input

    # Check if job role exists in dataset
    if processed_title not in valid_job_roles:
        best_match = find_best_match(processed_title, valid_job_roles)
        if best_match:
            return jsonify({'error': f"❌ Job not found. Did you mean '{best_match}'?"})
        return jsonify({'error': "❌ Invalid job role. Please enter a valid job title from the dataset."})

    # Transform the job title using the vectorizer
    job_title_vector = vectorizer.transform([job_title])

    # Get probabilities for each skill
    probabilities = model.predict_proba(job_title_vector)[0]

    # Set probability threshold to filter relevant skills
    threshold = 0.05  
    relevant_skills = [skill for skill, prob in zip(model.classes_, probabilities) if prob > threshold]

    return jsonify({'skills': relevant_skills})

if __name__ == '__main__':
    app.run(debug=True)
