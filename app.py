from flask import Flask, render_template, request, jsonify
import pickle
import spacy
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load trained model and vectorizer
with open("models/skills_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/job_roles_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/valid_job_roles.pkl", "rb") as f:
    valid_job_roles = pickle.load(f)  # List of valid job titles

# Load spaCy model (medium model for better embeddings)
nlp = spacy.load("en_core_web_md")

# Precompute job role embeddings
job_embeddings = {job: nlp(job).vector for job in valid_job_roles}

def preprocess(text):
    """Normalize text (lowercase & strip spaces)."""
    return text.lower().strip()

def find_best_match(user_input, job_titles):
    """Find closest job title using fuzzy matching."""
    best_match, score = process.extractOne(user_input, job_titles)
    return best_match, score  # Return both match & similarity score

def find_top_n_closest_embeddings(input_title, top_n=3):
    """Find top N closest job titles using word embeddings."""
    input_vector = nlp(input_title).vector.reshape(1, -1)
    similarities = {job: cosine_similarity(input_vector, vec.reshape(1, -1))[0][0] for job, vec in job_embeddings.items()}
    
    # Sort by similarity score
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(match[0], match[1]) for match in top_matches if match[1] > 0.5]  # Lower threshold to 0.5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    job_title = request.form['job_title'].strip()
    processed_title = preprocess(job_title)

    # Check if job exists in dataset
    if processed_title not in valid_job_roles:
        fuzzy_match, fuzzy_score = find_best_match(processed_title, valid_job_roles)
        semantic_matches = find_top_n_closest_embeddings(processed_title, top_n=3)

        print(f"DEBUG: Fuzzy Match - {fuzzy_match}")
        print(f"DEBUG: Semantic Matches - {semantic_matches}")

        # Decide best match approach
        if fuzzy_score >= 80:
            return jsonify({'error': f"❌ Job not found. Did you mean '{fuzzy_match}'?"})
        elif semantic_matches:
            match_list = ", ".join([match[0] for match in semantic_matches])
            return jsonify({'error': f"❌ Job not found. Closest matches: {match_list}"})
        
        return jsonify({'error': "❌ Invalid job role. Please enter a valid job title."})

    # Transform job title using vectorizer
    job_title_vector = vectorizer.transform([job_title])

    # Get skill probabilities
    probabilities = model.predict_proba(job_title_vector)[0]

    # Filter relevant skills (Threshold: 0.05)
    threshold = 0.05  
    relevant_skills = [skill for skill, prob in zip(model.classes_, probabilities) if prob > threshold]

    return jsonify({'skills': relevant_skills})

if __name__ == '__main__':
    app.run(debug=True)
