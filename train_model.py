import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import spacy

# Load dataset
data_path = "data/jobskills.csv"
data = pd.read_csv(data_path)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Normalize job titles by removing stopwords, lemmatizing, and lowercasing."""
    doc = nlp(text.lower().strip())  # Convert to lowercase and process with spaCy
    return " ".join([token.lemma_ for token in doc if not token.is_stop])  # Remove stopwords & lemmatize

# Normalize job roles using NLU
data['processed_job_role'] = data['job_role'].apply(preprocess)

# Store valid job roles for input validation
valid_job_roles = set(data['processed_job_role'].unique())  

# Vectorize the 'processed_job_role' column
vectorizer = CountVectorizer()
job_roles_vector = vectorizer.fit_transform(data['processed_job_role'])

# Define features (X) and labels (y)
X = job_roles_vector
y = data['skills']

# Train the model
model = OneVsRestClassifier(RandomForestClassifier())
model.fit(X, y)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save the trained model, vectorizer, and valid job roles
with open("models/skills_recommendation_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/job_roles_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/valid_job_roles.pkl", "wb") as f:
    pickle.dump(valid_job_roles, f)

print("✅ Model, vectorizer, and valid job roles saved successfully!")
