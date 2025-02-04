import spacy
import pickle

# Load spaCy's medium or large model (better for word similarity)
nlp = spacy.load("en_core_web_md")

# Load job roles
with open("models/valid_job_roles.pkl", "rb") as f:
    valid_job_roles = pickle.load(f)

# Compute word embeddings for each job title
job_embeddings = {role: nlp(role).vector for role in valid_job_roles}

# Save embeddings to a file
with open("models/job_role_embeddings.pkl", "wb") as f:
    pickle.dump(job_embeddings, f)

print("✅ Job role embeddings saved successfully!")
