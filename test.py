from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load trained model & vectorizer
with open("models/skills_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/job_roles_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Sample test data (Job title → Correct Skills)
test_data = {
    "Software Engineer": ["Problem-Solving", "Object-Oriented Programming", "Data Structures and Algorithms", "Version Control (Git)", "Test-driven Development (TDD)"],
    "Software Developer": ["Programming (Java, C++, Python)", "Problem-solving", "Version control (Git)", "Databases (SQL, NoSQL)"],
    "Cloud Architect": ["Infrastructure as Code (Terraform, CloudFormation)", "System Design", "Networking", "Cloud Security", "Cloud Platforms (AWS, Azure, Google Cloud)"]
}

# Prepare test data for evaluation
y_true = []  # True skills
y_pred = []  # Predicted skills

threshold = 0.05  # Probability threshold for selecting skills

print("\n🔍 **Testing Model on Sample Job Titles:**\n")

for job_title, actual_skills in test_data.items():
    print(f"📌 **Job Title:** {job_title}")

    # Convert job title to vector
    job_vector = vectorizer.transform([job_title])

    # Get model predictions
    predicted_probs = model.predict_proba(job_vector)[0]

    # Select skills with high probability
    predicted_skills = [skill for skill, prob in zip(model.classes_, predicted_probs) if prob > threshold]

    print(f"   🎯 **Actual Skills:** {actual_skills}")
    print(f"   🤖 **Predicted Skills:** {predicted_skills}\n")

    # Store results for evaluation
    y_true.extend([1 if skill in actual_skills else 0 for skill in model.classes_])
    y_pred.extend([1 if skill in predicted_skills else 0 for skill in model.classes_])

# Compute Evaluation Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)

# Print evaluation results
print("📊 **Model Evaluation Results:**")
print(f"✅ **Accuracy:** {accuracy:.2f}")
print(f"✅ **Precision:** {precision:.2f}")
print(f"✅ **Recall:** {recall:.2f}")
print(f"✅ **F1 Score:** {f1:.2f}")
