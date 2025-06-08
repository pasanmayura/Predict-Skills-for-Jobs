# 🔍 Job Skill Predictor

This app helps you find the **key skills** needed for any **job role**. Great for students and job seekers! 💼✨

---

## ✅ What You Can Do

- 🔍 Enter a job title  
- 📋 See recommended skills  
- 🧠 Understand similar or misspelled job titles  

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS, Bootstrap, jQuery  
- **Backend**: Python (Flask)  
- **ML**: Scikit-learn  
- **NLP**: spaCy  
- **Fuzzy Matching**: FuzzyWuzzy  

---

## 📁 Folder Structure
```bash
project/
├── app.py # Web app entry point
├── train_model.py # Train the model
├── generate_embeddings.py # Create embeddings for job roles
├── test.py # Test model performance
│
├── data/
│ └── jobskills.csv # Dataset used for training
│
├── models/ # Saved models and embeddings
│ ├── skills_recommendation_model.pkl
│ ├── job_roles_vectorizer.pkl
│ ├── valid_job_roles.pkl
│ └── job_role_embeddings.pkl
│
├── templates/
│ └── index.html # Frontend page
│
├── static/
│ └── style.css # Page styling
```
---

## 📦 How to Run

🧪 **Step-by-step:**

1. **Install required packages**:
   ```bash
   pip install flask scikit-learn spacy fuzzywuzzy numpy
   python -m spacy download en_core_web_md
   ```

2. **Place your dataset in the data/ folder as**:
    ```bash
    data/jobskills.csv
    ```
3. **Train the model**:
    ```bash
    python train_model.py
    ```
3. **Generate embeddings for job titles**:
    ```bash
    python generate_embeddings.py
    ```
4. **Start the web app**:
    ```bash
    python app.py
    ```
5. **Open your browser and visit**:
    ```bash
    http://127.0.0.1:5000/
    ```
6. **(Optional) Test model performance**:
    ```bash
    python test.py
    ```
---

## 🧪 Sample Job Titles to Try

    - Software Engineer

    - Backend Developer

    - Data Scientist

    - Cloud Engineer

    - Mobile App Developer

---

## 📊 Test Output

**You will see**:

    - ✅ Accuracy

    - 📏 Precision

    - 📈 Recall

    - 🏆 F1 Score