# 🤖 Smart Aid-Matching Bot — Personalized Welfare Scheme Finder

## 📌 Project Overview
The Smart Aid-Matching Bot is an AI-powered recommendation system designed to help users discover **relevant government welfare schemes** based on their personal profile and needs.

The system uses **Natural Language Processing (NLP), semantic similarity, and rule-based filtering** to match users with suitable schemes. It combines user inputs such as age, gender, income, state, and preferences with intelligent ranking techniques to deliver personalized recommendations.

This project demonstrates an end-to-end intelligent system integrating **data processing, machine learning, and interactive UI**.

---

## 🎯 Objectives
- Provide personalized welfare scheme recommendations  
- Filter schemes based on user profile (age, gender, income, state)  
- Rank schemes using semantic similarity techniques  
- Improve accessibility to government benefits  
- Deliver explainable recommendations with reasoning  

---

## 🛠 Tools & Technologies
- **Frontend/UI** – Streamlit  
- **Programming Language** – Python  
- **NLP** – TF-IDF, SBERT (optional)  
- **Libraries** – pandas, numpy, scikit-learn, nltk  
- **Similarity** – Cosine Similarity  
- **Data Storage** – CSV dataset  

---

## 📂 Dataset
Dataset: Government Welfare Schemes Dataset  

The dataset includes:
- Scheme name  
- Eligibility criteria  
- Benefits and details  
- Application process  
- Required documents  
- Category and tags  

---

## 📁 Repository Structure
- `app.py` – Main Streamlit application  
- `cleaned_schemes.csv` – Processed dataset  
- `vectorizer.pkl` – TF-IDF model  
- `embeddings_tfidf.npz` – Precomputed embeddings  
- `README.md` – Project documentation  

---

## 🧠 System Approach

### Data Processing
- Text cleaning and normalization  
- Tokenization and stemming using NLTK  
- Feature extraction using TF-IDF  
- Optional SBERT embeddings for advanced semantic understanding  

### Filtering Logic
- Age-based eligibility parsing  
- Gender-based filtering  
- Income-based filtering  
- State and category matching  

### Ranking Strategy
- Semantic similarity (TF-IDF / SBERT)  
- Keyword-based rule scoring  
- Filter match bonus  

Final Score:
- 60% Semantic similarity  
- 30% Keyword match  
- 10% Filter relevance  

---

## 📊 Key Features
- Personalized scheme recommendations  
- Advanced filtering (age, gender, income, state, category)  
- Semantic search using NLP  
- Explainable AI (shows why a scheme is recommended)  
- Real-time interactive UI  
- Highlighted keyword matching  

---

## 🧠 Recommendation Logic

1. User inputs profile and preferences  
2. System filters relevant schemes  
3. Converts user query into vector form  
4. Computes similarity with scheme data  
5. Applies rule-based scoring  
6. Combines scores for final ranking  
7. Displays top recommended schemes  

---

## 🎯 Use Cases
- 👨‍👩‍👧 General Public — Discover eligible schemes  
- 🏛️ Government Platforms — Improve accessibility  
- 🎓 Students — Learn NLP and recommendation systems  
- 📊 Researchers — Analyze welfare distribution  

---

## 🚀 Business Impact
- Improves awareness of welfare schemes  
- Enhances accessibility for citizens  
- Reduces information gap  
- Enables data-driven recommendations  

---

## 💼 Professional Value
This project demonstrates:

- NLP and semantic search implementation  
- Recommendation system design  
- Feature engineering and ranking logic  
- Streamlit-based application development  
- Real-world problem solving using AI  

---

## ⚠️ Disclaimer
This project is for **educational and informational purposes only**.  

Users should verify scheme details from official government sources before applying.

---

## ⭐ Tagline
**Connecting People to the Right Benefits — Smart, Fast, Personalized.**
