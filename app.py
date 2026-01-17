# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Optional SBERT import (if available)
USE_SBERT = False
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except Exception:
    USE_SBERT = False

# ---------------------------
# Files & config
# ---------------------------
CLEANED_CSV = "cleaned_schemes.csv"      # your dataset file
EMBEDDINGS_TFIDF = "embeddings_tfidf.npz"
VECTORIZER_PKL = "vectorizer.pkl"

# Weights for final ranking
WEIGHT_SEMANTIC = 0.6
WEIGHT_RULE = 0.3
WEIGHT_FILTER_MATCH_BONUS = 0.1

# ---------------------------
# Text helpers
# ---------------------------
tokenizer = RegexpTokenizer(r"\w+")
ps = PorterStemmer()

def clean_text(s):
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stem_text(s):
    s = clean_text(s)
    tokens = tokenizer.tokenize(s)
    stems = [ps.stem(t) for t in tokens if len(t) > 1]
    return " ".join(stems)

def highlight_matches(text, keywords):
    # Return HTML-marked snippet with keywords bolded (safe small HTML)
    if not isinstance(text, str) or text == "":
        return ""
    out = clean_text(text)
    for kw in sorted(set(keywords), key=len, reverse=True):
        if not kw: continue
        # simple word-boundary replacement
        out = re.sub(rf'(\b{re.escape(kw)}\b)', r'<strong>\1</strong>', out, flags=re.IGNORECASE)
    # Show first 400 chars
    if len(out) > 400:
        out = out[:400] + "..."
    return out

# ---------------------------
# Load data
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(CLEANED_CSV):
        st.error(f"{CLEANED_CSV} not found. Place it in the same folder as app.py.")
        st.stop()
    df = pd.read_csv(CLEANED_CSV)
    # Ensure expected columns exist (use empty strings if missing)
    expected = ["scheme_name","details","benefits","eligibility","application",
                "documents","level","schemeCategory","tags","combined_text",
                "combined_text_clean","scheme_name_clean","tokens_stemmed"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    # If tokens_stemmed is empty, populate from combined_text_clean
    if df['tokens_stemmed'].isnull().all() or df['tokens_stemmed'].eq("").all():
        df['tokens_stemmed'] = df['combined_text_clean'].fillna(df['combined_text']).apply(lambda x: stem_text(x))
    # Normalize combined_text_clean
    df['combined_text_clean'] = df['combined_text_clean'].fillna(df['combined_text']).apply(clean_text)
    # Keep an index column for later mapping
    df = df.reset_index(drop=False).rename(columns={'index':'orig_index'})
    return df

# ---------------------------
# TF-IDF & SBERT loaders
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_or_generate_tfidf(df):
    if os.path.exists(EMBEDDINGS_TFIDF) and os.path.exists(VECTORIZER_PKL):
        X = sparse.load_npz(EMBEDDINGS_TFIDF)
        with open(VECTORIZER_PKL, "rb") as f:
            vect = pickle.load(f)
        return X, vect
    # generate
    vect = TfidfVectorizer(max_features=60000, ngram_range=(1,2))
    X = vect.fit_transform(df['tokens_stemmed'])
    sparse.save_npz(EMBEDDINGS_TFIDF, X)
    with open(VECTORIZER_PKL, "wb") as f:
        pickle.dump(vect, f)
    return X, vect

@st.cache_resource(show_spinner=False)
def load_sbert_model():
    if USE_SBERT:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    return None

# ---------------------------
# Robust filters
# ---------------------------
def apply_filters(df, age=None, gender=None, income=None, state=None, category=None):
    filtered = df.copy()

    # Gender: match flexible terms
    if gender and gender.lower() != "any":
        gender_map = {
            "male": ["male","man","men","boy"],
            "female": ["female","woman","women","girl","widow"],
            "other": ["transgender","non-binary","other"]
        }
        kws = gender_map.get(gender.lower(), [gender.lower()])
        filtered = filtered[filtered['tokens_stemmed'].apply(lambda t: any(k in t for k in kws) or "any" in t)]

    # State: look in combined_text_clean or application details
    if state and state.strip():
        s = state.strip().lower()
        filtered = filtered[filtered['combined_text_clean'].apply(lambda t: s in t or "all" in t)]

    # Category: match schemeCategory or tags or combined_text_clean
    if category and category.strip().lower() != "any":
        c = category.strip().lower()
        filtered = filtered[
            filtered['schemeCategory'].apply(lambda x: c in str(x).lower()) |
            filtered['tags'].apply(lambda x: c in str(x).lower()) |
            filtered['combined_text_clean'].apply(lambda x: c in str(x).lower())
        ]

    # Age: parse eligibility (many formats)
    if age is not None:
        def age_ok(row):
            ar = str(row.get('eligibility',"")).lower()
            if ar == "" or "all" in ar:
                return True
            # patterns like "60+", "18-60", "18 years and above"
            m_plus = re.findall(r'(\d{1,3})\s*\+', ar)
            if m_plus:
                try:
                    low = int(m_plus[0])
                    return age >= low
                except:
                    return True
            m_range = re.findall(r'(\d{1,3})\s*-\s*(\d{1,3})', ar)
            if m_range:
                try:
                    low, high = int(m_range[0][0]), int(m_range[0][1])
                    return low <= age <= high
                except:
                    return True
            m_above = re.findall(r'(\d{1,3}).{0,10}above', ar)
            if m_above:
                try:
                    return age >= int(m_above[0])
                except:
                    return True
            return True
        filtered = filtered[filtered.apply(age_ok, axis=1)]

    # Income: try to detect numeric caps; default is keep if not present
    if income is not None and income > 0:
        def income_ok(row):
            text = str(row.get('combined_text_clean',"")).lower().replace(",","")
            if "any" in text or text == "":
                return True
            nums = re.findall(r'(\d{3,})', text)
            if not nums:
                return True
            try:
                min_val = min(int(n) for n in nums)
                return income <= min_val
            except:
                return True
        filtered = filtered[filtered.apply(income_ok, axis=1)]

    return filtered

# ---------------------------
# Rule-based scoring and match explanation
# ---------------------------
def compute_rule_score(row, user_keywords):
    if not user_keywords:
        return 0
    text = str(row.get('tokens_stemmed',""))
    return sum(1 for kw in user_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text))

def compute_filter_match_bonus(row, gender, state, category):
    bonus = 0.0
    # add small bonus if any filter term appears in row text (means filter is good match)
    txt = str(row.get('combined_text_clean',""))
    if gender and gender.lower() != "any" and gender.lower() in txt:
        bonus += 0.05
    if state and state.strip() and state.strip().lower() in txt:
        bonus += 0.03
    if category and category.strip().lower() != "any" and category.strip().lower() in str(row.get('schemeCategory',"")).lower():
        bonus += 0.02
    return bonus

# ---------------------------
# Semantic similarity ranking
# ---------------------------
def semantic_rank_and_score(filtered_df, user_text, top_k, tfidf_matrix, tfidf_vectorizer, sbert_model, user_keywords, gender, state, category):
    if filtered_df.empty:
        return filtered_df

    # Semantic similarity
    if USE_SBERT and sbert_model:
        # SBERT path
        doc_texts = filtered_df['combined_text_clean'].tolist()
        doc_embs = sbert_model.encode(doc_texts, convert_to_numpy=True)
        user_emb = sbert_model.encode([user_text], convert_to_numpy=True)
        sem_sims = cosine_similarity(user_emb, doc_embs)[0]
        filtered_df = filtered_df.copy()
        filtered_df['sim_score'] = sem_sims
    else:
        # TF-IDF path
        if tfidf_vectorizer is None or tfidf_matrix is None:
            st.error("TF-IDF embeddings missing. Please ensure preprocessing ran correctly.")
            return filtered_df
        user_vec = tfidf_vectorizer.transform([stem_text(clean_text(user_text))])
        idxs = filtered_df['orig_index'].to_numpy()
        submat = tfidf_matrix[idxs]
        sims = cosine_similarity(user_vec, submat)[0]
        filtered_df = filtered_df.copy()
        filtered_df['sim_score'] = sims

    # Rule-based and combined score
    filtered_df['rule_score'] = filtered_df.apply(lambda r: compute_rule_score(r, user_keywords), axis=1)
    filtered_df['filter_bonus'] = filtered_df.apply(lambda r: compute_filter_match_bonus(r, gender, state, category), axis=1)

    # Normalize rule_score to [0,1] scale (if needed)
    max_rule = filtered_df['rule_score'].max()
    if max_rule and max_rule > 0:
        filtered_df['rule_score_norm'] = filtered_df['rule_score'] / max_rule
    else:
        filtered_df['rule_score_norm'] = 0.0

    # final combined score
    filtered_df['combined_score'] = (
        WEIGHT_SEMANTIC * filtered_df['sim_score'] +
        WEIGHT_RULE * filtered_df['rule_score_norm'] +
        WEIGHT_FILTER_MATCH_BONUS * filtered_df['filter_bonus']
    )

    return filtered_df.sort_values('combined_score', ascending=False).head(top_k)

# ---------------------------
# UI / main
# ---------------------------
st.set_page_config(page_title="SMART AID-MATCHING BOT ", layout="wide")
st.title("SMART AID-MATCHING BOT  — Personalized Scheme Finder")

# Load resources
df = load_data()
tfidf_matrix, tfidf_vectorizer = load_or_generate_tfidf(df)
sbert_model = load_sbert_model()

# Sidebar: user profile + intent
with st.sidebar:
    st.header("Profile & Preferences")
    name = st.text_input("Name", "")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Any", "Male", "Female", "Other"])
    state = st.text_input("State (e.g., All / Tamil Nadu)", value="All")
    income = st.number_input("Annual income (₹)", min_value=0, value=0, step=1000)
    category = st.text_input("Scheme Category (or 'Any')", value="Any")
    st.markdown("---")
    st.markdown("What are you looking for? (keywords and a short description help)")
    keywords_input = st.text_input("Keywords (comma-separated)", help="e.g. farmer, training, small business")
    user_search_text = st.text_area("Describe your need (one or two sentences)", height=120)
    st.markdown("---")
    st.write("Model status:")
    if USE_SBERT:
        st.success("SBERT available — semantic ranking uses SBERT embeddings")
    else:
        st.info("SBERT not available — using TF-IDF fallback")

# Apply filters (robust) and show preview always
filtered = apply_filters(df, age=age, gender=gender, income=income, state=state, category=category)
st.subheader(f"Filtered schemes: {len(filtered)}")
if len(filtered) == 0:
    st.warning("No schemes matched your filters. Try relaxing state/category/gender or set gender to 'Any'.")
# Show small preview to help user understand filter effect
st.dataframe(filtered[['scheme_name','schemeCategory','eligibility']].head(20))

# Build keywords and user text
user_keywords = [ps.stem(k.strip().lower()) for k in keywords_input.split(",") if k.strip()]
if not str(user_search_text).strip():
    # construct a helpful default query based on profile
    user_search_text = " ".join([str(name), str(gender), str(category), str(state), str(age), str(income)] + user_keywords)

# Rank top-k
top_k = st.slider("Top results to show", min_value=3, max_value=30, value=8)
ranked = semantic_rank_and_score(filtered, user_search_text, top_k, tfidf_matrix, tfidf_vectorizer, sbert_model, user_keywords, gender, state, category)

# Display final results with explanation
st.subheader("Recommended Schemes (personalized)")
if ranked.empty:
    st.info("No ranked schemes to show. If you see filtered results above, try adding keywords or relaxing filters.")
else:
    for i, row in ranked.reset_index().iterrows():
        scheme = row.get('scheme_name') or "Untitled scheme"
        score = float(row.get('combined_score', 0))
        sim = float(row.get('sim_score', 0))
        rule_score = int(row.get('rule_score', 0))
        eligibility = row.get('eligibility', '')
        docs = row.get('documents', '') or row.get('application', '') or ""
        details = row.get('details', '') or row.get('combined_text_clean', '')
        scheme_cat = row.get('schemeCategory', '')
        link = row.get('application', '')  # try application link if present

        with st.expander(f"{scheme}  —  Score: {score:.3f}"):
            st.markdown(f"**Category:** {scheme_cat}")
            st.markdown(f"**Eligibility (raw):** {eligibility}")
            # highlight matched keywords in the snippet (safe small HTML)
            snippet_html = highlight_matches(details, user_keywords + [k.strip().lower() for k in keywords_input.split(",") if k.strip()])
            if snippet_html:
                st.markdown("**Description (snippet):**")
                st.markdown(snippet_html, unsafe_allow_html=True)
            else:
                st.markdown("**Description:**")
                st.write(details[:800] + ("..." if len(details) > 800 else ""))

            # Reasoning / explanation
            st.markdown("**Why this was recommended:**")
            reasons = []
            reasons.append(f"- Semantic similarity score: {sim:.3f}")
            if rule_score > 0:
                reasons.append(f"- Matched keywords: {rule_score} rule matches")
            if row.get('filter_bonus', 0) > 0:
                reasons.append(f"- Good filter match (gender/state/category) → bonus {row.get('filter_bonus'):.2f}")
            st.markdown("\n".join(reasons))

            # Documents & link
            if docs and str(docs).strip():
                st.markdown(f"**Required documents / notes:** {docs}")
            if link and str(link).strip():
                st.markdown(f"[Apply / More info]({link})")

            # Small metadata
            st.markdown(f"*Internal score (semantic {WEIGHT_SEMANTIC}, rule {WEIGHT_RULE}, filter-bonus {WEIGHT_FILTER_MATCH_BONUS})*")

# Footer quick tips
st.markdown("---")
st.markdown("**Tips:** Try adding short keywords (e.g., 'training', 'loan', 'widow', 'farmer') or making Gender = Any to see more results.")
