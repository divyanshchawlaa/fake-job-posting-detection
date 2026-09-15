import re
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "svm_tfidf_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"


# --------------------------------------------------
# NLP resources
# --------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# --------------------------------------------------
# Fraud risk indicators
# --------------------------------------------------

def detect_risk_indicators(text):
    text_lower = text.lower()

    indicators = []

    checks = [
        (
            r"\b(pay|payment|fee|deposit|registration fee|processing fee)\b",
            "Upfront payment or fee language detected"
        ),
        (
            r"\b(bank account|bank details|credit card|debit card|account number)\b",
            "Financial information request detected"
        ),
        (
            r"\b(password|social security|passport|personal information|identity)\b",
            "Sensitive personal information request detected"
        ),
        (
            r"\b(guaranteed income|guaranteed salary|guaranteed earnings)\b",
            "Guaranteed income claim detected"
        ),
        (
            r"\b(make|earn)\s+\$?\d{1,3}(?:,\d{3})*(?:\s*(?:per|a)\s*(?:week|day|hour))?\b",
            "High-income claim detected"
        ),
        (
            r"\b(no experience|no experience required|no qualifications required)\b",
            "No-experience requirement detected"
        ),
        (
            r"\b(act now|apply now|limited time|limited positions|immediately|urgent)\b",
            "Urgency or pressure language detected"
        ),
        (
            r"\b(send money|wire transfer|western union|gift card|crypto)\b",
            "Unusual payment method detected"
        )
    ]

    for pattern, message in checks:
        if re.search(pattern, text_lower):
            indicators.append(message)

    return indicators


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()


# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake Job Posting Detector")

st.write(
    "Analyse a job posting using a machine-learning model trained "
    "to identify potentially fraudulent job advertisements."
)

st.info(
    "Paste the job title, company information, description, "
    "requirements and benefits below."
)

job_text = st.text_area(
    "Job Posting",
    placeholder="Paste the complete job posting here...",
    height=350
)


# --------------------------------------------------
# Analysis
# --------------------------------------------------

if st.button("Analyze Job Posting", type="primary"):

    if not job_text.strip():
        st.warning("Please enter a job posting first.")

    else:
        # ML prediction
        cleaned_text = preprocess_text(job_text)
        text_features = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_features)[0]

        # Additional risk indicators
        indicators = detect_risk_indicators(job_text)

        # --------------------------------------------------
        # ML result
        # --------------------------------------------------

        if prediction == 1:
            st.error("⚠️ Likely Fraudulent Job Posting")
        else:
            st.success("✅ Likely Real Job Posting")

        # --------------------------------------------------
        # Risk indicators
        # --------------------------------------------------

        st.subheader("Risk Indicators")

        if indicators:
            st.warning(
                f"{len(indicators)} potential risk indicator(s) detected."
            )

            for indicator in indicators:
                st.write(f"⚠️ {indicator}")

        else:
            st.success("No predefined risk indicators detected.")

        # --------------------------------------------------
        # Overall assessment
        # --------------------------------------------------

        st.subheader("Assessment")

        if prediction == 1 and indicators:
            st.error(
                "The machine-learning model identified the posting as "
                "potentially fraudulent and additional risk indicators "
                "were detected."
            )

        elif prediction == 1:
            st.warning(
                "The machine-learning model identified the posting as "
                "potentially fraudulent."
            )

        elif prediction == 0 and indicators:
            st.warning(
                "The machine-learning model classified the posting as "
                "likely real, but some potential risk indicators were "
                "detected. Manual review is recommended."
            )

        else:
            st.success(
                "The posting was classified as likely real and no "
                "predefined risk indicators were detected."
            )

        st.caption(
            "The risk indicators are rule-based checks and are provided "
            "as additional supporting information. The main prediction "
            "comes from the trained Linear SVM model."
        )