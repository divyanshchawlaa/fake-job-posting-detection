# AI-Based Fake Job Posting Detection

An AI application that uses Natural Language Processing (NLP) and Machine Learning to identify potentially fraudulent job postings.

## Project Overview

Fake job advertisements are a problem for online recruitment platforms. Fraudulent postings can be used to collect personal information, request payments, or mislead job seekers. At the same time, manually checking a large number of job postings can be time-consuming.

This project develops a machine-learning based system that analyses the text of a job posting and predicts whether it is likely to be **Real** or **Fraudulent**.

The system uses Natural Language Processing to clean and prepare job-posting text, TF-IDF to convert the text into numerical features, and several machine-learning models to compare classification performance.

The final model selected for the application is a **Linear Support Vector Machine (SVM) with TF-IDF features**.

---

## Business Problem

The main business problem addressed by this project is the detection of fraudulent job advertisements on online recruitment platforms.

A recruitment platform needs to identify suspicious postings without incorrectly flagging legitimate jobs. An automated detection system could help:

- Reduce the amount of manual moderation required.
- Identify potentially fraudulent postings more quickly.
- Protect job seekers from suspicious advertisements.
- Improve trust in the recruitment platform.
- Support human moderators by highlighting postings that may require further review.

The dataset is highly imbalanced, with fraudulent postings representing only a small proportion of all listings. Because of this, accuracy alone is not sufficient for evaluating the system. Precision, recall, F1-score and ROC-AUC are also considered.

---

## Dataset

The project uses the **Fake Job Postings Dataset** from Kaggle.

**Dataset source:**  
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

The dataset contains **17,880 job postings** and 18 original columns.

The target variable is:

- `fraudulent = 0` → Real job posting
- `fraudulent = 1` → Fraudulent job posting

### Class Distribution

| Class | Number of postings | Percentage |
|---|---:|---:|
| Real | 17,014 | 95.16% |
| Fraudulent | 866 | 4.84% |
| **Total** | **17,880** | **100%** |

The main text fields used for the NLP analysis are:

- Job title
- Company profile
- Job description
- Requirements
- Benefits

These fields were combined into a single text representation before preprocessing.

---

## Methodology

The project follows the workflow below:

```text
Job Posting Dataset
        ↓
Exploratory Data Analysis
        ↓
Text Cleaning
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
Stratified Train/Test Split
        ↓
CountVectorizer / TF-IDF
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Final Linear SVM Model
        ↓
Streamlit Application
```

A stratified **80/20 train-test split** was used so that the proportion of fraudulent postings remained similar in both the training and test sets.

---

## Natural Language Processing

The text data was processed in several stages.

### 1. Basic Cleaning

The text was converted to lowercase, URLs were removed, non-alphabetic characters were removed, and extra whitespace was normalised.

### 2. Stopword Removal

Common English words that provide limited classification information were removed using the NLTK English stopword list.

### 3. Lemmatization

Words were lemmatised using the NLTK WordNet lemmatiser. This reduces different forms of words to their base form where possible.

The average text length was reduced from approximately **398 words after basic cleaning to 254 words after stopword removal**.

---

## Feature Extraction

Two text representation approaches were compared.

### CountVectorizer

CountVectorizer represents text based on the frequency of individual words and word combinations.

The implementation used:

- Unigrams and bigrams
- `min_df=2`
- `max_df=0.95`

### TF-IDF

TF-IDF was used to represent the importance of terms within the job postings while reducing the influence of very common terms.

The implementation used:

- Unigrams and bigrams
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True`

The vectorisers were fitted only on the training data and then used to transform the test data. This prevents information from the test set from influencing the feature vocabulary during training.

---

## Machine Learning Models

Three classification algorithms were evaluated:

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (SVM)**

Class weighting was used for Logistic Regression and Linear SVM to help account for the imbalance between real and fraudulent postings.

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Results

| Model | Features | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Naive Bayes | CountVectorizer | 98.49% | 96.12% | 71.68% | 82.12% | — |
| Naive Bayes | TF-IDF | 95.86% | 100.00% | 14.45% | 25.25% | — |
| Logistic Regression | CountVectorizer | 98.66% | 88.34% | 83.24% | 85.71% | — |
| Logistic Regression | TF-IDF | 99.02% | 90.59% | 89.02% | 89.80% | 98.86% |
| Linear SVM | CountVectorizer | 98.27% | 84.47% | 78.61% | 81.44% | — |
| **Linear SVM** | **TF-IDF** | **99.19%** | **98.65%** | **84.39%** | **90.97%** | **99.42%** |

The **Linear SVM with TF-IDF** achieved the highest F1-score and ROC-AUC and was selected as the final model.

---

## Final Model

The final model is a **Linear SVM using TF-IDF features**.

Performance on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | **99.19%** |
| Precision | **98.65%** |
| Recall | **84.39%** |
| F1-score | **90.97%** |
| ROC-AUC | **99.42%** |

### Confusion Matrix

The final model produced the following results on the test set:

- **True Negatives:** 3,401
- **False Positives:** 2
- **False Negatives:** 27
- **True Positives:** 146

The confusion matrix is included in the `models` directory.

### Model Selection

Logistic Regression with TF-IDF achieved a slightly higher recall of **89.02%**, meaning it detected more of the fraudulent postings in the test set.

However, it produced **16 false positives**, compared with only **2 false positives** from the Linear SVM.

The Linear SVM was therefore selected because it provided a stronger overall balance between precision, recall and F1-score while substantially reducing false positives. Its ROC-AUC of **99.42%** also indicates strong separation between the two classes.

---

## Streamlit Application

The trained model and TF-IDF vectoriser are saved using Joblib and used by a Streamlit web application.

The application allows a user to paste a job posting and receive a classification.

```text
Job Posting
     ↓
Text Preprocessing
     ↓
TF-IDF Transformation
     ↓
Linear SVM
     ↓
Real / Fraudulent
```

The application displays:

- Predicted classification
- Potential risk indicators
- Overall assessment

---

## Additional Risk Indicators

In addition to the machine-learning prediction, the application contains a small rule-based risk-indicator layer.

It checks for patterns such as:

- Requests for upfront payments or fees
- Requests for financial information
- Requests for sensitive personal information
- Guaranteed income claims
- Unusually high-income claims
- No-experience requirements
- Urgency or pressure language
- Unusual payment methods

These indicators are **not part of the trained SVM model**. They provide additional supporting information and can highlight suspicious characteristics that may not be captured by the machine-learning model alone.

The application therefore combines the ML prediction with simple explainable checks to provide additional context to the user.

---

## Project Structure

```text
fake-job-posting-detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── fake_job_postings.csv
│
├── models/
│   ├── count_vectorizer.joblib
│   ├── svm_tfidf_confusion_matrix.png
│   ├── svm_tfidf_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/divyanshchawlaa/fake-job-posting-detection.git
cd fake-job-posting-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

---

## Running the Application

From the project root, run:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit application will open in a browser.

---

## Limitations

The model was trained using a specific labelled dataset of job postings. Therefore, its performance depends on how representative the dataset is of current fraudulent and legitimate job advertisements.

The dataset contains significantly fewer fraudulent examples than legitimate postings. Although stratified splitting and class weighting were used, some types of fraudulent postings may still be difficult to identify.

The model is also based mainly on textual patterns. A new type of scam using wording that is different from the training data may not be classified correctly.

The rule-based indicators provide additional information but are based on predefined patterns and can also produce false alarms.

The system should therefore be treated as a **decision-support tool rather than a replacement for human moderation**.

---

## Future Improvements

Possible future improvements include:

- Retraining the model using newer job-posting data.
- Adding more labelled fraudulent examples.
- Testing transformer-based language models such as BERT.
- Adding probability calibration to provide more meaningful confidence estimates.
- Integrating the system with a recruitment platform's moderation workflow.
- Monitoring model performance over time to identify changes in fraudulent posting patterns.
- Expanding the rule-based risk indicators using additional fraud patterns.

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **NLTK**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Matplotlib**
- **Jupyter Notebook**
- **Git / GitHub**

---

## Repository

GitHub repository:

https://github.com/divyanshchawlaa/fake-job-posting-detection