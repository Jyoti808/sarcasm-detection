## 🚀 Live Demo

👉 [Try the Sarcasm Detection App](https://sarcasm-detection-eladdbhnhngkfpcd6ngo5g.streamlit.app)

# 😏 Sarcasm Detection System

A Natural Language Processing (NLP) based machine learning project that detects whether a given news headline or sentence is **Sarcastic** or **Non-Sarcastic**.

The project compares traditional machine learning with deep learning approaches and uses an **LSTM model** as the final selected model.


## 📌 Project Overview

Sarcasm can be difficult for machines to understand because its meaning often depends on context and word usage.

This project uses NLP techniques and deep learning to classify text into two categories:

- **0 — Non-Sarcastic**
- **1 — Sarcastic**

## 📊 Dataset

- Total Headlines: **28,617**
- Missing Values: **0**
- Duplicate Rows Removed: **2**
- Classes: **2**
- Vocabulary Size: **20,000**
- Maximum Sequence Length: **20**

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Converted text to lowercase
- Removed URLs
- Removed HTML tags
- Removed unnecessary characters
- Removed extra spaces
- Tokenized the text
- Converted text into numerical sequences
- Applied padding and truncation

## 🤖 Models Used

The following models were evaluated:

1. Logistic Regression with TF-IDF
2. LSTM
3. Bidirectional LSTM
4. GRU

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 84.52% | 83.24% | 84.53% | 83.88% | 92.32% |
| LSTM | **85.05%** | 81.64% | **88.52%** | **84.94%** | 93.20% |
| BiLSTM | 84.63% | **86.43%** | 80.34% | 83.28% | **93.21%** |
| GRU | 84.96% | 83.63% | 85.08% | 84.35% | 92.87% |

### 🏆 Final Model

The **LSTM model** was selected as the final model.

**Test Performance:**

- Accuracy: **85.05%**
- Precision: **81.64%**
- Recall: **88.52%**
- F1 Score: **84.94%**
- ROC-AUC: **93.20%**

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- NLP
- LSTM
- TF-IDF
- Streamlit
- Jupyter Notebook

## 📂 Project Files

```text
├── app.py
├── final_sarcasm_lstm_model.keras
├── sarcasm_tokenizer.pkl
├── sarcasm_model_config.json
├── requirements.txt
└── README.md
