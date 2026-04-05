# TrendMortem
> Will your Reddit post go viral?

TrendMortem is an ML-powered tool that analyzes any Reddit post URL and predicts its virality using classical machine learning, NLP, and LLM-generated explanations.

---

## How it works

1. User pastes a Reddit post URL
2. Backend fetches post data from Reddit API
3. Features are extracted automatically (sentiment, controversy index, timing, etc.)
4. XGBoost model predicts viral probability
5. Groq LLM generates a human-readable explanation

---

## Tech Stack

- **ML Model** — XGBoost (viral recall: 0.78)
- **NLP** — VADER Sentiment Analysis
- **Explainability** — SHAP values
- **Backend** — FastAPI + Uvicorn
- **LLM** — Groq API (LLaMA 3.3 70B)
- **Frontend** — Streamlit
- **Data** — Reddit REST API (no PRAW, custom scraper)

---

## Features

- Paste any Reddit URL → instant virality prediction
- Controversy index derived from upvote ratio + comment velocity
- SHAP-based feature importance — not a black box
- LLM explanation — why this post will/won't go viral
- HuggingFace sentiment tested, VADER retained (better performance)

---

## Model Performance

| Metric | Value |
|---|---|
| Viral Recall | 0.78 |
| Viral F1 | 0.57 |
| Accuracy | 0.72 |
| Threshold | 0.35 |
| Dataset | 1000 posts, 5 subreddits |

---

## Setup
```bash
# Clone the repo
git clone https://github.com/Nitin225/trendmortem
cd trendmortem

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Train the model
jupyter notebook model_training.ipynb

# Start backend
uvicorn app:app --reload

# Start frontend (new terminal)
streamlit run streamlit_app.py
```

---

## Project Structure
```
trendmortem/
├── scraper.py              # Reddit data collection
├── feature_engineering.py  # Feature extraction (production)
├── feature_engineering.ipynb
├── eda.ipynb               # Exploratory data analysis
├── model_training.ipynb    # XGBoost + SHAP
├── sentiment_analysis.ipynb # HuggingFace experiments
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
└── requirements.txt
```

---

## Key Design Decisions

- **No Kaggle dataset** — data scraped directly from Reddit API
- **created_utc removed** — data leakage prevention
- **num_comments removed** — data leakage prevention  
- **Threshold 0.35** — optimized for viral recall over accuracy
- **Per-subreddit viral label** — prevents subreddit bias in target variable
- **SHAP explainability** — instance-level explanation, not just global importance