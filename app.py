from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import requests
from feature_engineering import extract_features

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
model = joblib.load("xgb_model.pkl")

@app.get("/")
def home():
    return {"message": "TrendMortem API is running"}

class PostURL(BaseModel):
    url: str

FEATURE_ORDER = ['upvote_ratio', 'is_self', 'post_hour', 'post_day', 
                 'is_weekend', 'title_words_count', 'has_question', 'has_number', 
                 'controversy_index', 'sentiment_score', 'is_extreme_sentiment']


@app.post("/analyze")
def analyze_url(post: PostURL):
    try:
        clean_url = post.url.split("?")[0]   # remove query params
        json_url = clean_url.rstrip("/") + ".json"  
        headers = {"User-Agent": "TrendMortem/1.0"}
        reddit_response = requests.get(json_url, headers=headers)
        
        if reddit_response.status_code != 200:
            return {"error": "Failed to fetch Reddit data"}
        
        try:
            data = reddit_response.json()
        except:
            return {"error": "Invalid JSON from Reddit"}
        
        if not data or "data" not in data[0]:
            return {"error": "Invalid Reddit response"}
        features = extract_features(data)

        df = pd.DataFrame([features])
        df = df.reindex(columns=FEATURE_ORDER, fill_value=0)
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]

        
        prompt = f"""A Reddit post has the following features:

        - Upvote ratio (0–1, >0.85 high, 0.7–0.85 medium, <0.7 low): {features.get('upvote_ratio', 0)}
        - Controversy index (0–1 low, 1–3 moderate, >3 high): {features.get('controversy_index', 0)}
        - Post hour (0–23 UTC): {features.get('post_hour', 0)}
        - Title word count (5–20 typical): {features.get('title_words_count', 0)}
        - Sentiment score (-1 to 1): {features.get('sentiment_score', 0)}

        Model predicted viral = {int(prediction[0])} with probability {round(float(probability), 3)}.

        Explain in 2-3 sentences WHY based ONLY on these features.
        Do NOT force strict categories (low/medium/high). Interpret values naturally and relatively.
        Do not assume anything outside these features."""
                
        groq_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        
        explanation = groq_response.choices[0].message.content
        
        return {
        "success": True,
        "data": {
            "viral": int(prediction[0]),
            "probability": round(float(probability), 3),
            "explanation": explanation
        }
}
    
    except Exception as e:
        return {
        "success": False,
        "error": str(e)
        }
    

    

