from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
model = joblib.load("xgb_model.pkl")

@app.get("/")
def home():
    return {"message": "TrendMortem API is running"}

class PostFeatures(BaseModel):
    upvote_ratio: float
    post_hour: int
    post_day: int
    is_weekend: int
    title_words_count: int
    has_question: int
    has_number: int
    controversy_index: float
    sentiment_score: float
    is_extreme_sentiment: int
    is_self: int
    upvote_ratio: float
    
FEATURE_ORDER = ['upvote_ratio', 'is_self', 'post_hour', 'post_day', 
                 'is_weekend', 'title_words_count', 'has_question', 'has_number', 
                 'controversy_index', 'sentiment_score', 'is_extreme_sentiment']

@app.post("/analyze")
def analyze(post: PostFeatures):
    data = pd.DataFrame([post.dict()])
    data = data[FEATURE_ORDER]
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]
    
    prompt = f"""
    A Reddit post has these features:
    - Upvote ratio: {post.upvote_ratio}
    - Controversy index: {post.controversy_index}
    - Post hour (UTC): {post.post_hour}
    - Title word count: {post.title_words_count}
    - Sentiment score: {post.sentiment_score}
    
    Model predicted viral={int(prediction[0])} with probability {round(float(probability), 3)}.
    
    In 2-3 sentences, explain why this post is likely or unlikely to go viral.
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    explanation = response.choices[0].message.content
    
    return {
        "viral": int(prediction[0]),
        "probability": round(float(probability), 3),
        "explanation": explanation
    }
    

