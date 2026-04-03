import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def extract_features(data):
    post = data[0]["data"]["children"][0]["data"]

    title = str(post["title"])
    created = pd.to_datetime(post["created_utc"], unit="s")

    # features
    post_hour = created.hour
    post_day = created.dayofweek
    is_weekend = int(post_day in [5, 6])

    title_words_count = len(title.split())
    has_question = int("?" in title)
    has_number = int(any(c.isdigit() for c in title))

    upvote_ratio = post.get("upvote_ratio", 0)
    num_comments = post.get("num_comments", 0)

    controversy_index = (1 - upvote_ratio) * np.log1p(num_comments)

    sentiment_score = sia.polarity_scores(title)["compound"]
    is_extreme_sentiment = int(abs(sentiment_score) > 0.5)

    is_self = int(post.get("is_self", False))

    features = {
        "upvote_ratio": upvote_ratio,
        "is_self": is_self,
        "post_hour": post_hour,
        "post_day": post_day,
        "is_weekend": is_weekend,
        "title_words_count": title_words_count,
        "has_question": has_question,
        "has_number": has_number,
        "controversy_index": controversy_index,
        "sentiment_score": sentiment_score,
        "is_extreme_sentiment": is_extreme_sentiment
    }

    return features