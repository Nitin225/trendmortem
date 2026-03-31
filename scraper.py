import requests
import pandas as pd
import time

def scrape_subreddit(subreddit, limit=200):
    headers = {"User-Agent": "TrendMortem/1.0"}
    posts = []
    after = None

    while len(posts) < limit:
        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&t=month"
        if after:
            url += f"&after={after}"

        response = requests.get(url, headers=headers)
        data = response.json()

        children = data["data"]["children"]
        if not children:
            break

        for post in children:
            p = post["data"]
            posts.append({
                "title": p["title"],
                "score": p["score"],
                "upvote_ratio": p["upvote_ratio"],
                "num_comments": p["num_comments"],
                "created_utc": p["created_utc"],
                "subreddit": p["subreddit"],
                "is_self": p["is_self"],
                "awards": p["total_awards_received"],
                "url": p["url"]
            })

        after = data["data"]["after"]
        time.sleep(2)  # Reddit rate limit ke liye

    return posts[:limit]

subreddits = ["technology", "worldnews", "gaming", "science", "todayilearned"]
all_posts = []

for sub in subreddits:
    print(f"Scraping r/{sub}...")
    posts = scrape_subreddit(sub, limit=200)
    all_posts.extend(posts)
    print(f"  {len(posts)} posts fetched")

df = pd.DataFrame(all_posts)
df.to_csv("reddit_data.csv", index=False)
print(f"\nTotal: {len(df)} posts saved to reddit_data.csv")