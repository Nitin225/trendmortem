import requests

url = "https://www.reddit.com/r/technology/hot.json?limit=5"
headers = {"User-Agent": "TrendMortem/1.0"}

response = requests.get(url, headers=headers)
data = response.json()

for post in data["data"]["children"]:
    p = post["data"]
    print(p["title"])
    print("Score:", p["score"])
    print("Comments:", p["num_comments"])
    print("---")