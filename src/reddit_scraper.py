import requests
from datetime import datetime, timezone, timedelta
import pandas as pd
import time
import os
import re
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

USER_AGENT = "sentiment-oracle:v2.0"
SUBREDDITS = [
    "CryptoCurrency", "AltcoinDiscussion", "cryptomoonshots", "CryptoMarkets",
    "SatoshiStreetBets", "DeFi", "BitcoinMarkets", "Ethereum",
    "Crypto_Beginners", "CryptoCurrencyTrading",
    "CryptoTechnology", "ethfinance", "btc", "CryptoMoon",
    "Altcoins", "Crypto_General", "CryptoMarketsTech"
]
POSTS_PER_PAGE = 100
MAX_PAGES = 3

BUY_KEYWORDS = ["BUY", "LONG", "MOON", "HODL", "PUMP"]
SELL_KEYWORDS = ["SELL", "DUMP", "BUST", "BAGHOLD", "FUD"]

SUB_WEIGHTS = {
    "CryptoCurrency": 1.0, "AltcoinDiscussion": 1.0, "cryptomoonshots": 0.5,
    "CryptoMarkets": 0.8, "SatoshiStreetBets": 0.5, "DeFi": 0.7,
    "BitcoinMarkets": 0.9, "Ethereum": 0.9, "Crypto_Beginners": 0.6,
    "CryptoCurrencyTrading": 0.8, "CryptoTechnology": 0.7, "ethfinance": 0.9,
    "btc": 0.9, "CryptoMoon": 0.6, "Altcoins": 0.7, "Crypto_General": 0.6,
    "CryptoMarketsTech": 0.7
}

HEADERS = {"User-Agent": USER_AGENT}

blacklist = {"THE", "AND", "FOR", "BUT", "WITH"}

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model.eval()

def get_known_tickers(use_cache=True):
    cache_file = "data/ticker_cache.json"
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    url = "https://api.coingecko.com/api/v3/coins/list"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    coins = r.json()
    tickers = {c["symbol"].upper(): c["name"] for c in coins}
    os.makedirs("data", exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(tickers, f)
    return tickers

def batch_analyze_sentiment(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(scores, dim=1).values
        sentiments = scores[:, 2] - scores[:, 0]  # pos - neg
    return sentiments.tolist(), confidence.tolist()

def fetch_posts(subreddit, after=None):
    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    params = {"limit": POSTS_PER_PAGE}
    if after:
        params["after"] = after

    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            return data["data"]["children"], data["data"].get("after")
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
    return [], None

def find_tickers(text, known_tickers):
    cands = set(re.findall(r"\b\$?([A-Z]{3,5})\b", text))
    return {t for t in cands if t in known_tickers and t not in blacklist}

def count_keywords(text, keywords):
    t = text.upper()
    return sum(t.count(kw) for kw in keywords)

def fetch_comments(permalink):
    url = f"https://www.reddit.com{permalink}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        comments = data[1]["data"]["children"]
        texts = []
        for c in comments:
            if c["kind"] != "t1":
                continue
            body = c["data"].get("body", "")
            ups = c["data"].get("ups", 0)
            author = c["data"].get("author")
            if not body or author in ["[deleted]", None] or ups < 1:
                continue
            texts.append(body.upper())
        return texts
    except Exception:
        return []

def scrape_subreddit(subreddit, known_tickers):
    records = []
    after = None
    pages = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)

    while pages < MAX_PAGES:
        posts, after = fetch_posts(subreddit, after)
        if not posts:
            break

        texts, metas = [], []

        for post in posts:
            d = post["data"]
            created = datetime.fromtimestamp(d.get("created_utc"), timezone.utc)
            if created < cutoff:
                continue
            if d.get("author") in ["[deleted]", None] or not d.get("is_self", True):
                continue
            title = d.get("title", "")
            selftext = d.get("selftext", "")
            if d.get("ups", 0) < 5 or len(title.split()) < 3:
                continue
            content = f"{title} {selftext}".upper()
            comments = fetch_comments(d.get("permalink", ""))
            comment_sentiments, comment_conf = batch_analyze_sentiment(comments) if comments else ([], [])
            if comment_sentiments:
                comment_score = sum(comment_sentiments) / len(comment_sentiments)
            else:
                comment_score = 0.0
            tickers = find_tickers(content, known_tickers)
            if not tickers:
                continue
            texts.append(content)
            metas.append({
                "created": created,
                "tickers": tickers,
                "buy_count": count_keywords(content, BUY_KEYWORDS),
                "sell_count": count_keywords(content, SELL_KEYWORDS),
                "ups": d.get("ups", 0),
                "subreddit": subreddit
            })

        if texts:
            sentiments, confidences = batch_analyze_sentiment(texts)
            filtered = [(m, s) for m, s, c in zip(metas, sentiments, confidences) if c > 0.5]
        else:
            filtered = []

        for meta, sentiment in filtered:
            engagement_weight = 1 + min(meta["ups"] / 50, 1)
            # Engagement 7/10, Comment 8/10
            sentiment *= 1.2  # Boost engagement effect
            sentiment = max(min(sentiment, 1.0), -1.0)
            age_hours = (datetime.now(timezone.utc) - meta["created"]).total_seconds() / 3600
            time_weight = max(0.1, 0.95 ** (age_hours / 6))  # Make time less aggressive (5/10 weight)
            sub_weight = SUB_WEIGHTS.get(meta["subreddit"], 0.5)
            for ticker in meta["tickers"]:
                records.append({
                    "date": meta["created"].strftime("%Y-%m-%d"),
                    "coin": ticker,
                    "coin_name": known_tickers[ticker],
                    "subreddit": meta["subreddit"],
                    "sentiment_score": (0.2 * sentiment + 0.8 * comment_score),
                    "buy_count": meta["buy_count"],
                    "sell_count": meta["sell_count"],
                    "ups": meta["ups"],
                    "time_weight": time_weight,
                    "sub_weight": sub_weight
                })

        if not after:
            break
        pages += 1
        time.sleep(1)

    return records

def main():
    known = get_known_tickers()
    all_data = []
    print("\nFetching data from Reddit...")
    for sub in tqdm(SUBREDDITS, desc="Scraping subreddits"):
        all_data.extend(scrape_subreddit(sub, known))
    if not all_data:
        print("No recent data.")
        return

    df = pd.DataFrame(all_data)

    # Group sentiment per coin (across subreddits)
    overall = df.groupby(["date", "coin", "coin_name"]).agg({
        "sentiment_score": "mean",
        "buy_count": "sum",
        "sell_count": "sum",
        "ups": "sum",
        "time_weight": "mean",
        "sub_weight": "mean",
        "coin": "count"
    }).rename(columns={"coin": "mention_count"}).reset_index()

    # Adjusted weighting: comment 8/10, time 5/10, subreddit 3/10
    overall["weighted_sentiment"] = overall["sentiment_score"] * (0.5 * overall["time_weight"] + 0.3 * overall["sub_weight"])
    overall["pct_buy_calls"] = overall["buy_count"] / overall["mention_count"]

    def decide(row):
        if row["mention_count"] >= 5 and row["weighted_sentiment"] > 0.2 and row["pct_buy_calls"] >= 0.03:
            return "STRONG BUY"
        if row["mention_count"] >= 3 and row["weighted_sentiment"] > 0.1 and row["pct_buy_calls"] >= 0.015:
            return "BUY"
        if row["mention_count"] >= 5 and row["weighted_sentiment"] < -0.2 and row["sell_count"] > row["buy_count"] * 1.2:
            return "STRONG SELL"
        if row["mention_count"] >= 3 and row["weighted_sentiment"] < -0.1 and row["sell_count"] > row["buy_count"]:
            return "SELL"
        return "NEUTRAL"

    overall["signal"] = overall.apply(decide, axis=1)

    overall.to_csv("data/overall_coin_sentiment.csv", index=False)
    print("Saved overall sentiment summary to data/overall_coin_sentiment.csv")
    
if __name__ == "__main__":
    main()