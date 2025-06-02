# Crypto Sentiment Oracle

A Python pipeline that extracts and analyzes cryptocurrency sentiment from Reddit using NLP. It classifies coins based on sentiment signals (e.g., STRONG BUY, NEUTRAL) and separates Tier 1 market cap coins from meme or low-cap coins.

## Overview

This project scrapes Reddit posts and comments, runs transformer-based sentiment analysis, weights sentiment using engagement, recency, and subreddit importance, and outputs structured daily sentiment scores per coin.

## Features

- Scrapes posts and comments from relevant crypto subreddits
- Applies transformer-based sentiment model (`cardiffnlp/twitter-roberta-base-sentiment`)
- Filters based on confidence thresholds
- Applies time decay, vote weight, and subreddit weight
- Aggregates sentiment by coin
- Classifies coins by sentiment signal
- Distinguishes Tier 1 (top 500 market cap) vs Meme/Other coins using CoinGecko

## Project Structure

crypto-sentiment-oracle/
├── src/
│ ├── reddit_scraper.py # Main pipeline
│ └── config.py # Environment loading and subreddit config
├── notebooks/
│ └── exploration.ipynb # Visual analysis and scoring
├── data/
│ └── overall_coin_sentiment.csv # Sample output
├── .env # Reddit API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md


## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Add a .env file

REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=sentiment-oracle:v1

## Run the Scraper

python src/reddit_scraper.py

## Explore the Notebook
jupyter notebook notebooks/exploration.ipynb

Dependencies

    Python 3.8+

    pandas

    requests

    transformers

    torch

    matplotlib

    python-dotenv
    
Applications

    Cryptocurrency trend tracking

    Social sentiment research

    Signal generation for short-term plays

    Meme coin vs major market cap sentiment separation
