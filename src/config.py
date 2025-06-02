import os
from dotenv import load_dotenv

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
SUBREDDITS = [
    "CryptoCurrency",
    "AltcoinDiscussion",
    "cryptomoonshots",
    "CryptoMarkets",
    "SatoshiStreetBets",
    "DeFi",
    "BitcoinMarkets",
    "Ethereum",
    "Crypto_Beginners",
    "CryptoCurrencyTrading"
]
TICKERS = [
    "SOL", "MATIC", "RUNE", "BTC", "ETH", "ADA", "DOGE", "DOT",
    "AVAX", "LTC", "UNI", "LINK", "BCH", "XRP", "ALGO", "ATOM",
    "FIL", "ICP", "VET", "THETA"
]
POST_LIMIT = 300