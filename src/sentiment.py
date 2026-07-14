"""
Financial-text sentiment scoring: VADER (fast lexicon) + FinBERT (finance-tuned transformer).

Two scorers, a daily aggregator, and a LEAK-FREE join into a price feature frame.
Both models are loaded lazily and cached, so importing this module is cheap and the
~440 MB FinBERT weights are only downloaded the first time score_finbert() is called.
"""
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# VADER (NLTK) — fast lexicon baseline
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _vader_analyzer():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def score_vader(texts: List[str]) -> List[float]:
    """Return VADER compound sentiment in [-1, 1] for each text."""
    sia = _vader_analyzer()
    return [sia.polarity_scores(t or "")["compound"] for t in texts]


# --------------------------------------------------------------------------- #
# FinBERT (transformers) — finance-tuned classifier
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _finbert_pipeline():
    from transformers import pipeline
    # top_k=None returns the full probability distribution per input.
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


def score_finbert(texts: List[str], batch_size: int = 32) -> List[float]:
    """
    Signed FinBERT sentiment per text: P(positive) - P(negative), in [-1, 1].
    Neutral pulls the score toward 0.
    """
    if not texts:
        return []
    pipe = _finbert_pipeline()
    raw = pipe(
        [t or "" for t in texts],
        truncation=True,
        max_length=256,
        batch_size=batch_size,
    )
    scores: List[float] = []
    for dist in raw:
        probs = {d["label"].lower(): d["score"] for d in dist}
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return scores


# --------------------------------------------------------------------------- #
# Aggregation + leak-free join
# --------------------------------------------------------------------------- #
def daily_sentiment(
    news_df: pd.DataFrame,
    date_col: str = "date",
    text_col: str = "headline",
    use_finbert: bool = True,
) -> pd.DataFrame:
    """
    Collapse a headline-level frame into one row per calendar day.

    Returns a DataFrame indexed by normalized date with columns:
        vader_mean, news_count, [finbert_mean]
    """
    if news_df.empty:
        cols = ["vader_mean", "news_count"] + (["finbert_mean"] if use_finbert else [])
        return pd.DataFrame(columns=cols)

    df = news_df[[date_col, text_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    texts = df[text_col].fillna("").astype(str).tolist()

    df["vader"] = score_vader(texts)
    if use_finbert:
        df["finbert"] = score_finbert(texts)

    agg = {"vader": "mean"}
    if use_finbert:
        agg["finbert"] = "mean"
    out = df.groupby(date_col).agg(agg)
    out["news_count"] = df.groupby(date_col).size()

    rename = {"vader": "vader_mean", "finbert": "finbert_mean"}
    return out.rename(columns=rename).sort_index()


def merge_sentiment_feature(
    price_df: pd.DataFrame,
    sentiment_daily: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Attach daily sentiment to a price frame WITHOUT look-ahead leakage.

    Each price row is matched (merge_asof, backward) to the most recent sentiment
    date that is strictly older by at least `lag` trading-safe days: we shift the
    sentiment timeline forward by `lag` days so a given day only ever sees
    sentiment published on prior days. Missing days are forward-filled from the
    last known sentiment; days before any news get a neutral 0.

    Both indexes must be DatetimeIndex. Returns price_df with sentiment columns added.
    """
    price = price_df.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)

    sent_cols = list(sentiment_daily.columns)
    if sentiment_daily.empty:
        for c in sent_cols or ["vader_mean", "finbert_mean"]:
            price[c] = 0.0
        return price

    sent = sentiment_daily.copy()
    sent.index = pd.to_datetime(sent.index)
    # Push sentiment forward by `lag` days so "today" cannot see today's news.
    sent.index = sent.index + pd.Timedelta(days=lag)
    sent = sent.sort_index()

    merged = pd.merge_asof(
        price.sort_index(),
        sent,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    # Forward-fill gaps between news days; neutral before the first news day.
    merged[sent_cols] = merged[sent_cols].ffill().fillna(0.0)
    return merged.loc[price_df.index]
