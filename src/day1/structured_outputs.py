"""ShopAgent — Claude structured output with Pydantic (Day 1, Prompt 10)."""

import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class ReviewAnalysis(BaseModel):
    total_reviews: int
    average_rating: float
    sentiment_distribution: dict[str, int]
    top_complaints: list[str]
    top_praises: list[str]


def load_reviews(path: str, limit: int = 10) -> list[dict]:
    reviews = []
    with open(path) as f:
        for line in f:
            reviews.append(json.loads(line.strip()))
            if len(reviews) >= limit:
                break
    return reviews


def analyze_reviews(reviews: list[dict]) -> ReviewAnalysis:
    client = anthropic.Anthropic()

    reviews_text = json.dumps(reviews, indent=2, ensure_ascii=False)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze these e-commerce reviews and return a structured analysis.\n\n"
                    f"Reviews:\n{reviews_text}\n\n"
                    "Return a JSON object with these exact fields:\n"
                    "- total_reviews: number of reviews analyzed\n"
                    "- average_rating: average rating (float)\n"
                    '- sentiment_distribution: {"positive": N, "neutral": N, "negative": N}\n'
                    "- top_complaints: list of main complaints found\n"
                    "- top_praises: list of main praises found\n\n"
                    "Return ONLY the JSON object, no other text."
                ),
            }
        ],
    )

    raw = response.content[0].text
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    return ReviewAnalysis(**data)


if __name__ == "__main__":
    reviews_path = Path(__file__).resolve().parents[2] / "gen" / "data" / "reviews" / "reviews.jsonl"

    if not reviews_path.exists():
        print(f"Reviews file not found: {reviews_path}")
        raise SystemExit(1)

    reviews = load_reviews(str(reviews_path))
    print(f"Loaded {len(reviews)} reviews from {reviews_path.name}")

    analysis = analyze_reviews(reviews)
    print("\n" + "=" * 50)
    print("ShopAgent — Review Analysis (Structured Output)")
    print("=" * 50)
    print(f"Total reviews: {analysis.total_reviews}")
    print(f"Average rating: {analysis.average_rating:.1f}")
    print(f"Sentiment: {analysis.sentiment_distribution}")
    print(f"Top complaints: {analysis.top_complaints}")
    print(f"Top praises: {analysis.top_praises}")
