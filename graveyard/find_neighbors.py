import os
import json
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIG ────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

# paths to the files
API_META_PATH = "api_metadata.json"
EMBED_PATH    = "embeddings_combined_openai.npy"

# fixed query
GENERAL_TOOLS_LIST = [
     # Current weather
    (
        "WeatherAPI",
        "Reliable global weather data provider",
        "GetCurrentWeather",
        "Returns the current weather conditions for a specified city."
    ),
    # Cryptocurrency prices
    (
        "CryptoPriceAPI",
        "Real-time cryptocurrency price provider",
        "GetCryptoPrice",
        "Returns the current price of a specified cryptocurrency in a given fiat currency."
    ),
    # Stock market news
    (
        "Stock API",
        "Global stock market data provider",
        "Get News",
        "Stock market news from around the world."
    ),
    # Forward geocoding
    (
        "GeocodingAPI",
        "Address to geographic coordinates conversion service",
        "GeocodeAddress",
        "Converts a postal address into latitude and longitude coordinates."
    ),
    # Reverse geocoding
    (
        "ReverseGeocodingAPI",
        "Coordinates to address conversion service",
        "ReverseGeocode",
        "Returns the nearest address for a given latitude and longitude."
    ),
    # Currency conversion
    (
        "CurrencyExchangeAPI",
        "Live currency exchange rate service",
        "ConvertCurrency",
        "Converts an amount from one currency to another using current exchange rates."
    ),
    # Text translation
    (
        "TranslationAPI",
        "Multilingual text translation service",
        "TranslateText",
        "Translates input text from a source language to a target language."
    ),
    # News headlines
    (
        "NewsHeadlinesAPI",
        "Latest news headline aggregator",
        "GetTopHeadlines",
        "Retrieves the current top news headlines for a specified country or category."
    ),
    # Time zone lookup
    (
        "TimeZoneAPI",
        "Time zone information service",
        "GetTimeZone",
        "Returns the time zone and current local time for a specified location."
    ),
    # IP geolocation
    (
        "IPGeolocationAPI",
        "IP address geolocation service",
        "GetIPLocation",
        "Provides geographic location details for a specified IP address."
    ),
    # Weather forecast
    (
        "WeatherForecastAPI",
        "Global weather forecasting service",
        "GetWeatherForecast",
        "Returns a multi-day weather forecast for a given city or coordinates."
    ),
    # OCR text extraction
    (
        "OCRAPI",
        "Optical character recognition service",
        "ExtractText",
        "Extracts and returns text from an input image file."
    ),
    # Route planning
    (
        "MapRoutingAPI",
        "Driving and walking route planning service",
        "GetRoute",
        "Calculates and returns directions between two geographic locations."
    ),
    # Domain WHOIS lookup
    (
        "WhoisAPI",
        "Domain registration lookup service",
        "GetDomainWhois",
        "Retrieves WHOIS information for a given domain name."
    )
]

QUERY_TOOL, QUERY_TOOL_DESC, QUERY_API_NAME, QUERY_API_DESC = GENERAL_TOOLS_LIST[0]

# how many neighbors to show
K = 5

# which OpenAI model to use when embedding a new query
EMBED_MODEL = "text-embedding-ada-002"
# ────────────────────────────────────────────────────────────────────────


def load_records_and_texts(meta_path):
    """Flatten the api_metadata.json into a list of texts and record dicts."""
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    texts   = []
    for category, tools in data.items():
        for tool, info in tools.items():
            tool_desc = info.get("tool_desc", "").strip() or tool
            for api_name, api_desc in info.get("apis", []):
                txt = f"{tool_desc} | {api_name}: {api_desc}"
                records.append({
                    "category":   category,
                    "tool":       tool,
                    "tool_desc":  tool_desc,
                    "api_name":   api_name,
                    "api_desc":   api_desc
                })
                texts.append(txt)
    return records, texts


def embed_query(text, model=EMBED_MODEL):
    """Call OpenAI to embed a single text."""
    resp = openai.Embedding.create(model=model, input=[text])
    return np.array(resp["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)


def main():
    # load precomputed embeddings + records
    records, _ = load_records_and_texts(API_META_PATH)
    embs = np.load(EMBED_PATH)  # shape (N, D)
    N, D = embs.shape
    print(f"Loaded {N} embeddings of dimension {D}.")

    # build the query string and embed it
    query_text = f"{QUERY_TOOL}: {QUERY_TOOL_DESC} | {QUERY_API_NAME}: {QUERY_API_DESC}"
    print(f"\nQuery text:\n  {query_text}\n")
    q_emb = embed_query(query_text)

    # compute similarities
    sims = cosine_similarity(q_emb, embs)[0]

    # find unique top-K by tool
    seen_tools = set()
    unique_idxs = []
    # sort all indices descending by similarity
    for idx in np.argsort(sims)[::-1]:
        tool = records[idx]["tool"]
        if tool in seen_tools:
            continue
        seen_tools.add(tool)
        unique_idxs.append(idx)
        if len(unique_idxs) >= K:
            break

    # display
    print(f"Top {K} unique-tool nearest APIs:\n")
    for rank, idx in enumerate(unique_idxs, start=1):
        r = records[idx]
        print(f"{rank:2d}.  score={sims[idx]:.4f}")
        print(f"     Category:   {r['category']}")
        print(f"     Tool:       {r['tool']} — {r['tool_desc']}")
        print(f"     Endpoint:   {r['api_name']} — {r['api_desc']}\n")


if __name__ == "__main__":
    main()