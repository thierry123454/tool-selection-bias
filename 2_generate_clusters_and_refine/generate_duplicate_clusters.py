import os
import json
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

# ─── CONFIG ────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

# paths to your files
API_META_PATH = "../1_endpoint_metadata_and_embed/api_metadata.json"
EMBED_PATH    = "../1_endpoint_metadata_and_embed/embeddings_combined_openai.npy"
QUERIES_JSON    = "../data/instruction/G1_query.json"

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
    ),
    # URL shortening
    (
        "URLShortenerAPI",
        "Fast, reliable URL shortening and expansion service",
        "ShortenURL",
        "Takes a long URL and returns a compact, share-friendly short link."
    ),

    # Email validation
    (
        "EmailValidationAPI",
        "Bulk & single email address verification service",
        "ValidateEmail",
        "Checks whether an email address is syntactically correct, exists, and can receive mail."
    ),

    # Text sentiment analysis
    (
        "SentimentAnalysisAPI",
        "Multilingual text sentiment detection",
        "AnalyzeSentiment",
        "Returns sentiment (positive/neutral/negative) and confidence for a given piece of text."
    ),

    # Language detection
    (
        "LanguageDetectionAPI",
        "Identify the language of a given text snippet",
        "DetectLanguage",
        "Analyzes input text and returns its ISO-639 language code along with confidence scores."
    ),

    # QR-code decoding
    (
        "QRCodeAPI",
        "Scan and decode QR codes from images or URLs",
        "DecodeQRCode",
        "Accepts an image or URL containing a QR code and returns the embedded payload."
    ),

    # Spell checking
    (
        "SpellCheckAPI",
        "Proofreading service for spelling and grammar",
        "CheckSpelling",
        "Returns a list of misspelled words and suggested corrections for an input text."
    ),

    # Phone number formatting & validation
    (
        "PhoneValidationAPI",
        "International phone number formatting and validation",
        "ValidatePhoneNumber",
        "Parses and validates a phone number, returning its country, format, and validity status."
    ),

    # Currency lists & symbols
    (
        "CurrencyInfoAPI",
        "Metadata on world currencies (codes, names, symbols)",
        "ListCurrencies",
        "Returns ISO codes, names, and symbols for all supported fiat currencies."
    ),
]

QUERY_TOOL, QUERY_TOOL_DESC, QUERY_API_NAME, QUERY_API_DESC = GENERAL_TOOLS_LIST[0]

# how many neighbors to show
K = 8

# which OpenAI model to use when embedding a new query
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL   = "gpt-4"
SLEEP_BETWEEN_CALLS = 0.3  # to avoid rate limits
MAX_OUTLIER_LOOPS   = 5
# ────────────────────────────────────────────────────────────────────────


def load_records_and_texts(meta_path):
    """Flatten api_metadata.json into a list of texts and record dicts."""
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

def load_required_params(queries_path):
    """
    Build a mapping from (tool_name, api_name) to required_parameters
    by scanning all entries in queries.json.
    """
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    param_map = {}
    for q in queries:
        for api in q.get("api_list", []):
            key = (api["tool_name"], api["api_name"])
            param_map[key] = api.get("required_parameters", [])
    return param_map

def embed_query(text, model=EMBED_MODEL):
    """Call OpenAI to embed a single text."""
    resp = openai.Embedding.create(model=model, input=[text])
    return np.array(resp["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)

def is_same_cluster(api_list):
    """
    Ask the LLM if the provided endpoints all solve the same task.
    Includes required parameters in the prompt for clarity.
    """
    system = (
        "You are an expert at reading API endpoint names, descriptions, and their required parameters. "
        "I will give you a list of endpoints. "
        "Answer exactly “yes” or “no” (lowercase) to the question: "
        "Does there exist a task that **every single endpoint** in the list can perform?"
    )

    lines = ["Here are the endpoints to check:\n"]
    for i, api in enumerate(api_list, 1):
        lines.append(f"{i}) Endpoint: {api['api_name']} — {api['api_desc']}")
        reqs = api.get("required_parameters", [])
        if reqs:
            lines.append("   Required parameters:")
            for p in reqs:
                lines.append(f"     - {p['name']} ({p['type']})")
    user = "\n".join(lines)

    print("\n" + user + "\n")

    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role":   "system", "content": system},
            {"role":     "user", "content": user}
        ],
        temperature=0,
        max_tokens=4,
    )
    ans = resp.choices[0].message.content.strip().lower()
    return ans == "yes"

def detect_outliers(endpoints):
    """
    Ask the LLM to spot any endpoint(s) that cannot solve the same task as the rest.
    Returns a list of indices in `endpoints` to remove, or empty if all belong.
    """
    system = (
        "You are an expert at reading API endpoint names, descriptions, and required parameters.  "
        "I will give you a numbered list of endpoints.  "
        "Does there exist a specific user query that **every single endpoint** in the list can perform?"
        "If this is the case, answer exactly yes."
        "If not, identify any endpoints that cannot perform that query and answer with an array of the endpoint numbers to remove so that"
        " the answer would be yes. DO NOT GIVE ANY EXPLANATION, JUST THE ARRAY."
    )
    lines = ["Here are the endpoints:\n"]
    for i, api in enumerate(endpoints, 1):
        lines.append(f"{i}) {api['api_name']} — {api['api_desc']}")
        reqs = api.get("required_parameters", [])
        if reqs:
            lines.append("   Required parameters:")
            for p in reqs:
                name = p.get("name", "")
                ptype = p.get("type", "")
                desc = p.get("description", "").strip()
                # one way to format them:
                lines.append(f"     - {name} ({ptype})")
                if desc:
                    lines.append(f"         Description: {desc}")
    user = "\n".join(lines)

    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system", "content": system},
            {"role":"user",   "content": user}
        ],
        temperature=0,
        max_tokens=50,
    )
    txt = resp.choices[0].message.content.strip()
    print("User prompt:", user)
    print("LLM raw reply:", txt)

    # handle the "yes" case
    if txt.lower() == "yes":
        return []

    # parse JSON, allowing for quoted numbers
    try:
        raw = json.loads(txt)
    except json.JSONDecodeError:
        print("⚠️ Could not parse JSON from txt reply:", txt)
        return []

    outliers = []
    for x in raw:
        # convert numeric strings to int if needed
        if isinstance(x, str) and x.isdigit():
            idx = int(x)
        elif isinstance(x, (int, float)) and int(x) == x:
            idx = int(x)
        else:
            continue
        if 1 <= idx <= len(endpoints):
            outliers.append(idx)

    return outliers

def main():
    param_map = load_required_params(QUERIES_JSON)

    # load metadata + embeddings
    records, _ = load_records_and_texts(API_META_PATH)
    embs = np.load(EMBED_PATH)  # shape (N, D)
    print(f"Loaded {len(records)} records and embeddings of shape {embs.shape}.")

    clusters = []

    # for each general API, find top-K nearest neighbors
    for tool, tool_desc, name, desc in GENERAL_TOOLS_LIST:
        query_text = f"{tool}: {tool_desc} | {name}: {desc}"
        print(f"\n→ Querying neighbors for {tool}::{name}")
        q_emb = embed_query(query_text)
        sims = cosine_similarity(q_emb, embs)[0]

        # select top-K unique-tools
        seen, idxs = set(), []
        for idx in np.argsort(sims)[::-1]:
            t = records[idx]["tool"]
            if t in seen:
                continue
            seen.add(t)
            idxs.append(idx)
            if len(idxs) >= K:
                break

        # build candidate cluster: original + neighbors
        candidate = []
        for idx in idxs:
            rec = records[idx].copy()
            key = (rec["tool"], rec["api_name"])
            rec["required_parameters"] = param_map.get(key, [])
            candidate.append(rec)

        # iteratively remove outliers
        for _ in range(MAX_OUTLIER_LOOPS):
            outliers = detect_outliers(candidate)
            if not outliers:
                break
            print(f"  ↳ Removing endpoints at positions {outliers}")
            # drop by descending index to not shift positions
            for i in sorted(outliers, reverse=True):
                candidate.pop(i-1)
            time.sleep(SLEEP_BETWEEN_CALLS)
        else:
            print("  ⚠️  Max outlier loops reached.")

        if len(candidate) > 3:
            print(f"  ✅ Final cluster size: {len(candidate)}")
            clusters.append(candidate)
        else:
            print("  ❌ Cluster too small after outlier removal.")
        
        time.sleep(SLEEP_BETWEEN_CALLS)

    # 5) write out clusters
    out_path = "duplicate_api_clusters_2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {len(clusters)} clusters to {out_path}")


if __name__ == "__main__":
    main()