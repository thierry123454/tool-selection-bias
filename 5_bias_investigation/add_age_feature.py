import re
import requests
import concurrent.futures
from datetime import datetime, timezone
import json

def fetch_age_days(url):
    """
    Fetch the RapidAPI page at 'url', extract the 'createdAt' timestamp,
    and compute the age in days since that timestamp.
    """
    # ensure trailing slash
    url = url.rstrip("/") + "/"

    try:
        r = requests.get(url, timeout=10)
        # look for the 13-digit createdAt in the embedded JSON
        # print(r.text)
        # print('\\"createdAt\\":\\"1639314751198\\"' in r.text)
        m = re.search(r'\\"createdAt\\":\\"(\d{13})', r.text)
        if not m:
            return None
        ts_ms = int(m.group(1))
        # turn ms-since-epoch into a timezone-aware datetime
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return None

with open('correct_api_meta.json', 'r', encoding='utf-8') as f:
    meta_items = json.load(f)

with open('final_features.json', 'r', encoding='utf-8') as f:
    feature_items = json.load(f)

url_map = {
    (item['cluster_id'], item['api']): item['url']
    for item in meta_items
    if 'url' in item
}

unique_urls = set(url_map.values())

age_map = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for url, days in zip(unique_urls, executor.map(fetch_age_days, unique_urls)):
            age_map[url] = days

print(age_map)

for feat in feature_items:
    key = (feat['cluster_id'], feat['api'])
    url = url_map.get(key)
    feat['age_days'] = age_map.get(url)

with open('final_features.json', 'w', encoding='utf-8') as f:
    json.dump(feature_items, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(feature_items)} entries with 'age_days' to final_features_with_age.json")
