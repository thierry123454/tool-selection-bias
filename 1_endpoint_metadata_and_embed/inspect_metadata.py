import json
from statistics import mean

def analyze_metadata(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Number of categories
    num_categories = len(data)

    # Count total APIs per category (sum across all tools)
    counts = {
        category: sum(len(api_list) for api_list in tools.values())
        for category, tools in data.items()
    }

    total_apis = sum(counts.values())
    avg_per_cat = mean(counts.values()) if counts else 0

    print(f"Total categories: {num_categories}")
    print(f"Total APIs:       {total_apis}")
    print(f"Avg APIs/category:{avg_per_cat:.1f}\n")

    # Top 5 categories
    print("Top 5 categories by number of APIs:")
    for category, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {category:<20} {cnt}")

    # Bottom 5 categories
    print("\nBottom 5 categories by number of APIs:")
    for category, cnt in sorted(counts.items(), key=lambda x: x[1])[:5]:
        print(f"  {category:<20} {cnt}")

if __name__ == '__main__':
    analyze_metadata('api_metadata.json')