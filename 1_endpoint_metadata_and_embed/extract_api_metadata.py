import os
import json
import re

def first_sentence(text: str):
    """
    Return the first full sentence (ending in . ! or ?) from text,
    or the entire text if no sentence-ending punctuation is found.
    """
    text = text.strip()
    # split on punctuation (., !, ?) followed by whitespace
    parts = re.split(r'(?<=[\.!?])\s+', text, maxsplit=1)
    return parts[0]

def is_valid_api(name: str, desc: str):
    # 1) must be reasonably long
    if len(name) < 3 or len(desc) < 15:
        return False

    # 2) description shouldn’t just echo the name
    if desc.strip().lower() == name.strip().lower():
        return False

    # 3) must have at least a few words
    if len(desc.split()) <= 3:
        return False

    # 4) should start with an action verb (“Get”, “List”, “Search”, etc.)?
    # if not desc.split()[0].istitle() or desc.split()[0] not in {"Get","List","Search","Fetch","Create","Delete","Update"}:
    #     return False

    return True

def extract_api_metadata(tools_root: str):
    """
    Walks through tools_root/<category>/*.json, extracts (name, description)
    groups them by category.
    """
    all_metadata = {}

    # Iterate over each category folder.
    for category in os.listdir(tools_root):
        category_path = os.path.join(tools_root, category)
        if not os.path.isdir(category_path):
            continue
        all_metadata[category] = {}

        # For each tool JSON in category
        for fname in os.listdir(category_path):
            api_tuples = []
            if not fname.endswith('.json'):
                continue

            file_path = os.path.join(category_path, fname)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                tool_key = data.get('name')
                tool_description = first_sentence(data.get("tool_description", '').strip())
                # Extract every entry in "api_list"
                for api in data.get('api_list', []):
                    name = api.get('name', '').strip()
                    desc = api.get('description', '').strip()
                    if name and desc and is_valid_api(name, desc):
                        api_tuples.append([name, desc])
                        print(f"Added API '{name}' with description '{desc}'")

                if api_tuples:
                    all_metadata[category][tool_key] = {
                        "tool_desc": tool_description,
                        "apis": api_tuples
                    }

            except Exception as e:
                print(f"Warning: failed to process {file_path}: {e}")

    return all_metadata

if __name__ == '__main__':
    tools_root = 'data/toolenv/tools'
    output_file = 'api_metadata.json'

    metadata = extract_api_metadata(tools_root)
    with open(output_file, 'w') as out:
        json.dump(metadata, out)

    print(f"Extracted API metadata for {len(metadata)} categories -> {output_file}")