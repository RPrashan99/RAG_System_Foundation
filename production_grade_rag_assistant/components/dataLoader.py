import json

def load_json(file_path = "data/bm25_corpus.json"):
    with open(f"{file_path}", "r") as f:
        data = json.load(f)
    return data