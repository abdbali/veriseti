import pandas as pd
import json

def load_passages(csv_path='data/mevsimler.csv'):
    df = pd.read_csv(csv_path)
    passages = []
    for i, row in df.iterrows():
        text = f"Soru: {row['input']}\nCevap: {row['output']}"
        passages.append({"id": str(i), "text": text})
    return passages

if __name__ == "__main__":
    passages = load_passages()
    with open('data/passages.json', 'w', encoding='utf-8') as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    print(f"{len(passages)} passage kaydedildi.")
