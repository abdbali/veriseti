from sentence_transformers import SentenceTransformer
import faiss
import json

def build_index(passages_json='data/passages.json', index_path='data/faiss.index', meta_path='data/meta.json'):
    with open(passages_json, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    texts = [p['text'] for p in passages]
    ids = [p['id'] for p in passages]

    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({"ids": ids, "texts": texts}, f, ensure_ascii=False)

    print("Index oluşturuldu ve kaydedildi.")

if __name__ == "__main__":
    build_index()
