from sentence_transformers import SentenceTransformer
import faiss
import json

class Retriever:
    def __init__(self, index_path='data/faiss.index', meta_path='data/meta.json'):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.texts = meta['texts']

    def query(self, q, top_k=3):
        emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        return [self.texts[i] for i in I[0]]
