RAG ile "Mevsimler ve İklim" 

**Amaç:** Hugging Face dataset `abdbali/mevsimler_iklim_veriseti` kullanarak öğretmen/öğrenci odaklı bir RAG (Retrieval-Augmented Generation) uygulaması geliştirmek. Proje basit bir retriever (FAISS), embedding üretimi (sentence-transformers), ve bir generative LLM (örnek: OpenAI veya yerel LLM) ile çalışır.
---

## Proje yapısı

```
rag-mevsimler-iklim/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ mevsimler.csv         # HuggingFace'den indirilen ham CSV
├─ notebooks/
│  └─ 01_explore_dataset.ipynb
├─ src/
│  ├─ preprocess.py        # CSV -> bölümlere ayırma (passage chunking)
│  ├─ embed_index.py       # Embedding oluşturma + FAISS index
│  ├─ retriever.py         # Sorgu ile en yakın pasajları getiren fonksiyon
│  └─ generator.py         # Retriever çıktısını LLM'e verip cevap üreten fonksiyon
├─ backend/
│  └─ app.py               # FastAPI sunucusu: /query endpoint
├─ web/
│  ├─ index.html           # Basit HTML/JS arayüzü
│  └─ style.css
└─ examples/
   └─ demo_queries.txt
```

---

## Gereksinimler

`requirements.txt` içeriği (başlangıç):

```text
fastapi
uvicorn[standard]
transformers
sentence-transformers
faiss-cpu
pandas
requests
python-dotenv
openai    # eğer OpenAI kullanacaksanız (opsiyonel)
```

> Not: Yerel bir açık kaynak LLM (Llama 2, MPT, Falcon) kullanacaksanız `transformers` ve ilgili model dosyalarını yönetmeniz gerekir. Bulut LLM (OpenAI, Anthropic) tercih ederseniz API anahtarlarını `.env` içinde tutun.

---

## Dataset'i indirme / önizleme

1. Hugging Face dataset sayfasından (`abdbali/mevsimler_iklim_veriseti`) CSV'yi indirin veya HF `datasets` kütüphanesi ile yükleyin.

Örnek (Python):

```python
# notebooks/01_explore_dataset.ipynb veya scripts
from datasets import load_dataset
import pandas as pd

ds = load_dataset('abdbali/mevsimler_iklim_veriseti')
df = pd.DataFrame(ds['train'])
df.to_csv('data/mevsimler.csv', index=False)
print(df.head())
```

Dataset kısa (82 satır) — soru-cevap tarzı pasajlar içeriyor. Her satır `input` (soru veya ifade) ve `output` (cevap) formatında.

---

## Ön işleme (preprocess.py)

Amaç: uzun metinleri 200–400 token'lık pasajlara bölmek veya soru-cevap çiftlerini doğrudan birimler (passage) olarak kullanmak.

`src/preprocess.py` (özet):

```python
import pandas as pd
from pathlib import Path

def load_and_chunk(csv_path='data/mevsimler.csv'):
    df = pd.read_csv(csv_path)
    # Bu dataset kısa ve zaten soru-cevap; her satırı bir passage yapabiliriz
    passages = []
    for i, row in df.iterrows():
        text = f"Soru: {row['input']}\nCevap: {row['output']}"
        passages.append({'id': str(i), 'text': text})
    return passages

if __name__ == '__main__':
    p = load_and_chunk()
    import json
    Path('data/passages.json').write_text(json.dumps(p, ensure_ascii=False, indent=2))
    print('Passages saved:', len(p))
```

---

## Embedding ve FAISS index (embed\_index.py)

`src/embed_index.py` örneği:

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

MODEL_NAME = 'all-mpnet-base-v2'  # küçük, kaliteli embedding modeli

def build_index(passages_json='data/passages.json', index_path='data/faiss.index', meta_path='data/meta.json'):
    with open(passages_json, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    texts = [p['text'] for p in passages]
    ids = [p['id'] for p in passages]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # save metadata
    import json
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'ids': ids, 'texts': texts}, f, ensure_ascii=False)
    print('Index built')

if __name__ == '__main__':
    build_index()
```

Not: Daha büyük veri/production için `IndexIVFFlat` gibi disk tabanlı yapı düşünün.

---

## Retriever (retriever.py)

`src/retriever.py`:

```python
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-mpnet-base-v2'

class Retriever:
    def __init__(self, index_path='data/faiss.index', meta_path='data/meta.json'):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.texts = meta['texts']

    def query(self, q, top_k=5):
        emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        results = [self.texts[i] for i in I[0]]
        return results

if __name__ == '__main__':
    r = Retriever()
    print(r.query('21 Mart ne demek?'))
```

---

##  Generator (generator.py)

Seçenek A — OpenAI (hızlı, kolay):

```python
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

PROMPT_TEMPLATE = '''Aşağıdaki pasajları öğretici, kısa ve öğrenci-dostu şekilde kullanarak soruyu yanıtla. Eğer pasajlarda cevap yoksa "Bilmiyorum" diye açıkla.

Pasajlar:
{passages}

Soru: {question}

Cevap:'"''

def generate_with_openai(question, passages):
    prompt = PROMPT_TEMPLATE.format(passages='\n\n'.join(passages), question=question)
    resp = openai.Completion.create(model='gpt-3.5-turbo', prompt=prompt, max_tokens=256, temperature=0.2)
    return resp['choices'][0]['text'].strip()
```

**Not:** `gpt-3.5-turbo` genellikle ChatCompletion formatı ister; örnek basitleştirildi. Alternatif: `transformers` ile yerel model veya `llama.cpp` tabanlı pipeline.

Seçenek B — Yerel açık kaynak model (transformers):

```python
from transformers import pipeline

def generate_local(question, passages):
    prompt = PROMPT_TEMPLATE.format(passages='\n\n'.join(passages), question=question)
    gen = pipeline('text-generation', model='meta-llama/Llama-2-7b-chat-hf')
    out = gen(prompt, max_new_tokens=200)
    return out[0]['generated_text']
```

---

## API (FastAPI)

`backend/app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever import Retriever
from src.generator import generate_with_openai

app = FastAPI()
retriever = Retriever()

class Query(BaseModel):
    question: str

@app.post('/query')
async def query(q: Query):
    passages = retriever.query(q.question, top_k=5)
    answer = generate_with_openai(q.question, passages)
    return {'answer': answer, 'passages': passages}

# Run: uvicorn backend.app:app --reload --port 8000
```

---

## HTML arayüzü (web/index.html)

Basit `index.html` (fetch ile `/query` çağırır):

```html
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Mevsimler & İklim - Soru Cevap (RAG)</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;max-width:800px;margin:2rem auto;padding:1rem}
    textarea{width:100%;height:100px}
    .btn{padding:0.6rem 1rem;margin-top:0.5rem}
    pre{background:#f4f4f4;padding:1rem;border-radius:6px}
  </style>
</head>
<body>
  <h1>Mevsimler & İklim — Soru Sor</h1>
  <p>Dataset: abdbali/mevsimler_iklim_veriseti</p>
  <textarea id="q" placeholder="Sorunuzu yazın...">21 Mart'ta kuzey yarım kürede hangi mevsim başlar?</textarea>
  <br>
  <button id="send" class="btn">Gönder</button>
  <h2>Cevap</h2>
  <pre id="answer">Bekleniyor...</pre>
  <h3>Kullanılan Pasajlar</h3>
  <div id="passages"></div>

  <script>
    document.getElementById('send').onclick = async () => {
      const q = document.getElementById('q').value
      document.getElementById('answer').textContent = 'Sorgulanıyor...'
      const resp = await fetch('/query', {
        method: 'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question:q})
      })
      const j = await resp.json()
      document.getElementById('answer').textContent = j.answer
      document.getElementById('passages').innerHTML = j.passages.map(p => `<pre>${p}</pre>`).join('')
    }
  </script>
</body>
</html>
```

> Not: Eğer FastAPI backend'iniz farklı bir portta ise `fetch` URL'sini `http://localhost:8000/query` olarak güncelleyin ve CORS izinlerini ekleyin.

---

##  Çalıştırma adımları 

1. Sanal ortam oluşturun: `python -m venv .venv && source .venv/bin/activate` (Windows: `.\.venv\Scripts\activate`)
2. Paketleri yükleyin: `pip install -r requirements.txt`
3. Dataset'i indirin ve `data/mevsimler.csv` olarak kaydedin (notebook içinde gösterildi).
4. `python src/preprocess.py` ile `data/passages.json` oluşturun.
5. `python src/embed_index.py` ile FAISS index oluşturun.
6. `.env` içine LLM API anahtarınızı ekleyin (ör. `OPENAI_API_KEY=...`) veya lokal model ayarlayın.
7. `uvicorn backend.app:app --reload --port 8000` ile backend'i başlatın.
8. Tarayıcıda `web/index.html` dosyasını açın (veya basit bir static server ile sunun) — backend'e CORS gerekiyorsa FastAPI'ye `from fastapi.middleware.cors import CORSMiddleware` ekleyin.

---

## Güvenlik ve Etik Notları

* LLM cevapları mutlaka öğretmen gözetiminde kullanılmalı — özellikle yanlış bilgi olasılığı her zaman vardır.
* API anahtarlarını kodda doğrudan paylaşmayın; `.env` kullanın.
* Dataset zaten eğitim amaçlı, lisans `apache-2.0` olarak listelenmiş.

---

## Geliştirme fikirleri (ilerisi)

* Çok dilli destek (Türkçe embedding ve Türkçe LLM tercih edin).
* Öğrenme modülü: öğrenci cevaplarına göre açıklayıcı geri bildirim sağlayan ek mantık.
* Sınav modu: çoktan seçmeli soru üretimi + otomatik değerlendirme.
* Mobil uyumlu, öğretmen paneli (kullanım analitiği), vs.

---

## Ek: Hazır kod parçaları ve ipuçları

* `sentence-transformers` ile Türkçe destekli modeller de değerlendirilebilir: `paraphrase-multilingual-mpnet-base-v2`, `all-mpnet-base-v2` vs.
* FAISS index'i disk tabanlı yapmak için `IndexIVFFlat` + `index.train(...)` adımları.
* Eğer OpenAI ChatCompletion kullanıyorsanız `messages=[{"role":"user","content":prompt}]` formatını kullanın.

---


