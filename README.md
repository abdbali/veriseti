# Mevsimler ve İklim Soru-Cevap Sistemi

Bu proje, **Türkçe mevsimler ve iklim ile ilgili soru-cevap sistemi** oluşturmak için hazırlanmıştır. Küçük bir veri seti kullanarak, kullanıcı sorularına **TF-IDF, fuzzy matching ve overlap** yöntemleriyle cevap verir.

---

##  Gereksinimler:

* Python 3.9+
* Pandas
* Scikit-learn
* Rapidfuzz
* Joblib
* Gradio (opsiyonel)

```bash
pip install pandas scikit-learn rapidfuzz joblib gradio
```

---

##  Veri Seti:

Veri seti Hugging Face üzerinde bulunur ve `mevsimler_iklim_veriseti.csv` olarak indirilir.

* Sütunlar:

  * `input`: Kullanıcının sorusu
  * `output`: Cevap

---
<img width="851" height="673" alt="image" src="https://github.com/user-attachments/assets/75c8484b-f6ca-4d55-a3a9-6302dd4921b3" />
---

##  Kullanım:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

# Veri yükleme
df = pd.read_csv('mevsimler_iklim_veriseti.csv')
df.columns = [c.strip().lower() for c in df.columns]

# Normalizasyon
def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9ığüşöçİĞÜŞÖÇ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df['input_norm'] = df['input'].astype(str).apply(normalize_text)
df['output'] = df['output'].astype(str)

# TF-IDF vektörü
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df['input_norm'].tolist())

# Soru-cevap fonksiyonu
def get_answer(user_query: str):
    q_norm = normalize_text(user_query)
    
    # 1) Tam eşleşme
    mask = df['input_norm'] == q_norm
    if mask.any():
        idx = df[mask].index[0]
        return df.at[idx, 'output'], {'method': 'exact', 'matched_input': df.at[idx, 'input'], 'score': 1.0}
    
    # 2) Kısmi eşleşme
    for idx, row in df[['input_norm', 'input', 'output']].iterrows():
        if row['input_norm'] in q_norm or q_norm in row['input_norm']:
            return row['output'], {'method': 'containment', 'matched_input': row['input'], 'score': 0.9}
    
    # 3) Ortak kelime
    q_tokens = set(q_norm.split())
    if q_tokens:
        overlaps = df['input_norm'].apply(lambda s: len(q_tokens & set(s.split())))
        best_overlap = int(overlaps.max())
        best_idx = int(overlaps.idxmax())
        ratio = best_overlap / max(1, len(q_tokens))
        if best_overlap >= 1 and ratio >= 0.30:
            return df.at[best_idx, 'output'], {'method': 'overlap', 'matched_input': df.at[best_idx, 'input'], 'score': float(ratio), 'common_words': best_overlap}
    
    # 4) TF-IDF + cosine similarity
    q_vec = vectorizer.transform([q_norm])
    cosines = cosine_similarity(q_vec, X_tfidf).flatten()
    best_idx = int(np.argmax(cosines))
    best_sim = float(cosines[best_idx])
    if best_sim >= 0.35:
        return df.at[best_idx, 'output'], {'method': 'tfidf_cosine', 'matched_input': df.at[best_idx, 'input'], 'score': best_sim}
    
    # 5) Fuzzy fallback
    choices = df['input_norm'].tolist()
    best = process.extractOne(q_norm, choices, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= 70:
        idx = choices.index(best[0])
        return df.at[idx, 'output'], {'method': 'fuzzy', 'matched_input': df.at[idx, 'input'], 'score': best[1]}
    
    # 6) Öneriler
    top_indices = np.argsort(-cosines)[:3]
    suggestions = df.iloc[top_indices]['input'].tolist()
    return ("Üzgünüm, bunu anlayamadım. Aşağıdaki benzer sorulardan birini deneyebilirsin:\n- " + "\n- ".join(suggestions),
            {'method': 'no_match', 'suggestions': suggestions})
```

---
<img width="855" height="636" alt="image" src="https://github.com/user-attachments/assets/23bef829-14cf-4ef8-99f8-b7cef54e7cee" />

---

##  Hızlı Test:

```python
print("Soru-cevap döngüsü. Çıkmak için 'q' yaz.")
while True:
    q = input("Soru: ")
    if q.strip().lower() in ('q','quit','exit'):
        print("Çıkılıyor.")
        break
    ans, meta = get_answer(q)
    print("\nCevap:")
    print(ans)
    print("\n(Eşleşme bilgisi):", meta)
    print("-"*40)
```

---

##  Notlar: 

* Türkçe karakterler korunur.
* Küçük veri setleri için hızlı çalışır.
* Daha büyük veri setleri veya web arayüzü için **Gradio entegrasyonu** yapılabilir.
