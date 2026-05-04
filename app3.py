import streamlit as st
import json
import faiss
import torch
import numpy as np
import pickle
import re
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ограничение памяти
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- Инициализация интерфейса ---
st.set_page_config(page_title="Knowledge Guard Pro", layout="wide", page_icon="⚖️")

st.title("⚖️ Knowledge Guard: Гибридная верификация (BM25 + FAISS)")
st.markdown("""
Система использует **Entity Boosting** и **RRF (Reciprocal Rank Fusion)** для объединения 
лексического поиска (BM25) и семантического (FAISS).
""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Вспомогательные функции
def tokenize_ru(text):
    return re.findall(r'\w+', text.lower())

def split_into_atomic_claims(text):
    """Разделяет сложный текст на атомарные утверждения по союзам."""
    text = text.strip().rstrip('.')
    # Паттерн для разделения на логические части[cite: 1]
    split_pattern = r'(?:,\s*а\s+)|(?:,\s*но\s+)|(?:,\s*и\s+)'
    parts = re.split(split_pattern, text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3] or [text]

@st.cache_resource
def load_resources():
    search_model = SentenceTransformer('./hugg', device=device)
    rerank_model = CrossEncoder('./rerank', device=device)
    
    nli_path = './bert'
    tokenizer = AutoTokenizer.from_pretrained(nli_path)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_path).to(device)
    
    index = faiss.read_index("tree_kb.faiss")
    with open("tree_metadata.json", "r", encoding="utf-16") as f:
        metadata = json.load(f)
        
    with open("tree_kb_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
        
    return search_model, rerank_model, tokenizer, nli_model, index, metadata, bm25

try:
    search_model, rerank_model, tokenizer, nli_model, index, metadata, bm25 = load_resources()
    id2label = nli_model.config.id2label
except Exception as e:
    st.error(f"Ошибка загрузки ресурсов: {e}. Убедитесь, что вы запустили скрипт индексации!")
    st.stop()

def reciprocal_rank_fusion(faiss_indices, bm25_indices, k=60):
    """Слияние результатов двух поисковиков."""
    rrf_scores = {}
    for rank, doc_idx in enumerate(faiss_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_idx in enumerate(bm25_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

def verify_fact(claim, threshold):
    k_coarse = 10
    
    # 1. FAISS Search
    query_vec = search_model.encode([f"query: {claim}"]).astype('float32')
    faiss.normalize_L2(query_vec)
    faiss_dist, faiss_idx = index.search(query_vec, k=k_coarse)
    valid_faiss = [idx for i, idx in enumerate(faiss_idx[0]) if faiss_dist[0][i] >= (threshold - 0.15)]
    
    # 2. BM25 Search
    tokenized_query = tokenize_ru(claim)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_idx = np.argsort(bm25_scores)[::-1][:k_coarse].tolist()
    valid_bm25 = [idx for idx in bm25_idx if bm25_scores[idx] > 0]

    # 3. RRF Fusion
    fused_indices = reciprocal_rank_fusion(valid_faiss, valid_bm25)
    if not fused_indices: return "NO_DATA", 0.0, []

    # 4. Reranking (Cross-Encoder)
    candidate_indices = fused_indices[:10]
    candidate_contexts = [metadata[idx]['rich_context'] for idx in candidate_indices]
    pairs = [[claim, ctx] for ctx in candidate_contexts]
    rerank_scores = rerank_model.predict(pairs)
    
    reranked = sorted(zip(candidate_indices, rerank_scores, candidate_contexts), key=lambda x: x[1], reverse=True)
    best_results = reranked[:3]
    
    # 5. NLI Verification
    final_contexts = [res[2] for res in best_results]
    inputs = tokenizer(final_contexts, [claim]*len(final_contexts), truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = nli_model(**inputs)
        probs_batch = torch.softmax(outputs.logits, dim=1).tolist()

    valid_results = []
    for i, probs in enumerate(probs_batch):
        m = metadata[best_results[i][0]]
        scores_dict = {
            'Подтверждено': probs[0], # Предполагая стандартный порядок labels в NLI
            'Нейтрально': probs[1],
            'Противоречие': probs[2]
        }
        # Корректировка маппинга если в модели другой id2label
        scores_dict = {('Подтверждено' if 'entail' in id2label[j].lower() else 
                        'Противоречие' if 'contradict' in id2label[j].lower() else 
                        'Нейтрально'): p for j, p in enumerate(probs)}

        best_label = max(scores_dict, key=scores_dict.get)
        final_verdict = "Недостаточно контекста" if (best_label == 'Нейтрально' and scores_dict['Нейтрально'] > 0.6) else best_label

        valid_results.append({
            "meta": m,
            "verdict": final_verdict,
            "scores": scores_dict,
            "rerank_score": float(best_results[i][1])
        })

    return "SUCCESS", float(best_results[0][1]), valid_results

# ... остальной код Streamlit UI без изменений ...

# --- UI Sidebar ---
with st.sidebar:
    st.header("⚙️ Конфигурация")
    st.markdown(f"**Объектов в KB:** {len(metadata)}")
    threshold = st.slider("Мягкий порог FAISS", 0.50, 0.99, 0.65, 0.01) # Снизил дефолт, т.к. решатель спасает
    st.info("💡 **Логика решателя:** FAISS ищет 10 черновиков. Cross-Encoder ранжирует их по смыслу. Топ-3 отправляются в NLI.")

# --- Main UI ---
user_input = st.text_area("Введите утверждение для проверки:", placeholder="Пример: Декартово произведение создает ориентированные пары...")

if st.button("🚀 Выполнить логический вывод", type="primary", use_container_width=True):
    if user_input.strip():
        with st.status("Запуск цепочки рассуждений...", expanded=True):
            claims = split_into_atomic_claims(user_input)
            results = []
            for claim in claims:
                status, sim, data = verify_fact(claim, threshold)
                verdict = "Нет данных"
                best = None
                if status == "SUCCESS" and data:
                    best = data[0]
                    verdict = best['verdict']
                results.append({"fact": claim, "verdict": verdict, "data": best})

        st.divider()
        if any(r['verdict'] == "Противоречие" for r in results):
            st.error("🚨 **ВЕРДИКТ: ОБНАРУЖЕНО ЛОГИЧЕСКОЕ ПРОТИВОРЕЧИЕ / ГАЛЛЮЦИНАЦИЯ**")
        elif all(r['verdict'] == "Подтверждено" for r in results):
            st.success("✅ **ВЕРДИКТ: ИНФОРМАЦИЯ ПОЛНОСТЬЮ ВЕРИФИЦИРОВАНА**")
        else:
            st.warning("⚠️ **ВЕРДИКТ: ЧАСТИЧНАЯ ВЕРИФИКАЦИЯ ИЛИ НЕХВАТКА ДАННЫХ**")

        for i, res in enumerate(results):
            with st.expander(f"Факт {i+1}: {res['fact'][:60]}...", expanded=True):
                st.write(f"**Текст:** {res['fact']}")
                
                # Цветовая индикация вердикта
                color = {"Подтверждено": "green", "Противоречие": "red", "Недостаточно контекста": "orange"}.get(res['verdict'], "gray")
                st.subheader(f"Статус: :{color}[{res['verdict']}]")
                
                if res['data']:
                    m = res['data']['meta']
                    s = res['data']['scores']
                    c1, c2, c3 = st.columns(3)
                    
                    # Показываем чистые вероятности NLI-модели
                    c1.metric("Confirmed", f"{s.get('Подтверждено', 0)*100:.1f}%")
                    c2.metric("Neutral", f"{s.get('Нейтрально', 0)*100:.1f}%")
                    c3.metric("Contradiction", f"{s.get('Противоречие', 0)*100:.1f}%")
                    
                    st.markdown(f"### 📚 Обоснование из базы: **{m['term']}**")
                    
                    # Исправленная строка: используем rerank_score вместо raw_sim
                    st.caption(f"**Доверие поисковика (Cross-Encoder): {res['data']['rerank_score']:.2f}**")
                    
                    # Если есть формула в метаданных, отображаем её
                    if m.get('formal_logic'): 
                        st.latex(m['formal_logic'])
                    
                    # Вывод контекста[cite: 1]
                    st.info(m['rich_context'])
                else:
                    st.warning("Для данного утверждения не найдено подходящих данных в базе знаний.")