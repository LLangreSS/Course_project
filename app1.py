import streamlit as st
import json
import faiss
import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Инициализация интерфейса ---
st.set_page_config(page_title="IS Knowledge Guard", layout="wide", page_icon="🛡️")

st.title("🛡️ Верификатор знаний (Enterprise Edition)")
st.markdown("""
Система глубокого логического анализа утверждений. 
Использует **строгий режим верификации**: если сумма сомнений и противоречий больше уверенности, факт отклоняется.
""")

# --- Определение устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Кэширование ресурсов ---
@st.cache_resource
def load_all_models():
    search_model = SentenceTransformer('./hugg')
    nli_path = './bert'
    tokenizer = AutoTokenizer.from_pretrained(nli_path)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_path)
    nli_model.to(device)
    index = faiss.read_index("tree_kb.faiss")
    with open("tree_metadata.json", "r", encoding="utf-16") as f:
        metadata = json.load(f)
    return search_model, tokenizer, nli_model, index, metadata

try:
    search_model, tokenizer, nli_model, index, metadata = load_all_models()
    id2label = nli_model.config.id2label
except Exception as e:
    st.error(f"Ошибка загрузки ресурсов: {e}")
    st.stop()

# --- Логика разбиения ---
def split_into_atomic_claims(complex_claim):
    complex_claim = complex_claim.strip().rstrip('.')
    split_pattern = r'(?:,\s*где\s+)|(?:,\s*а\s+)|(?:,\s*но\s+)|(?:,\s*котор[а-я]{2}\s+)|(?:,\s*и\s+)'
    parts = re.split(split_pattern, complex_claim)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3] or [complex_claim]

# --- Логика проверки ОДНОГО факта ---
def verify_atomic_fact(claim, threshold):
    query_vec = search_model.encode([claim]).astype('float32')
    faiss.normalize_L2(query_vec)
    k_matches = 3
    distances, indices = index.search(query_vec, k=k_matches)
    
    if distances[0][0] < threshold:
        return "NO_DATA", distances[0][0], []

    contexts, valid_matches = [], []
    for i in range(k_matches):
        if distances[0][i] >= (threshold - 0.05):
            idx = indices[0][i]
            match = metadata[idx]
            clean_context = match.get('rich_context', '').replace("Термин: ", "").replace(". Определение: ", " — это ")
            contexts.append(clean_context)
            valid_matches.append((match, distances[0][i]))
            
    if not contexts: return "NO_DATA", distances[0][0], []

    inputs = tokenizer(contexts, [claim]*len(contexts), truncation=True, padding=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = nli_model(**inputs)
        probs_batch = torch.softmax(outputs.logits, dim=1).tolist()
        
    verification_results = []
    claim_lower = claim.lower()
    
    for i, probs in enumerate(probs_batch):
        match, sim_score = valid_matches[i]
        term = match.get('term', '').lower()
        
        # 1. СТРОГИЙ СЕМАНТИЧЕСКИЙ ФИЛЬТР (Subject Match)
        subject_found = term in claim_lower or any(w in claim_lower for w in term.split() if len(w) > 3)
        
        res = {}
        n_val, e_val, c_val = 0, 0, 0
        
        for j, prob in enumerate(probs):
            label = id2label[j].lower()
            if 'entail' in label:
                e_val = prob if subject_found else 0.0 # Блокировка подмены
            elif 'contradict' in label:
                c_val = prob if subject_found else 0.95 # Искусственный риск[cite: 1]
            else:
                n_val = prob

        # 2. ТВОЯ НОВАЯ ЛОГИКА: Вес сомнений (Neutral + Contradict > Entail)
        if (n_val + c_val) > e_val:
            e_val = 0.0 # Отменяем подтверждение, если сомнений больше
            
        res['Подтверждено'], res['Нейтрально'], res['Противоречие'] = e_val, n_val, c_val
        
        verification_results.append({
            "term": term, "similarity": sim_score, "scores": res, 
            "context": contexts[i], "subject_match": subject_found
        })
        
    verification_results.sort(key=lambda x: x['scores'].get('Подтверждено', 0), reverse=True)
    return "SUCCESS", distances[0][0], verification_results

# --- Сайдбар ---
with st.sidebar:
    st.header("⚙️ Настройки")
    st.info(f"📚 База: {len(metadata)} узлов[cite: 1]")
    SIMILARITY_THRESHOLD = st.slider("Строгость поиска", 0.70, 0.99, 0.86, 0.01)
    if st.button("🧹 Очистить кэш"):
        st.cache_resource.clear()
        if device.type == 'cuda': torch.cuda.empty_cache()

# --- Основной экран ---
user_input = st.text_area("Введите утверждение:", height=120, placeholder="Например: Пустое множество имеет мощность 0...")

if st.button("🕵️ Проверить", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Введите текст.")
    else:
        with st.status("Верификация...", expanded=True) as status:
            atomic_claims = split_into_atomic_claims(user_input)
            all_results, has_contradiction, missing_count = [], False, 0
            
            for i, claim in enumerate(atomic_claims):
                st.write(f"Анализ части {i+1}...")
                stat, max_sim, data = verify_atomic_fact(claim, SIMILARITY_THRESHOLD)
                
                fact_verdict, best_match = "Нет данных", None
                if stat == "SUCCESS" and data:
                    best_match = data[0]
                    # Берем лейбл с макс. значением из скорректированных весов
                    fact_verdict = max(best_match['scores'], key=best_match['scores'].get)
                    
                    # Если подтверждение обнулено логикой весов, переключаемся на лидера среди сомнений
                    if best_match['scores']['Подтверждено'] == 0.0:
                        temp = best_match['scores'].copy()
                        del temp['Подтверждено']
                        fact_verdict = max(temp, key=temp.get)
                    
                    if fact_verdict == "Противоречие": has_contradiction = True
                else: missing_count += 1
                
                all_results.append({"fact": claim, "verdict": fact_verdict, "best_match": best_match, "max_sim": max_sim})
            status.update(label="Готово!", state="complete", expanded=False)

        # Вывод вердиктов
        st.divider()
        if has_contradiction: st.error("🚨 **ВЕРДИКТ: СОДЕРЖИТ ГАЛЛЮЦИНАЦИЮ**")
        elif missing_count == len(atomic_claims): st.warning("❓ **ВЕРДИКТ: НЕТ ДАННЫХ В БАЗЕ**[cite: 1]")
        else: st.success("✅ **ВЕРДИКТ: ИНФОРМАЦИЯ ПОДТВЕРЖДЕНА**")

        for i, res in enumerate(all_results):
            with st.expander(f"Детализация: Факт {i+1}", expanded=True):
                st.write(f"**Текст:** {res['fact']}")
                cl = {"Подтверждено": "green", "Противоречие": "red", "Нейтрально": "orange"}.get(res['verdict'], "gray")
                st.subheader(f"Статус: :{cl}[{res['verdict']}]")
                if res['best_match']:
                    b = res['best_match']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Подтверждение", f"{b['scores']['Подтверждено']*100:.1f}%")
                    c2.metric("Нейтрально", f"{b['scores']['Нейтрально']*100:.1f}%")
                    c3.metric("Противоречие", f"{b['scores']['Противоречие']*100:.1f}%")
                    st.info(f"**Источник (БД):** {b['term'].capitalize()} — {b['context']}[cite: 1]")
                    if not b['subject_match']:
                        st.error("⚠️ **Ошибка субъекта**: Термины не совпадают. Система заблокировала ложное подтверждение[cite: 1].")
                    elif b['scores']['Подтверждено'] == 0.0 and res['verdict'] != "Противоречие":
                        st.warning("⚠️ **Пессимистичный фильтр**: Сумма сомнений выше уверенности. Вердикт пересмотрен.")