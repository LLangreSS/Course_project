import streamlit as st
import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Инициализация интерфейса ---
st.set_page_config(page_title="IS Knowledge Guard", layout="wide", page_icon="🛡️")

st.title("🛡️ Верификатор знаний интеллектуальных систем")
st.markdown("""
Это приложение проверяет утверждения на соответствие локальной базе знаний. 
Оно использует **FAISS** для поиска и **mDeBERTa** для логического анализа.
""")

# --- Кэширование ресурсов ---
@st.cache_resource
def load_all_models():
    # Загрузка модели поиска
    search_model = SentenceTransformer('./hugg')
    
    # Загрузка NLI модели и токенизатора
    nli_path = './bert'
    tokenizer = AutoTokenizer.from_pretrained(nli_path)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_path)
    
    # Загрузка индекса FAISS и метаданных
    index = faiss.read_index("tree_kb.faiss")
    with open("tree_metadata.json", "r", encoding="utf-16") as f:
        metadata = json.load(f)
        
    return search_model, tokenizer, nli_model, index, metadata

# Загружаем всё один раз
try:
    search_model, tokenizer, nli_model, index, metadata = load_all_models()
    id2label = nli_model.config.id2label
except Exception as e:
    st.error(f"Ошибка загрузки ресурсов: {e}")
    st.stop()

# --- Логика проверки ---
def perform_verification(claim):
    # Поиск в FAISS[cite: 1]
    query_vec = search_model.encode([claim]).astype('float32')
    faiss.normalize_L2(query_vec)
    
    SIMILARITY_THRESHOLD = 0.86 # Порог отсечения[cite: 1]
    k_matches = 3
    distances, indices = index.search(query_vec, k=k_matches)
    
    if distances[0][0] < SIMILARITY_THRESHOLD:
        return "NO_DATA", distances[0][0], []

    verification_results = []
    
    for i in range(k_matches):
        sim_score = distances[0][i]
        if sim_score < (SIMILARITY_THRESHOLD - 0.05):
            continue
            
        idx = indices[0][i]
        match = metadata[idx]
        term = match.get('term', '').lower()
        context_text = match.get('rich_context', '')
        
        # Очистка текста для NLI[cite: 1]
        clean_context = context_text.replace("Термин: ", "").replace(". Определение: ", " — это ")
        
        # Семантический фильтр (Subject Match)[cite: 1]
        claim_lower = claim.lower()
        subject_found = term in claim_lower or any(w in claim_lower for w in term.split() if len(w) > 3)
        
        # NLI анализ[cite: 1]
        inputs = tokenizer(clean_context, claim, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
        
        # Маппинг результатов[cite: 1]
        res = {}
        for j, prob in enumerate(probs):
            label = id2label[j].lower()
            if 'entail' in label:
                # Применяем штраф, если субъект не совпал[cite: 1]
                res['Подтверждено'] = prob if subject_found else prob * 0.5
            elif 'contradict' in label:
                res['Противоречие'] = prob
            else:
                res['Нейтрально'] = prob
        
        verification_results.append({
            "term": term,
            "similarity": sim_score,
            "scores": res,
            "context": clean_context,
            "subject_match": subject_found
        })
    
    # Сортируем по лучшему подтверждению
    verification_results.sort(key=lambda x: x['scores'].get('Подтверждено', 0), reverse=True)
    return "SUCCESS", distances[0][0], verification_results

# --- Сайдбар с информацией ---
with st.sidebar:
    st.header("Статистика базы")
    st.info(f"Узлов в базе: {len(metadata)}")
    st.write("Порог схожести: `0.86`[cite: 1]")
    if st.button("Очистить кэш"):
        st.cache_resource.clear()

# --- Основная форма ---
user_input = st.text_area("Введите утверждение для проверки:", height=100, 
                          placeholder="Пример: Бинарное отношение — это множество пар...")

if st.button("🕵️ Проверить на галлюцинации"):
    if not user_input.strip():
        st.warning("Пожалуйста, введите текст.")
    else:
        status, max_sim, data = perform_verification(user_input)
        
        if status == "NO_DATA":
            st.error(f"Результат: **ИНФОРМАЦИЯ ОТСУТСТВУЕТ**")
            st.write(f"Максимальное сходство в базе всего **{max_sim:.2f}** (порог 0.86).")
        else:
            best_match = data[0]
            verdict = max(best_match['scores'], key=best_match['scores'].get)
            
            # Цветовое кодирование[cite: 1]
            colors = {"Подтверждено": "green", "Противоречие": "red", "Нейтрально": "orange"}
            st.subheader(f"Вердикт: :{colors[verdict]}[{verdict}]")
            
            # Метрики
            c1, c2, c3 = st.columns(3)
            c1.metric("Подтверждено", f"{best_match['scores']['Подтверждено']*100:.1f}%")
            c2.metric("Нейтрально", f"{best_match['scores']['Нейтрально']*100:.1f}%")
            c3.metric("Противоречие", f"{best_match['scores']['Противоречие']*100:.1f}%")
            
            # Детали
            with st.expander("📝 Показать обоснование (эталон из базы)"):
                st.write(f"**Наиболее релевантный термин:** {best_match['term'].capitalize()}")
                st.write(f"**Текст из базы:** {best_match['context']}")
                if not best_match['subject_match']:
                    st.warning("⚠️ Внимание: Термин из базы не найден в вашем утверждении (применен штраф к уверенности).[cite: 1]")

            # Таблица всех совпадений
            if len(data) > 1:
                st.write("---")
                st.write("### Другие найденные совпадения:")
                for item in data[1:]:
                    st.write(f"- **{item['term']}** (Сходство: {item['similarity']:.2f}, Подтверждение: {item['scores']['Подтверждено']*100:.1f}%)")