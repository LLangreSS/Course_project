import streamlit as st
import requests

# --- Конфигурация ---
API_URL = "http://localhost:8000/"  # Адрес твоего FastAPI бэкенда
VERIFY_ENDPOINT = f"{API_URL}verify/"

# --- Инициализация интерфейса ---
st.set_page_config(page_title="Knowledge Guard Pro", layout="wide", page_icon="⚖️")

st.title("⚖️ Knowledge Guard: Микросервисная Архитектура")
st.markdown("""
Система использует **гибридный поиск (BM25 + FAISS)**, переранжирование через **Cross-Encoder** 
и логический вывод **NLI (BERT)**.  
*ML-модели работают на выделенном высокопроизводительном FastAPI-бэкенде.*
""")

# --- UI Sidebar ---
with st.sidebar:
    st.header("⚙️ Конфигурация")
    st.info(
        "💡 **Логика системы:** Фронтенд отправляет текст на бэкенд. FastAPI сам разбивает текст на факты, выполняет векторный и лексический поиск, применяет RRF, ранжирует кандидатов и прогоняет через нейросеть.")

    # Проверка статуса бэкенда
    try:
        res = requests.get(API_URL + "docs", timeout=2)
        if res.status_code == 200:
            st.success("🟢 Бэкенд в сети и готов к работе")
        else:
            st.warning("🟡 Бэкенд отвечает с ошибкой")
    except requests.exceptions.ConnectionError:
        st.error("🔴 Бэкенд недоступен. Запустите FastAPI!")

# --- Main UI ---
user_input = st.text_area(
    "Введите утверждение для проверки:",
    placeholder="Пример: Пустое множество содержит один элемент, а синглетон — два.",
    height=150
)

if st.button("🚀 Выполнить логический вывод", type="primary", use_container_width=True):
    if user_input.strip():
        with st.status("Связь с ML-бэкендом и анализ данных...", expanded=True) as status:
            try:
                st.write("Отправка запроса на сервер...")
                # Отправляем POST запрос на наш FastAPI
                response = requests.post(VERIFY_ENDPOINT, json={"text": user_input})
                response.raise_for_status()  # Проверка на ошибки (500, 404 и тд)

                data = response.json()
                results = data.get("results", [])
                status.update(label="Анализ успешно завершен!", state="complete", expanded=False)

            except requests.exceptions.RequestException as e:
                status.update(label="Ошибка соединения с бэкендом!", state="error")
                st.error(f"Не удалось связаться с API: {e}")
                st.stop()

        st.divider()

        # --- Определение общего вердикта ---
        verdicts = [r['verdict'] for r in results]
        if "противоречие" in verdicts:
            st.error("🚨 **ВЕРДИКТ: ОБНАРУЖЕНО ЛОГИЧЕСКОЕ ПРОТИВОРЕЧИЕ / ГАЛЛЮЦИНАЦИЯ**")
        elif all(v == "подтверждено" for v in verdicts):
            st.success("✅ **ВЕРДИКТ: ИНФОРМАЦИЯ ПОЛНОСТЬЮ ВЕРИФИЦИРОВАНА**")
        else:
            st.warning("⚠️ **ВЕРДИКТ: ЧАСТИЧНАЯ ВЕРИФИКАЦИЯ ИЛИ НЕХВАТКА ДАННЫХ**")

        # --- Красивый вывод каждого факта ---
        for i, res in enumerate(results):
            fact_text = res['claim']
            verdict = res['verdict']
            confidence = res['confidence'] * 100

            with st.expander(f"Факт {i + 1}: {fact_text[:80]}...", expanded=True):
                st.write(f"**Текст:** {fact_text}")

                # Цветовая индикация
                color = {
                    "подтверждено": "green",
                    "противоречие": "red",
                    "недостаточно контекста": "orange"
                }.get(verdict, "gray")

                c1, c2 = st.columns([1, 3])
                with c1:
                    st.subheader(f"Статус: :{color}[{verdict.upper()}]")
                    st.metric("Уверенность нейросети", f"{confidence:.1f}%")

                with c2:
                    if verdict != "нет данных":
                        st.markdown(f"### 📚 Обоснование из базы знаний:")
                        st.caption(f"**Идентифицированный термин:** `{res['entity_name']}`")
                        st.info(res['source_text'])
                    else:
                        st.warning("Для данного утверждения бэкенд не нашел подходящих данных в базе знаний.")