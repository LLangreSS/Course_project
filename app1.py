import streamlit as st
import requests

API_URL = "http://localhost:8000/verify/"

st.set_page_config(page_title="IS Knowledge Guard", layout="wide", page_icon="🛡️")

st.title("🛡️ Верификатор знаний (Enterprise Edition)")
st.markdown("""
Система глубокого логического анализа утверждений. 
Использует **строгий режим верификации**: если сумма сомнений и противоречий больше уверенности, факт отклоняется.
""")

# --- Сайдбар ---
with st.sidebar:
    st.header("⚙️ Настройки")
    SIMILARITY_THRESHOLD = st.slider("Строгость поиска", 0.70, 0.99, 0.86, 0.01)

# --- Основной экран ---
user_input = st.text_area("Введите утверждение:", height=120,
                          placeholder="Например: Конечные множества имеют фиксированное число элементов...")

if st.button("🕵️ Проверить", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Введите текст.")
    else:
        with st.status("Отправка запроса на сервер...", expanded=True) as status:
            try:
                # 1. Отправляем запрос на наш FastAPI бэкенд
                response = requests.post(
                    API_URL,
                    json={"text": user_input, "threshold": SIMILARITY_THRESHOLD}
                )
                response.raise_for_status()  # Проверка на ошибки (500, 404)
                data = response.json()  # Парсим JSON-ответ

                status.update(label="Анализ завершен!", state="complete", expanded=False)
            except requests.exceptions.RequestException as e:
                status.update(label="Ошибка соединения с сервером", state="error")
                st.error(f"Не удалось связаться с API: {e}")
                st.stop()

        # 2. Вывод итогового вердикта (на основе статуса от бэкенда)
        st.divider()
        if data["has_contradiction"]:
            st.error("🚨 **ВЕРДИКТ: СОДЕРЖИТ ГАЛЛЮЦИНАЦИЮ**")
        elif data["status"] == "NO_DATA":
            st.warning("❓ **ВЕРДИКТ: НЕТ ДАННЫХ В БАЗЕ**")
        elif data["status"] == "PARTIAL_SUCCESS":
            st.warning("⚠️ **ВЕРДИКТ: ЧАСТИЧНО ПОДТВЕРЖДЕНО (Не для всех фактов найден контекст)**")
        else:
            st.success("✅ **ВЕРДИКТ: ИНФОРМАЦИЯ ПОДТВЕРЖДЕНА**")

        for i, res in enumerate(data["results"]):
            with st.expander(f"Детализация: Факт {i + 1}", expanded=True):
                st.write(f"**Текст:** {res['fact']}")

                verdict = res['verdict']
                cl = {"Подтверждено": "green", "Противоречие": "red", "Нейтрально": "orange"}.get(verdict, "gray")
                st.subheader(f"Статус: :{cl}[{verdict}]")

                b = res.get('best_match')
                if b:
                    # Метрики
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Подтверждение", f"{b['scores']['Подтверждено'] * 100:.1f}%")
                    c2.metric("Нейтрально", f"{b['scores']['Нейтрально'] * 100:.1f}%")
                    c3.metric("Противоречие", f"{b['scores']['Противоречие'] * 100:.1f}%")

                    st.info(f"**Источник (БД):** {b['term'].capitalize()} — {b['context']}")

                    # Предупреждения
                    if not b['subject_match']:
                        st.error(
                            "⚠️ **Ошибка субъекта**: Термины не совпадают. Система заблокировала ложное подтверждение.")
                    elif b['scores']['Подтверждено'] == 0.0 and verdict != "Противоречие":
                        st.warning(
                            "⚠️ **Пессимистичный фильтр**: Сумма сомнений выше уверенности. Вердикт пересмотрен.")

                    # Отрисовка альтернативных вариантов (если они есть)
                    if len(res.get('all_matches', [])) > 1:
                        st.write("---")
                        st.write("🔄 **Альтернативные совпадения из базы:**")
                        for alt_match in res['all_matches'][1:]:
                            st.caption(f"- **{alt_match['term']}** (Сходство: {alt_match['similarity']:.2f})")