import streamlit as st
import requests

API_URL = "http://localhost:8000"
KNOWLEDGE_BASE_ENDPOINT = f"{API_URL}/knowledge_base/"
VERIFY_ENDPOINT = f"{API_URL}/verify/"

st.set_page_config(page_title="Knowledge Guard Pro", layout="wide", page_icon="⚖️")

st.title("⚖️ Knowledge Guard: Knowledge Base Management")

with st.sidebar:
    st.header("⚙️ Статус Системы")
    try:
        res = requests.get(f"{API_URL}/docs", timeout=2)
        if res.status_code == 200:
            st.success("🟢 Бэкенд онлайн")
        else:
            st.warning("🟡 Ошибка API")
    except:
        st.error("🔴 Бэкенд недоступен")

    st.divider()
    st.markdown("### Справка по типам")
    st.caption("**entity_id:** Уникальное имя (напр. 'Пустое множество')")
    st.caption("**fact_type:** definition, theorem, example, constraint")

tab1, tab2, tab3 = st.tabs(["🔍 Верификация текста", "📚 Просмотр базы", "📝 Добавление/Удаление"])

with tab1:
    user_input = st.text_area(
        "Введите текст для комплексной проверки:",
        placeholder="Введите утверждение...",
        height=150
    )

    threshold = st.slider("Порог строгости (threshold)", 0.0, 1.0, 0.85, 0.05)

    if st.button("🚀 Запустить проверку", type="primary", use_container_width=True):
        if user_input.strip():
            # Показываем процесс
            with st.spinner("Нейросеть анализирует факты..."):
                try:
                    payload = {"text": user_input, "threshold": threshold}
                    response = requests.post(VERIFY_ENDPOINT, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    # Получаем данные
                    overall_status = data.get("status", "недостаточно данных").lower()
                    results = data.get("results", [])

                    # --- 1. МГНОВЕННЫЙ ОБЩИЙ ВЕРДИКТ ---
                    st.divider()

                    # Маппинг для перевода (если бэк прислал на англ или в другом формате)
                    status_map = {
                        "подтверждено": ("ПОДТВЕРЖДЕНО", "success", "✅"),
                        "confirmed": ("ПОДТВЕРЖДЕНО", "success", "✅"),
                        "галлюцинация": ("ГАЛЛЮЦИНАЦИЯ", "error", "🚨"),
                        "hallucination": ("ГАЛЛЮЦИНАЦИЯ", "error", "🚨"),
                        "contradiction": ("ГАЛЛЮЦИНАЦИЯ", "error", "🚨"),
                        "недостаточно данных": ("НЕДОСТАТОЧНО ДАННЫХ", "warning", "⚠️"),
                        "not_enough_data": ("НЕДОСТАТОЧНО ДАННЫХ", "warning", "⚠️")
                    }

                    label, style, icon = status_map.get(overall_status, (overall_status.upper(), "info", "ℹ️"))

                    # Отображаем большой баннер
                    if style == "success":
                        st.success(f"### {icon} Общий вердикт: {label}")
                    elif style == "error":
                        st.error(f"### {icon} Общий вердикт: {label}")
                    else:
                        st.warning(f"### {icon} Общий вердикт: {label}")

                    # --- 2. ДЕТАЛИЗАЦИЯ ПО ФАКТАМ ---
                    st.write("#### Детальный разбор:")

                    for i, res in enumerate(results):
                        fact_text = res.get('fact', '')
                        verdict = res.get('verdict', 'недостаточно данных').lower()
                        best_match = res.get('best_match')

                        # Определяем статус для конкретного факта
                        f_label, f_style, _ = status_map.get(verdict, (verdict.upper(), "gray", ""))

                        # Цвет текста в зависимости от статуса
                        t_color = "green" if f_style == "success" else "red" if f_style == "error" else "orange"

                        # Если галлюцинация — экспандер открыт по умолчанию (expanded=True)
                        is_problem = f_style != "success"

                        with st.expander(f"Факт №{i + 1}: {fact_text[:60]}...", expanded=is_problem):
                            st.markdown(f"**Текст:** {fact_text}")
                            st.markdown(f"**Статус:** :{t_color}[{f_label}]")

                            if best_match:
                                st.markdown(f"**Источник в базе:** `{best_match['term']}`")
                                st.info(f"**Контекст из БД:**\n{best_match['context']}")
                                st.caption(f"Схожесть: {best_match['similarity']:.4f} (порог: {threshold})")
                            else:
                                st.caption("Совпадений в базе знаний не найдено.")

                except Exception as e:
                    st.error(f"Ошибка соединения с бэкендом: {e}")

# --- TAB 2: ПРОСМОТР БАЗЫ ---
with tab2:
    st.subheader("🔍 Исследование базы знаний")
    search_id = st.text_input("Введите название термина (entity_id):", placeholder="Например: Бесконечное множество")

    if st.button("Найти в базе", type="secondary"):
        if search_id.strip():
            try:
                response = requests.get(f"{KNOWLEDGE_BASE_ENDPOINT}{search_id}")

                if response.status_code == 200:
                    fact = response.json()
                    st.success(f"### 📄 Карточка объекта: {fact['entity_id']}")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Область", fact.get('domain', 'Общая'))
                    m2.metric("Тип", fact.get('fact_type', 'Не указан'))

                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.markdown("#### 📝 Содержимое (Content)")
                        content = fact.get('content', {})
                        if content.get('natural_language'):
                            st.write(f"**Определение:** {content['natural_language']}")
                        if content.get('explanation'):
                            st.info(f"**Пояснение:** {content['explanation']}")
                        if content.get('formal_logic'):
                            st.latex(content['formal_logic'])
                        if content.get('examples'):
                            st.write("**Примеры:**")
                            for ex in content['examples']:
                                st.write(f"- {ex}")

                    with col_right:
                        st.markdown("#### 🔗 Связи (Relations)")
                        rels = fact.get('relations', {})
                        if rels:
                            if rels.get('parent_id'):
                                st.write(f"**Родитель:** {rels['parent_id']}")
                            if rels.get('conflicts_with'):
                                st.error(f"**Конфликтует с:** {', '.join(rels['conflicts_with'])}")

                elif response.status_code == 404:
                    st.error(
                        f"❌ Запись '{search_id}' не найдена. Проверьте опечатки (например, 'Бесконечное' вместо 'Бесонечное').")
                else:
                    st.error(f"Ошибка сервера: {response.status_code}")

            except Exception as e:
                st.error(f"Ошибка при рендеринге данных: {e}")
                st.info("Возможно, в базе данных отсутствуют некоторые поля. Проверьте migrate.py.")

with tab3:
    col_add, col_del = st.columns([2, 1])

    with col_add:
        st.subheader("➕ Добавить новый факт")
        with st.form("add_fact_form"):
            e_id = st.text_input("Entity ID (уникальный)*")
            domain = st.text_input("Предметная область", value="Теория Множеств")
            f_type = st.selectbox("Тип факта", ["definition", "theorem", "example", "constraint"])

            st.markdown("**Содержимое (Content)**")
            natural = st.text_area("Определение (NL)*")
            formal = st.text_input("Формальная логика (LaTeX)")
            explanation = st.text_area("Пояснение")
            examples = st.text_input("Примеры (через запятую)")

            st.markdown("**Связи (Relations)**")
            parent = st.text_input("Parent ID")
            conflicts = st.text_input("Противоречит (через запятую)")

            if st.form_submit_button("Сохранить в БД"):
                payload = {
                    "entity_id": e_id,
                    "domain": domain,
                    "fact_type": f_type,
                    "natural_language": natural,
                    "formal_logic": formal,
                    "explanation": explanation,
                    "examples": [x.strip() for x in examples.split(",")] if examples else [],
                    "parent_id": parent,
                    "depends_on": [],
                    "conflicts_with": [x.strip() for x in conflicts.split(",")] if conflicts else []
                }
                res = requests.post(KNOWLEDGE_BASE_ENDPOINT, json=payload)
                if res.status_code == 201:
                    st.success("Факт успешно добавлен!")
                else:
                    st.error(f"Ошибка: {res.json().get('detail')}")

    with col_del:
        st.subheader("🗑️ Удаление")
        del_id = st.text_input("ID для удаления", key="del_input")
        if st.button("Удалить навсегда", type="secondary"):
            if del_id:
                res = requests.delete(KNOWLEDGE_BASE_ENDPOINT, json={"entity_id": del_id})
                if res.status_code == 200:
                    st.success(f"Запись '{del_id}' удалена.")
                else:
                    st.error("Ошибка при удалении.")