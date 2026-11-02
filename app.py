"""Інтерфейс стенда: питання, режим пошуку, джерела і рівень довіри."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from ingest import get_collection, ingest_file
from pipeline import run
from trust_score import trust

st.set_page_config(page_title="Оцінювання відповідей RAG", layout="wide")
st.title("Підтримка рішень: відповідь і довіра до неї")
st.caption("Генератор відповідає лише через pipeline.run. Довіра рахується окремо і не змінює текст відповіді.")

with st.sidebar:
    st.header("База документів")
    try:
        st.metric("Чанків у базі", get_collection().count())
    except Exception as error:
        st.warning(f"Базу ще не зібрано: {error}")
    uploads = st.file_uploader(
        "PDF, DOCX або TXT",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Додати в базу"):
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            total = 0
            for upload in uploads:
                destination = config.DOCS_DIR / Path(upload.name).name
                destination.write_bytes(upload.getvalue())
                total += ingest_file(destination)
        except Exception as error:
            st.error(f"Не вдалося додати документи: {error}")
        else:
            st.success(f"Додано чанків: {total}")

with st.form("ask"):
    mode = st.selectbox(
        "Режим",
        list(config.MODES),
        format_func=lambda item: {
            "none": "без пошуку",
            "local": "лише документи",
            "web": "лише веб",
            "combined": "документи і веб",
        }[item],
    )
    question = st.text_area("Питання", height=100)
    include_ragas = st.checkbox("Включити RAGAS у розрахунок довіри", value=False)
    show_trust = st.checkbox("Порахувати рівень довіри", value=True)
    submitted = st.form_submit_button("Відповісти", type="primary")

if submitted and not question.strip():
    st.warning("Введіть питання.")
elif submitted:
    try:
        with st.spinner("Формую відповідь"):
            result = run(question.strip(), mode)
    except Exception as error:
        st.error(f"Не вдалося отримати відповідь: {error}")
    else:
        st.subheader("Відповідь")
        st.write(result["answer"])
        st.caption(f"Час: {result['latency']:.1f} с. Токени генерації: {result['tokens']}.")
        st.subheader("Джерела")
        if result["contexts"]:
            for index, chunk in enumerate(result["contexts"], start=1):
                st.markdown(f"**[{index}] {chunk['source']}** · {chunk['origin']}")
                st.write(chunk["text"])
        else:
            st.info("Контекст не використовувався.")
        if show_trust:
            try:
                with st.spinner("Оцінюю довіру"):
                    report = trust(
                        question.strip(),
                        result["contexts"],
                        result["answer"],
                        include_ragas=include_ragas,
                    )
            except Exception as error:
                st.error(f"Відповідь є, але довіру порахувати не вдалося: {error}")
            else:
                st.metric("Рівень довіри", report["level"], f"{report['score']:.2f}")
                st.caption(f"Модель показника: {report['model']}")
                st.json(report["breakdown"])
