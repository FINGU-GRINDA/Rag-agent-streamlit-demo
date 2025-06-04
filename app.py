# app.py – Streamlit demo: chat with any Excel workbook
import os
import re
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.llms import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# ---------- Streamlit setup ----------
st.set_page_config(page_title="Excel Q&A", layout="wide")
st.title("📊 Chat with your Excel Data")

# ---------- ENV / Keys ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error(
        "Add your OpenAI API key to a .env file as\n\n"
        "OPENAI_API_KEY=sk-...\n\nand restart the app."
    )
    st.stop()

# ---------- Helpers ----------
def sanitize(name: str) -> str:
    """Return a SQLite-safe table name."""
    return re.sub(r"\W+", "_", name.strip()) or "Sheet1"


def excel_to_sqlite(xl_bytes: bytes, db_path: Path) -> list[str]:
    """
    Load all sheets from an Excel file (bytes) into SQLite.
    Returns a list of created table names.
    """
    xls = pd.ExcelFile(io.BytesIO(xl_bytes))
    engine = create_engine(f"sqlite:///{db_path}")
    tables: list[str] = []

    with engine.begin() as conn:
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            tbl_name = sanitize(sheet)
            df.to_sql(tbl_name, conn, if_exists="replace", index=False)
            tables.append(tbl_name)
    return tables


@st.cache_data(show_spinner=False)
def get_chain(db_path: Path, tables: list[str]) -> SQLDatabaseChain:
    """Instantiate a LangChain SQL agent for the given SQLite DB."""
    llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY)
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", include_tables=tables)
    return SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)


# ---------- UI ----------
uploaded_file = st.file_uploader(
    "Upload an Excel workbook (.xls or .xlsx)",
    type=["xls", "xlsx"],
    help="Each sheet becomes a table in a temporary SQLite database."
)

if uploaded_file:
    db_path = Path("uploaded.db")

    try:
        table_names = excel_to_sqlite(uploaded_file.getvalue(), db_path)
    except Exception as err:
        st.error(f"Failed to read Excel file: {err}")
        st.stop()

    st.success(
        f"Loaded {len(table_names)} sheet(s) as table(s): "
        f"{', '.join(table_names)}"
    )

    chain = get_chain(db_path, table_names)

    st.markdown("#### Ask questions about the data")
    question = st.text_input("For example: *Total sales by month for 2025-05?*")

    if st.button("Ask") and question:
        with st.spinner("Thinking…"):
            try:
                answer = chain.run(question)
                st.markdown("##### Answer")
                st.write(answer)
            except Exception as err:
                st.error(f"Sorry, could not answer that: {err}")

    st.caption(
        "The agent translates your question to SQL, executes it on the tables "
        "above, then converts the result to natural language. For best accuracy, "
        "use clear column names and ask specific questions."
    )
else:
    st.info("👆 Upload an Excel file to start chatting with your data.")
