# Streamlit Excel-to-Chat Demo
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine

# ---------- Setup ----------
st.set_page_config(page_title="Excel Q&A", layout="wide")
st.title("📊 Chat with your Excel Data")

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error(
        "Add your OpenAI key to a .env file as `OPENAI_API_KEY=sk-...` "
        "and restart the app."
    )
    st.stop()

# ---------- Helpers ----------
def sanitize(name: str) -> str:
    """SQLite-safe table name."""
    return re.sub(r"\W+", "_", name)

def excel_to_sqlite(xl_bytes: bytes, db_path: Path) -> list[str]:
    """Load an Excel file (all sheets) into SQLite; return table names."""
    xls = pd.ExcelFile(xl_bytes)
    engine = create_engine(f"sqlite:///{db_path}")
    tables = []
    with engine.begin() as conn:
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            tbl = sanitize(sheet or "Sheet1")
            df.to_sql(tbl, conn, if_exists="replace", index=False)
            tables.append(tbl)
    return tables

@st.cache_data(show_spinner=False)
def get_chain(db_path: Path, tables: list[str]):
    """Create LangChain SQL agent for the given DB."""
    llm = OpenAI(temperature=0, api_key=openai_key)
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", include_tables=tables)
    return SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# ---------- UI ----------
uploaded = st.file_uploader(
    "Upload an Excel workbook (.xls, .xlsx)", type=["xls", "xlsx"]
)

if uploaded:
    tmp_db = Path("uploaded.db")
    tbls = excel_to_sqlite(uploaded.getvalue(), tmp_db)

    st.success(f"Loaded {len(tbls)} sheet(s) ➜ tables: {', '.join(tbls)}")

    chain = get_chain(tmp_db, tbls)

    st.markdown("#### Ask a question about your data")
    user_q = st.text_input("For example: *Total sales this month?*")

    if st.button("Ask") and user_q:
        with st.spinner("Thinking..."):
            try:
                answer = chain.run(user_q)
                st.markdown("##### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"😕 Sorry, I could not answer that.\n\n*{e}*")

    st.caption(
        "Tip: The agent auto-generates SQL using your question, runs it on the "
        "tables above, then turns the result into plain-language answers. "
        "Accuracy improves when questions are unambiguous."
    )
else:
    st.info("Upload an Excel file to begin.")
