import os, uuid, hashlib, json, base64, shutil, time
from io import BytesIO
from pathlib import Path

import streamlit as st
from pdf2image import convert_from_path, PDFInfoNotInstalledError
from PIL import Image
from byaldi import RAGMultiModalModel
from openai import OpenAI

# ---------------- config ----------------
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)

st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("🦾 RAG Agent – PDF Q&A")

@st.cache_resource(show_spinner="Loading ColPali…")
def load_retriever():
    return RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device="cpu")

retriever = load_retriever()

# -------------- helpers -----------------
def pdf_sha(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:20]

def save_pdf(tmp_dir: Path, file_name: str, data: bytes) -> Path:
    p = tmp_dir / file_name
    with open(p, "wb") as f: f.write(data)
    return p

def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO(); img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# -------------- sidebar -----------------
st.sidebar.header("📄 Documents")
files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Build / Load index", disabled=not files):
    tmp_dir = UPLOAD_ROOT / uuid.uuid4().hex
    tmp_dir.mkdir()

    sha_list, new_pdfs = [], []
    for f in files:
        data = f.read()
        sha = pdf_sha(data)
        sha_list.append(sha)
        if not (CACHE_DIR / f"{sha}.json").exists():
            p = save_pdf(tmp_dir, f.name, data)
            new_pdfs.append((sha, p))

    if not sha_list:
        st.sidebar.warning("Nothing to index.")
        st.stop()

    # --- step 1: (re)build missing PDFs ---
    if new_pdfs:
        with st.sidebar.status("Indexing new PDFs…", expanded=True) as status:
            for sha, path in new_pdfs:
                status.update(label=f"Converting {path.name}")
                try:
                    pages = convert_from_path(path, dpi=150)
                except PDFInfoNotInstalledError:
                    st.error("Poppler missing – ensure packages.txt has poppler-utils")
                    st.stop()

                status.update(label=f"Embedding {path.name}")
                retriever.add_images(pages, doc_id=sha)   # stream in

                retriever.save(str(CACHE_DIR / sha))      # writes .json & .faiss
                status.update(label=f"Cached {sha}")

    # --- step 2: load all selected PDFs into current retriever ---
    retriever.reset()
    for sha in sha_list:
        retriever.load(str(CACHE_DIR / sha))

    st.session_state["sha_list"] = sha_list
    st.session_state["ready"] = True
    st.sidebar.success("Index ready ✅")

# -------------- main ----------------
if st.session_state.get("ready"):
    q = st.text_area("Ask your question")
    if st.button("Get answer", disabled=not q.strip()):
        with st.spinner("Searching & querying Qwen…"):
            hits = retriever.search(q, k=3)
            if not hits:
                st.error("No relevant information found.")
                st.stop()

            # gather context images
            ctx_imgs = [hit["image"] for hit in hits]     # byaldi returns PILs
            urls = [pil_to_b64(img) for img in ctx_imgs]

            client = OpenAI(base_url="https://openrouter.ai/api/v1",
                            api_key=os.getenv("OPENROUTER_API_KEY"))
            msg = [
                {"role": "system", "content": "You are a helpful factory assistant."},
                {"role": "user", "content": [*[
                    {"type":"image_url","image_url":{"url":u}} for u in urls
                ], {"type":"text","text": q}]}
            ]
            resp = client.chat.completions.create(
                model="qwen/qwen-2.5-vl-7b-instruct:free",
                messages=msg, temperature=0
            )
        st.success("Answer")
        st.write(resp.choices[0].message.content)
else:
    st.info("Upload PDFs and click **Build / Load index**.")
