import os, uuid, hashlib, base64, shutil
from io import BytesIO
from pathlib import Path

import streamlit as st
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
from byaldi import RAGMultiModalModel
from openai import OpenAI

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
CACHE_DIR = Path("data/cache")           # persisted between restarts
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)

GLOBAL_INDEX = CACHE_DIR / "image_index"   # .json + .faiss

# --------------------------------------------------------------------
# UI config
# --------------------------------------------------------------------
st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("🦾 RAG Agent – PDF Q&A")

# --------------------------------------------------------------------
# Load ColPali once (CPU-only)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ColPali embeddings…")
def load_retriever():
    model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device="cpu")
    # Load existing global index if present
    if (GLOBAL_INDEX.with_suffix(".json")).exists():
        try:
            model.load(str(GLOBAL_INDEX))
            st.sidebar.info("Global index loaded from cache.")
        except Exception as e:
            st.sidebar.warning(f"Could not load cache: {e}")
    return model

retriever = load_retriever()

# Keep thumbnails in session
if "images" not in st.session_state:
    st.session_state["images"] = {}   # sha -> list[PIL]
if "ready" not in st.session_state:
    st.session_state["ready"] = False

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def pdf_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:20]

def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# --------------------------------------------------------------------
# Sidebar – upload & index
# --------------------------------------------------------------------
st.sidebar.header("📄 Documents")
files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Build / Load index", disabled=not files):
    new_folder = UPLOAD_ROOT / uuid.uuid4().hex
    new_folder.mkdir()

    new_pdfs = []          # [(sha, path)]
    for f in files:
        data = f.read()
        sha = pdf_sha(data)
        pdf_path = new_folder / f"{sha}.pdf"
        pdf_path.write_bytes(data)

        # Convert pages for preview (always, so we have images)
        try:
            pages = convert_from_path(pdf_path, dpi=150)
        except PDFInfoNotInstalledError:
            st.error("Poppler is missing. Make sure packages.txt has `poppler-utils`.")
            st.stop()

        st.session_state["images"][sha] = pages

        # Only embed if this PDF hasn't been cached yet
        if not (GLOBAL_INDEX.with_suffix(".json")).exists() or sha not in retriever.model.doc_ids:
            new_pdfs.append(pdf_path)

    # Embed batch if there are new PDFs
    if new_pdfs:
        with st.spinner("Embedding new PDFs…"):           # ← use st.spinner, not st.sidebar.spinner
    retriever.index(
        input_path=str(embed_dir),
        index_name=str(GLOBAL_INDEX),
        store_collection_with_index=True,
        overwrite=False
    )
    st.sidebar.success("New PDFs embedded and cached.")

        

    st.session_state["ready"] = True

# --------------------------------------------------------------------
# Main – ask questions
# --------------------------------------------------------------------
if st.session_state["ready"]:
    # Preview thumbnails
    st.subheader("Preview")
    for sha, pages in st.session_state["images"].items():
        with st.expander(f"Document {sha} — {len(pages)} pages"):
            cols = st.columns(5)
            for i, img in enumerate(pages[:20]):
                thumb = img.copy()
                thumb.thumbnail((300, 300))
                cols[i % 5].image(thumb, caption=f"Page {i + 1}", use_column_width=True)

    st.divider()
    st.subheader("Ask a question")

    q = st.text_area("Question", height=100)
    if st.button("Get answer", disabled=not q.strip()):
        with st.spinner("Searching & querying Qwen…"):
            hits = retriever.search(q, k=3)
            if not hits:
                st.error("No relevant information found.")
                st.stop()

            ctx_imgs = [st.session_state["images"][h['doc_id']][h['page_num'] - 1]
                        for h in hits]
            urls = [pil_to_b64(i) for i in ctx_imgs]

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            msg = [
                {"role": "system", "content": "You are a helpful assistant for factory professionals."},
                {"role": "user", "content": [*[
                    {"type": "image_url", "image_url": {"url": u}} for u in urls
                ], {"type": "text", "text": q}]}
            ]
            resp = client.chat.completions.create(
                model="qwen/qwen-2.5-vl-7b-instruct:free",
                messages=msg,
                temperature=0
            )

        st.success("Answer")
        st.write(resp.choices[0].message.content)

        st.subheader("Context pages")
        ccols = st.columns(len(ctx_imgs))
        for i, img in enumerate(ctx_imgs):
            ccols[i].image(img, use_column_width=True)
else:
    st.info("Upload PDFs and click **Build / Load index** to begin.")
