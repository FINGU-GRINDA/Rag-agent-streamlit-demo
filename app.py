import os, uuid, base64, shutil
from io import BytesIO

import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
from byaldi import RAGMultiModalModel
from openai import OpenAI

# ------------------------------------------------------------------
# Force CPU everywhere
# ------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # make CUDA invisible
torch_dtype = "float32"                         # safe dtype for CPU

UPLOAD_ROOT = "uploads"
os.makedirs(UPLOAD_ROOT, exist_ok=True)

st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("🦾 RAG Agent – PDF Q&A")

# ------------------------------------------------------------------
# Load ColPali once (CPU-only)
# ------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Loading ColPali embeddings (first run may take a few minutes)…"
)
def load_retriever():
    return RAGMultiModalModel.from_pretrained(
        "vidore/colpali-v1.2",
        n_gpu=0,                  # <-- key: bypass auto cuda device-map
        torch_dtype=torch_dtype,
    )

retriever = load_retriever()

# session state for thumbnails
if "all_images" not in st.session_state:
    st.session_state["all_images"] = {}
if "ready" not in st.session_state:
    st.session_state["ready"] = False

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ------------------------------------------------------------------
# Sidebar – upload & index
# ------------------------------------------------------------------
st.sidebar.header("📄 Documents")
files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Build index", disabled=not files):
    tmp_dir = os.path.join(UPLOAD_ROOT, uuid.uuid4().hex)
    os.makedirs(tmp_dir, exist_ok=True)
    st.session_state["all_images"].clear()

    with st.sidebar.status("Indexing…", expanded=True) as status:
        for f in files:
            dest = os.path.join(tmp_dir, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
            status.update(label=f"Saved {f.name}")

        for doc_id, pdf in enumerate(os.listdir(tmp_dir)):
            if pdf.lower().endswith(".pdf"):
                status.update(label=f"Converting {pdf}")
                pages = convert_from_path(os.path.join(tmp_dir, pdf))
                st.session_state["all_images"][doc_id] = pages

        status.update(label="Embedding images")
        retriever.index(
            input_path=tmp_dir,
            index_name="image_index",
            store_collection_with_index=False,
            overwrite=True,
        )
        st.session_state["ready"] = True
        status.update(state="complete", label="Index ready ✅")

# ------------------------------------------------------------------
# Main – preview & ask
# ------------------------------------------------------------------
if st.session_state["ready"]:
    st.subheader("Preview")
    for doc_id, pages in st.session_state["all_images"].items():
        with st.expander(f"Document {doc_id + 1} — {len(pages)} pages"):
            cols = st.columns(5)
            for i, img in enumerate(pages[:20]):
                thumb = img.copy()
                thumb.thumbnail((300, 300))
                cols[i % 5].image(thumb, caption=f"Page {i + 1}", use_column_width=True)

    st.divider()
    st.subheader("Ask a question")

    q = st.text_area("Question", height=100)
    if st.button("Get answer", disabled=not q.strip()):
        with st.spinner("Retrieving context & querying Qwen…"):
            res = retriever.search(q, k=3)
            if not res:
                st.error("No relevant information found.")
            else:
                imgs = [
                    st.session_state["all_images"][r["doc_id"]][r["page_num"] - 1]
                    for r in res
                ]
                img_urls = [pil_to_b64(i) for i in imgs]

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for professionals working in factories.",
                    },
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image_url", "image_url": {"url": u}} for u in img_urls],
                            {"type": "text", "text": q},
                        ],
                    },
                ]

                resp = client.chat.completions.create(
                    model="qwen/qwen-2.5-vl-7b-instruct:free",
                    messages=messages,
                    temperature=0,
                )
                st.success("Answer")
                st.write(resp.choices[0].message.content)

                st.subheader("Context pages")
                ccols = st.columns(len(imgs))
                for i, img in enumerate(imgs):
                    ccols[i].image(img, use_column_width=True)
else:
    st.info("Upload PDFs in the sidebar and click **Build index** to begin.")
