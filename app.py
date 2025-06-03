import os, uuid, base64, shutil
from io import BytesIO

import streamlit as st
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
from byaldi import RAGMultiModalModel
from openai import OpenAI

# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
UPLOAD_ROOT = "uploads"
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("🦾 RAG Agent – PDF Q&A")

# ------------------------------------------------------------------
# load ColPali on **CPU** (Streamlit Cloud has no GPU)
# ------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Loading ColPali embeddings (first run may take a few minutes)…"
)
def load_retriever():
    return RAGMultiModalModel.from_pretrained(
        "vidore/colpali-v1.2",
        device="cpu"                # <-- crucial
    )

retriever = load_retriever()

# thumbnails & flags in session
if "all_images" not in st.session_state:
    st.session_state["all_images"] = {}
if "ready" not in st.session_state:
    st.session_state["ready"] = False

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ------------------------------------------------------------------
# sidebar – upload & index
# ------------------------------------------------------------------
st.sidebar.header("📄 Documents")
files = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

if st.sidebar.button("Build index", disabled=not files):
    tmp_dir = os.path.join(UPLOAD_ROOT, uuid.uuid4().hex)
    os.makedirs(tmp_dir, exist_ok=True)
    st.session_state["all_images"].clear()

    with st.sidebar.status("Indexing…", expanded=True) as status:
        # save PDFs
        for f in files:
            path = os.path.join(tmp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            status.update(label=f"Saved {f.name}")

        # convert pages
        for doc_id, pdf in enumerate(os.listdir(tmp_dir)):
            if pdf.lower().endswith(".pdf"):
                status.update(label=f"Converting {pdf}")
                try:
                    pages = convert_from_path(os.path.join(tmp_dir, pdf))
                except PDFInfoNotInstalledError:
                    st.error(
                        "Poppler is missing. "
                        "Make sure `packages.txt` contains `poppler-utils`."
                    )
                    st.stop()
                st.session_state["all_images"][doc_id] = pages

        # embed
        status.update(label="Embedding images")
        retriever.index(
            input_path=tmp_dir,
            index_name="image_index",
            store_collection_with_index=True,   # keep passages so search works
            overwrite=True,
        )
        st.session_state["ready"] = True
        status.update(state="complete", label="Index ready ✅")

# ------------------------------------------------------------------
# main – preview & ask
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
            try:
                res = retriever.search(q, k=3)
            except ValueError as e:        # e.g. “No passages provided”
                st.error(f"Search failed: {e}")
                st.stop()

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
                        "content": (
                            "You are a helpful assistant for professionals "
                            "working in factories."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            *[
                                {"type": "image_url", "image_url": {"url": u}}
                                for u in img_urls
                            ],
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
