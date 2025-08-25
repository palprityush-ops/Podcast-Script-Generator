import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# ------------------ Setup ------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --------- Streamlit Page Config ----------
st.set_page_config(
    page_title="🎙️ RAG Podcast Script Generator",
    page_icon="🎧",
    layout="wide",
)

# --------- Custom CSS ----------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #1f1c2c, #928DAB);
            color: white;
        }
        h1 {
            text-align: center;
            font-size: 3rem !important;
            color: #f5f5f5 !important;
        }
        .stTextArea textarea {
            border-radius: 12px;
            background-color: #2c2c38 !important;
            color: white !important;
        }
        .stButton>button {
            border-radius: 12px;
            background: #6a11cb;
            background: linear-gradient(315deg, #6a11cb 0%, #2575fc 74%);
            color: white;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(315deg, #2575fc 0%, #6a11cb 74%);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------- App Header ----------
st.markdown("<h1>🎙️ RAG Podcast Script Generator</h1>", unsafe_allow_html=True)
st.write("✨ Convert **news articles or transcripts** into a polished podcast script with AI.")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=120)
    st.markdown("## ⚙️ Options")
    st.info("Upload or paste your transcript/news article and click **Generate**.")
    st.markdown("Made with ❤️ using **Streamlit + LangChain + Gemini**")

# Embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------- Input Section ----------
st.subheader("📝 Input Content")
text_data = st.text_area("Paste transcript or news article below:", height=200, placeholder="Paste your transcript here...")

# --------- Button & Processing ----------
if st.button("🚀 Generate Podcast Script"):
    if not text_data.strip():
        st.error("Please provide text input")
        st.stop()

    st.info("📌 Starting pipeline...")

    try:
        # Step 1: Documents
        docs = [Document(page_content=text_data)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        st.write(f"🔹 Split into {len(split_docs)} chunks")

        # Step 2: FAISS
        vectorstore = FAISS.from_documents(split_docs, embedder)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        query = "Generate a structured podcast script (intro, 3 segments, outro, show notes)"
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_text = " ".join([d.page_content for d in retrieved_docs])
        st.success("✅ Retrieved top chunks")

        # Step 3: Prompt
        prompt = f"""
        You are a professional podcast script writer.
        Transform the retrieved transcript into a podcast script.

        Structure:
        1. Intro (engaging, 30-60s)
        2. 3 Segments with headings + smooth transitions
        3. Outro with call-to-action
        4. Show Notes (3 bullets + 3 tags)

        Source Material:
        {retrieved_text}
        """
        st.write("⚡ Sending request to Gemini...")

        # Step 4: Gemini Call
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        script = response.text if hasattr(response, "text") else str(response)

        # Step 5: Output
        st.subheader("🎙️ Generated Podcast Script")
        st.markdown(f"<div style='background:#2c2c38; padding:15px; border-radius:12px;'>{script}</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Script", script, file_name="podcast_script.md")

        st.success("Pipeline completed ✅")

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
        logger.exception("Pipeline crashed")
