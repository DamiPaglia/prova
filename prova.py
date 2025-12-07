import streamlit as st
import PyPDF2
from docx import Document
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile

st.set_page_config(page_title="📚 Document AI Chatbot",
                   layout="wide",
                   initial_sidebar_state="expanded")

# -------------------- UI HEADER -------------------- #
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #666;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 Document AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fai domande al tuo documento PDF o DOCX</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------- SESSION STATE -------------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# -------------------- FUNZIONE: estrai testo PDF -------------------- #
def extract_pdf_to_text(pdf_path, output_txt_path):
    """Estrae testo da PDF e lo salva in un file .txt nascosto."""
    try:
        text = ""
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                text += f"--- Pagina {page_num + 1} ---\n"
                text += page_text + "\n\n"

        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)

        return True
    except Exception as e:
        st.error(f"❌ Errore nell'estrazione: {str(e)}")
        return False

# -------------------- SIDEBAR -------------------- #
with st.sidebar:
    st.header("⚙️ Configurazione")
    st.write("### 📖 Carica il tuo documento:")

    # Cartella documents
    if not os.path.exists("documents"):
        os.makedirs("documents")

    # File presenti nella repo
    repo_files = [
        f for f in os.listdir("documents")
        if f.endswith((".pdf", ".docx")) and not f.endswith("_testo.txt")
    ]

    document_path = None
    use_repo_file = False

    if repo_files:
        st.write("**File disponibili nella repo:**")
        for f in repo_files:
            st.write(f"✅ {f}")

        selected_file = st.selectbox("Seleziona un file:", repo_files)
        document_path = os.path.join("documents", selected_file)
        use_repo_file = True

        # Se è un PDF, crea/usa il .txt nascosto
        if selected_file.endswith(".pdf"):
            txt_filename = selected_file.replace(".pdf", "_testo.txt")
            txt_path = os.path.join("documents", txt_filename)

            if not os.path.exists(txt_path):
                with st.spinner("📄 Estrazione testo dal PDF in corso..."):
                    ok = extract_pdf_to_text(document_path, txt_path)
                    if ok:
                        st.success("✅ Testo estratto e pronto!")

            document_path = txt_path
    else:
        st.write("*Nessun file trovato nella cartella 'documents'*")

    st.markdown("---")

    # Upload manuale
    uploaded_file = st.file_uploader("Oppure carica un nuovo file:", type=["pdf", "docx"])

    if uploaded_file is not None:
        use_repo_file = False
        document_path = uploaded_file
    elif not use_repo_file and not repo_files:
        st.warning("⚠️ Carica un documento PDF o DOCX per iniziare")
        document_path = None

    st.markdown("---")

    # API key HF
    st.write("### 🔑 API Key Hugging Face (Opzionale)")
    hf_api_key = st.text_input(
        "Inserisci la tua chiave API:",
        type="password",
        help="Crea un token su https://huggingface.co/settings/tokens"
    )

    if not hf_api_key:
        st.info("💡 Usi un modello free. Per risultati migliori, aggiungi la tua API key Hugging Face")

# -------------------- LAYOUT PRINCIPALE -------------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 💬 Chat")
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    else:
        st.info("📝 Carica un documento e fai una domanda per iniziare!")

with col2:
    st.write("### 📊 Info")
    if st.session_state.document_loaded:
        st.success("✅ Documento caricato!")
    else:
        st.warning("⏳ Carica un documento...")

# -------------------- ELABORAZIONE DOCUMENTO -------------------- #
if document_path and not st.session_state.document_loaded:
    with st.spinner("📖 Elaborazione del documento..."):
        try:
            text = ""

            # Caso: file di testo creato dall'estrazione
            if isinstance(document_path, str) and document_path.endswith(".txt"):
                with open(document_path, "r", encoding="utf-8") as txt_file:
                    text = txt_file.read()

            # Caso: PDF su disco
            elif isinstance(document_path, str) and document_path.endswith(".pdf"):
                with open(document_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"

            # Caso: DOCX su disco
            elif isinstance(document_path, str) and document_path.endswith(".docx"):
                doc = Document(document_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"

            # Caso: file caricato via upload
            else:
                if hasattr(document_path, "name") and document_path.name.endswith(".pdf"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(document_path.read())
                        tmp_path = tmp.name
                    with open(tmp_path, "rb") as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page in pdf_reader.pages:
                            page_text = page.extract_text() or ""
                            text += page_text + "\n"
                    os.remove(tmp_path)

                elif hasattr(document_path, "name") and document_path.name.endswith(".docx"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                        tmp.write(document_path.read())
                        tmp_path = tmp.name
                    doc = Document(tmp_path)
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    os.remove(tmp_path)

            if text:
                # Split in chunk con CharacterTextSplitter (nuovo pacchetto)
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(text)
                st.info(f"📊 Documento suddiviso in {len(chunks)} sezioni")

                # Embeddings e vettori
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vectorstore = FAISS.from_texts(chunks, embeddings)

                # LLM
                if hf_api_key:
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
                    llm = HuggingFaceHub(
                        repo_id="HuggingFaceH4/zephyr-7b-beta",
                        model_kwargs={"temperature": 0.7, "max_length": 512}
                    )
                else:
                    llm = HuggingFaceHub(
                        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                        model_kwargs={"temperature": 0.7, "max_length": 512}
                    )

                # Catena QA
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                )

                st.session_state.document_loaded = True
                st.success("✅ Documento elaborato! Ora puoi fare domande.")
                st.rerun()
            else:
                st.error("❌ Nessun testo estratto dal documento")

        except Exception as e:
            st.error(f"❌ Errore nell'elaborazione: {str(e)}")

# -------------------- CHAT -------------------- #
if st.session_state.qa_chain:
    user_input = st.chat_input("Fai una domanda al documento...", key="chat_input")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        with st.spinner("🤔 Sto cercando la risposta..."):
            try:
                response = st.session_state.qa_chain.run(user_input)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()
            except Exception as e:
                st.error(f"❌ Errore nella risposta: {str(e)}")
else:
    if document_path:
        st.info("⏳ Attendere l'elaborazione del documento...")

st.markdown("---")
st.write("💡 **Pro Tips:**")
st.write("- Metti i PDF/DOCX nella cartella 'documents' della repo")
st.write("- I file di testo estratti vengono generati automaticamente")
st.write("- Aggiungi una API key Hugging Face per risposte migliori")
st.write("- Fai domande specifiche basate sul contenuto del documento")
