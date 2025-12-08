import streamlit as st
import PyPDF2
from docx import Document
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# ==================== CONFIGURAZIONE PAGINA ==================== #
st.set_page_config(
    page_title="Document AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CSS STILE CHATGPT ==================== #
st.markdown("""
<style>
    /* Sfondo principale dark mode */
    .stApp {
        background-color: #343541;
    }
    
    /* Nasconde elementi default Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Container principale chat */
    .main-chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header stile ChatGPT */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .chat-header h1 {
        color: white;
        font-size: 2em;
        margin: 0;
        font-weight: 600;
    }
    
    .chat-header p {
        color: rgba(255,255,255,0.9);
        margin: 5px 0 0 0;
        font-size: 0.95em;
    }
    
    /* Messaggi chat */
    .stChatMessage {
        background-color: #444654 !important;
        border: none !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        padding: 16px !important;
    }
    
    /* Messaggio utente */
    .stChatMessage[data-testid="user-message"] {
        background-color: #343541 !important;
        border-left: 3px solid #667eea !important;
    }
    
    /* Messaggio AI */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #444654 !important;
        border-left: 3px solid #10a37f !important;
    }
    
    /* Input chat */
    .stChatInputContainer {
        background-color: #40414f !important;
        border-radius: 12px !important;
        border: 1px solid #565869 !important;
        margin-top: 20px !important;
    }
    
    /* Pulsanti */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar dark */
    section[data-testid="stSidebar"] {
        background-color: #202123 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ececf1 !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #2d2e35;
        border-radius: 10px;
        padding: 15px;
        border: 1px dashed #565869;
    }
    
    /* Selectbox */
    .stSelectbox {
        color: white !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #2d2e35 !important;
        border-radius: 10px !important;
        border-left: 4px solid #667eea !important;
        color: #ececf1 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Text input */
    .stTextInput input {
        background-color: #40414f !important;
        color: white !important;
        border: 1px solid #565869 !important;
        border-radius: 8px !important;
    }
    
    /* Info badge */
    .info-badge {
        display: inline-block;
        background: #10a37f;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 5px;
    }
    
    /* Status indicator */
    .status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #10a37f;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ==================== #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

# ==================== FUNZIONI ==================== #
def extract_pdf_to_text(pdf_path, output_txt_path):
    """Estrae testo da PDF e lo salva in un file .txt"""
    try:
        text = ""
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                text += f"--- Pagina {page_num + 1} ---\n{page_text}\n\n"
        
        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
        return True
    except Exception as e:
        st.error(f"❌ Errore estrazione: {str(e)}")
        return False

def setup_groq_llm(api_key):
    """Configura Groq API (gratuita e velocissima)"""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        def llm_call(prompt):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Sei un assistente AI specializzato nell'analisi di documenti. Rispondi in modo chiaro e preciso basandoti sul contesto fornito."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-70b-versatile",
                temperature=0.5,
                max_tokens=1024
            )
            return response.choices[0].message.content
        
        return llm_call
    except Exception as e:
        st.error(f"❌ Errore Groq API: {str(e)}")
        return None

def setup_gemini_llm(api_key):
    """Configura Google Gemini API (gratuita con 2M context)"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        def llm_call(prompt):
            response = model.generate_content(prompt)
            return response.text
        
        return llm_call
    except Exception as e:
        st.error(f"❌ Errore Gemini API: {str(e)}")
        return None

def process_document(document_path, doc_name):
    """Elabora il documento e crea il retriever"""
    try:
        text = ""
        
        # Estrazione testo
        if isinstance(document_path, str) and document_path.endswith(".txt"):
            with open(document_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        elif isinstance(document_path, str) and document_path.endswith(".pdf"):
            with open(document_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += (page.extract_text() or "") + "\n"
        
        elif isinstance(document_path, str) and document_path.endswith(".docx"):
            doc = Document(document_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        else:  # File uploadato
            if hasattr(document_path, "name") and document_path.name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(document_path.read())
                    tmp_path = tmp.name
                with open(tmp_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text += (page.extract_text() or "") + "\n"
                os.remove(tmp_path)
            
            elif hasattr(document_path, "name") and document_path.name.endswith(".docx"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(document_path.read())
                    tmp_path = tmp.name
                doc = Document(tmp_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                os.remove(tmp_path)
        
        if not text.strip():
            return None, "Nessun testo estratto dal documento"
        
        # Chunking
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Embeddings e vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        return retriever, f"✅ {len(chunks)} sezioni indicizzate"
        
    except Exception as e:
        return None, f"❌ Errore: {str(e)}"

# ==================== SIDEBAR ==================== #
with st.sidebar:
    st.markdown("### ⚙️ Configurazione")
    st.markdown("---")
    
    # Selezione API
    st.markdown("#### 🔑 API Provider")
    api_provider = st.selectbox(
        "Scegli il provider AI:",
        ["Groq (Consigliato)", "Google Gemini", "HuggingFace"],
        help="Groq è gratuito e velocissimo!"
    )
    
    # API Key
    if api_provider == "Groq (Consigliato)":
        api_key = st.text_input(
            "Groq API Key:",
            type="password",
            help="Gratuita su console.groq.com"
        )
    elif api_provider == "Google Gemini":
        api_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Gratuita su aistudio.google.com"
        )
    else:
        api_key = st.text_input(
            "HuggingFace Token:",
            type="password",
            help="Su huggingface.co/settings/tokens"
        )
    
    st.markdown("---")
    
    # Caricamento documento
    st.markdown("#### 📄 Documento")
    
    if not os.path.exists("documents"):
        os.makedirs("documents")
    
    repo_files = [
        f for f in os.listdir("documents")
        if f.endswith((".pdf", ".docx")) and not f.endswith("_testo.txt")
    ]
    
    document_path = None
    doc_name = ""
    
    if repo_files:
        st.markdown("**File nella repository:**")
        selected_file = st.selectbox("📁 Seleziona:", repo_files)
        document_path = os.path.join("documents", selected_file)
        doc_name = selected_file
        
        if selected_file.endswith(".pdf"):
            txt_path = document_path.replace(".pdf", "_testo.txt")
            if not os.path.exists(txt_path):
                with st.spinner("📖 Estrazione testo..."):
                    extract_pdf_to_text(document_path, txt_path)
            document_path = txt_path
    
    uploaded_file = st.file_uploader(
        "📤 Oppure carica un file:",
        type=["pdf", "docx"],
        help="PDF o DOCX"
    )
    
    if uploaded_file:
        document_path = uploaded_file
        doc_name = uploaded_file.name
    
    st.markdown("---")
    
    # Pulsante elabora
    if document_path:
        if st.button("🚀 Elabora Documento", use_container_width=True):
            if not api_key:
                st.warning("⚠️ Inserisci prima l'API key!")
            else:
                with st.spinner("🔄 Elaborazione in corso..."):
                    # Setup LLM
                    if api_provider == "Groq (Consigliato)":
                        st.session_state.llm = setup_groq_llm(api_key)
                    elif api_provider == "Google Gemini":
                        st.session_state.llm = setup_gemini_llm(api_key)
                    else:
                        # HuggingFace Hub
                        from langchain_community.llms import HuggingFaceHub
                        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
                        llm = HuggingFaceHub(
                            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                            model_kwargs={"temperature": 0.5, "max_length": 512}
                        )
                        st.session_state.llm = lambda prompt: llm(prompt)
                    
                    # Processa documento
                    retriever, message = process_document(document_path, doc_name)
                    
                    if retriever:
                        st.session_state.retriever = retriever
                        st.session_state.document_loaded = True
                        st.session_state.document_name = doc_name
                        st.session_state.chat_history = []
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Status
    st.markdown("---")
    st.markdown("#### 📊 Status")
    if st.session_state.document_loaded:
        st.markdown(
            f'<div><span class="status-online"></span>Documento caricato</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**File:** `{st.session_state.document_name}`")
    else:
        st.markdown("⏸️ In attesa di documento...")
    
    # Reset
    if st.button("🔄 Nuova Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ==================== MAIN CHAT AREA ==================== #
# Header
st.markdown("""
<div class="chat-header">
    <h1>🤖 Document AI Assistant</h1>
    <p>Powered by AI • Analisi intelligente dei tuoi documenti</p>
</div>
""", unsafe_allow_html=True)

# Stato documento
if st.session_state.document_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div style="text-align:center; color: #ececf1;"><span class="info-badge">📄 {st.session_state.document_name}</span></div>',
            unsafe_allow_html=True
        )
else:
    st.info("👈 Carica un documento dalla sidebar per iniziare")

# Chat messages
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
else:
    if st.session_state.document_loaded:
        st.markdown("""
        <div style="text-align:center; color: #8e8ea0; padding: 40px;">
            <p style="font-size: 1.2em;">💬 Inizia a fare domande sul documento!</p>
            <p>Esempi:</p>
            <p>• "Riassumi il contenuto principale"</p>
            <p>• "Quali sono i punti chiave?"</p>
            <p>• "Cerca informazioni su [argomento]"</p>
        </div>
        """, unsafe_allow_html=True)

# Chat input
if st.session_state.retriever and st.session_state.llm:
    user_input = st.chat_input("💭 Scrivi la tua domanda...")
    
    if user_input:
        # Aggiungi messaggio utente
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Genera risposta
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Sto pensando..."):
                try:
                    # Recupera documenti rilevanti
                    docs = st.session_state.retriever.get_relevant_documents(user_input)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Crea prompt
                    prompt = f"""Basandoti SOLO sul seguente contesto, rispondi alla domanda in modo chiaro e conciso.
Se la risposta non si trova nel contesto, dì "Non ho trovato informazioni sufficienti nel documento per rispondere a questa domanda."

CONTESTO:
{context}

DOMANDA: {user_input}

RISPOSTA:"""
                    
                    # Genera risposta
                    response = st.session_state.llm(prompt)
                    
                    # Mostra risposta
                    st.markdown(response)
                    
                    # Salva in history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    
                except Exception as e:
                    error_msg = f"❌ Errore: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
