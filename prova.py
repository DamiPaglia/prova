import streamlit as st
import PyPDF2
from docx import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import json

# ==================== ⬇️ INSERISCI QUI IL PATH DEL TUO FILE ⬇️ ==================== #
DOCUMENT_PATH = "Project Work DACA Network Traffic Analyzer (3).pdf"  # 👈 RIGA 12: MODIFICA QUI!
# ==================================================================================== #

# Configurazione pagina
st.set_page_config(
    page_title="Document AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CSS STILE CHATGPT ==================== #
st.markdown("""
<style>
    .stApp {
        background-color: #343541;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .chat-header h1 {
        color: white;
        font-size: 2.2em;
        margin: 0;
        font-weight: 600;
    }
    
    .chat-header p {
        color: rgba(255,255,255,0.95);
        margin: 8px 0 0 0;
        font-size: 1em;
    }
    
    .stChatMessage {
        background-color: #444654 !important;
        border: none !important;
        border-radius: 12px !important;
        margin-bottom: 15px !important;
        padding: 18px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background-color: #343541 !important;
        border-left: 4px solid #667eea !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #444654 !important;
        border-left: 4px solid #10a37f !important;
    }
    
    .stChatInputContainer {
        background-color: #40414f !important;
        border-radius: 12px !important;
        border: 2px solid #565869 !important;
        margin-top: 20px !important;
        padding: 5px !important;
    }
    
    .stChatInputContainer textarea {
        background-color: #40414f !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.5);
    }
    
    .stAlert {
        background-color: #2d2e35 !important;
        border-radius: 12px !important;
        border-left: 4px solid #667eea !important;
        color: #ececf1 !important;
    }
    
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10a37f 0%, #0d8a6b 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 0.95em;
        font-weight: 600;
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(16, 163, 127, 0.3);
    }
    
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
    
    .welcome-box {
        background: #2d2e35;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: #ececf1;
        margin: 40px 0;
        border: 2px solid #40414f;
    }
    
    .welcome-box h3 {
        color: #10a37f;
        margin-bottom: 15px;
    }
    
    .example-question {
        background: #40414f;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #c5c5d2;
        cursor: pointer;
        transition: all 0.3s;
        border-left: 3px solid #667eea;
    }
    
    .example-question:hover {
        background: #4a4b5a;
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ==================== #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# ==================== FUNZIONI ==================== #
def query_huggingface_api(prompt):
    """
    Chiama l'API HuggingFace SENZA TOKEN - Completamente gratuita
    Usa modelli pubblici con rate limiting ma senza autenticazione
    """
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return str(result)
        
        elif response.status_code == 503:
            return "⏳ Il modello si sta caricando, riprova tra 20 secondi..."
        
        else:
            return f"❌ Errore API (codice {response.status_code})"
    
    except Exception as e:
        return f"❌ Errore di connessione: {str(e)}"

def extract_text_from_document(file_path):
    """Estrae testo da PDF o DOCX"""
    try:
        text = ""
        
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text += f"\n--- Pagina {page_num + 1} ---\n{page_text}\n"
        
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        else:
            return None, "Formato file non supportato. Usa PDF, DOCX o TXT"
        
        if not text.strip():
            return None, "Nessun testo trovato nel documento"
        
        return text, None
    
    except Exception as e:
        return None, f"Errore lettura file: {str(e)}"

@st.cache_resource
def process_document(file_path):
    """Elabora il documento e crea il retriever"""
    try:
        text, error = extract_text_from_document(file_path)
        if error:
            return None, error
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        chunks = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        return retriever, f"Documento elaborato: {len(chunks)} sezioni indicizzate"
        
    except Exception as e:
        return None, f"Errore elaborazione: {str(e)}"

# ==================== HEADER ==================== #
st.markdown("""
<div class="chat-header">
    <h1>🤖 Document AI Assistant</h1>
    <p>Intelligenza Artificiale gratuita per analizzare i tuoi documenti</p>
</div>
""", unsafe_allow_html=True)

# ==================== CARICAMENTO AUTOMATICO ==================== #
if not st.session_state.document_loaded:
    with st.spinner("🔄 Caricamento documento in corso..."):
        retriever, message = process_document(DOCUMENT_PATH)
        
        if retriever:
            st.session_state.retriever = retriever
            st.session_state.document_loaded = True
            st.success(f"✅ {message}")
            st.info("🤖 AI pronta! Usa l'API HuggingFace gratuita (senza token richiesto)")
            st.rerun()
        else:
            st.error(f"❌ {message}")
            st.warning(f"⚠️ Verifica che il file esista: `{DOCUMENT_PATH}`")
            st.stop()

# ==================== STATUS DOCUMENTO ==================== #
if st.session_state.document_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div style="text-align:center;"><span class="status-badge"><span class="status-online"></span>Documento caricato e pronto</span></div>',
            unsafe_allow_html=True
        )

# ==================== AREA CHAT ==================== #
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
else:
    st.markdown("""
    <div class="welcome-box">
        <h3>💬 Benvenuto! Inizia a chattare con il documento</h3>
        <p>Prova a fare domande come:</p>
        <div class="example-question">📋 Riassumi il contenuto principale del documento</div>
        <div class="example-question">🔍 Quali sono i punti chiave trattati?</div>
        <div class="example-question">📊 Dammi informazioni specifiche su [argomento]</div>
        <div class="example-question">❓ Cosa dice il documento riguardo a [tema]?</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== INPUT CHAT ==================== #
if st.session_state.retriever:
    user_input = st.chat_input("💭 Scrivi la tua domanda qui...")
    
    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Sto analizzando il documento..."):
                try:
                    docs = st.session_state.retriever.get_relevant_documents(user_input)
                    context = "\n\n".join([doc.page_content for doc in docs[:3]])
                    
                    prompt = f"""[INST] You are a helpful AI assistant. Answer the question based ONLY on the following context from a document. If the answer is not in the context, say you don't have that information.

Context:
{context}

Question: {user_input}

Provide a clear and concise answer in Italian. [/INST]"""
                    
                    response = query_huggingface_api(prompt)
                    
                    st.markdown(response)
                    
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    
                except Exception as e:
                    error_msg = f"❌ Errore durante la generazione della risposta: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )

# ==================== PULSANTE RESET ==================== #
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Nuova Conversazione", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ==================== FOOTER INFO ==================== #
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8e8ea0; padding: 20px;">
    <p>💡 <b>Info:</b> Questa app usa l'API gratuita di HuggingFace (Mistral 7B) senza necessità di token.</p>
    <p>⚡ Prima richiesta potrebbe richiedere 20 secondi (caricamento modello).</p>
</div>
""", unsafe_allow_html=True)
