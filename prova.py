import streamlit as st
import PyPDF2
from docx import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ==================== ⬇️ INSERISCI QUI IL PATH DEL TUO FILE ⬇️ ==================== #
DOCUMENT_PATH = "documents/Project Work DACA Network Traffic Analyzer (3).pdf"
# ==================================================================================== #

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
        color: #FFFFFF !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #FFFFFF !important;
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
        color: white !important;
        font-size: 2.2em;
        margin: 0;
        font-weight: 600;
    }
    
    .chat-header p {
        color: rgba(255,255,255,0.95) !important;
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
        color: #FFFFFF !important;
    }
    
    .stChatMessage p, .stChatMessage span, .stChatMessage div {
        color: #FFFFFF !important;
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
        color: white !important;
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
        color: #FFFFFF !important;
    }
    
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10a37f 0%, #0d8a6b 100%);
        color: white !important;
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
        color: #FFFFFF !important;
        margin: 40px 0;
        border: 2px solid #40414f;
    }
    
    .welcome-box h3 {
        color: #10a37f !important;
        margin-bottom: 15px;
    }
    
    .welcome-box p {
        color: #FFFFFF !important;
    }
    
    .example-question {
        background: #40414f;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #FFFFFF !important;
        cursor: pointer;
        transition: all 0.3s;
        border-left: 3px solid #667eea;
    }
    
    .example-question:hover {
        background: #4a4b5a;
        transform: translateX(5px);
    }
    
    .context-box {
        background: #2d2e35;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border-left: 3px solid #10a37f;
        color: #FFFFFF !important;
    }
    
    .context-box h4 {
        color: #10a37f !important;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ==================== #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = "basic"  # "basic" o "advanced"

# ==================== GESTIONE API KEY SICURA ==================== #
def get_api_key():
    """Prova a recuperare API key da Streamlit secrets (sicuro)"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

GROQ_API_KEY = get_api_key()

# ==================== FUNZIONI ==================== #
def generate_smart_response(context, question):
    """
    Genera una risposta intelligente SENZA API ESTERNA
    Usa solo analisi del contesto e pattern matching
    """
    context_lower = context.lower()
    question_lower = question.lower()
    
    # Riassunto
    if any(word in question_lower for word in ["riassumi", "riassunto", "sintesi", "overview", "sommario"]):
        # Prendi le prime 3 frasi significative
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 50]
        summary = '. '.join(sentences[:5]) + '.'
        
        return f"""📋 **Riassunto del documento:**

{summary}

---
💡 *Questo è un estratto automatico delle sezioni più rilevanti del documento. Per un'analisi più approfondita, puoi configurare l'AI avanzata.*"""
    
    # Cerca parole chiave nella domanda
    keywords = [word for word in question_lower.split() if len(word) > 4]
    
    # Filtra le frasi del contesto che contengono le keyword
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in keywords):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        response = "📄 **Informazioni trovate nel documento:**\n\n"
        for i, sent in enumerate(relevant_sentences[:5], 1):
            if sent:
                response += f"{i}. {sent}.\n\n"
        
        response += "\n---\n💡 *Risposta generata dall'analisi semantica del documento.*"
        return response
    else:
        return f"""❓ **Contenuto correlato:**

{context[:1000]}...

---
💡 *Non ho trovato una corrispondenza esatta per la tua domanda. Prova a riformularla o attiva l'AI avanzata per risposte più elaborate.*"""

def query_groq_api(prompt, context):
    """Usa Groq API solo se disponibile"""
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        messages = [
            {
                "role": "system",
                "content": "Sei un assistente AI esperto nell'analisi di documenti tecnici. Rispondi sempre in italiano in modo chiaro, preciso e dettagliato. Basa le tue risposte SOLO sul contesto fornito."
            },
            {
                "role": "user",
                "content": f"""Contesto dal documento:
{context}

Domanda: {prompt}

Rispondi in modo dettagliato e strutturato in italiano, utilizzando SOLO le informazioni presenti nel contesto."""
            }
        ]
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
            top_p=0.9
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return None

def extract_text_from_document(file_path):
    """Estrae testo da PDF con migliore parsing"""
    try:
        text = ""
        
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    page_text = page_text.replace('\x00', '')
                    text += page_text + "\n\n"
                
                st.info(f"📄 Estratte {total_pages} pagine dal PDF")
        
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
    """Elabora il documento con chunking ottimizzato"""
    try:
        text, error = extract_text_from_document(file_path)
        if error:
            return None, None, error
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separator="\n"
        )
        chunks = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        return vectorstore, text, f"✅ Documento elaborato: {len(chunks)} sezioni indicizzate"
        
    except Exception as e:
        return None, None, f"Errore elaborazione: {str(e)}"

# ==================== HEADER ==================== #
st.markdown("""
<div class="chat-header">
    <h1>🤖 Document AI Assistant</h1>
    <p>Analisi intelligente dei documenti - Funziona sempre, anche senza API key!</p>
</div>
""", unsafe_allow_html=True)

# ==================== INFO MODALITÀ ==================== #
col1, col2 = st.columns([3, 1])
with col1:
    if GROQ_API_KEY:
        st.success("🚀 **Modalità AI Avanzata** - Groq Llama 3.3 70B attiva")
        st.session_state.ai_mode = "advanced"
    else:
        st.info("⚡ **Modalità Base** - Analisi semantica attiva (nessuna API richiesta)")
        with st.expander("💡 Vuoi risposte più elaborate?"):
            st.markdown("""
            **Attiva l'AI Avanzata (opzionale):**
            
            1. Ottieni una chiave gratuita su [console.groq.com](https://console.groq.com)
            2. Su Streamlit Cloud: Settings → Secrets → aggiungi:
               ```
               GROQ_API_KEY = "gsk_tua_chiave"
               ```
            3. In locale: crea `.streamlit/secrets.toml` con la stessa chiave
            
            **La modalità base funziona già perfettamente!** 👍
            """)

# ==================== CARICAMENTO AUTOMATICO ==================== #
if not st.session_state.document_loaded:
    with st.spinner("🔄 Caricamento e analisi documento in corso..."):
        vectorstore, full_text, message = process_document(DOCUMENT_PATH)
        
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.document_loaded = True
            st.success(message)
            st.rerun()
        else:
            st.error(f"❌ {message}")
            st.warning(f"⚠️ Verifica che il file esista: `{DOCUMENT_PATH}`")
            st.stop()

# ==================== STATUS DOCUMENTO ==================== #
if st.session_state.document_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode_text = "AI Avanzata" if st.session_state.ai_mode == "advanced" else "Analisi Base"
        st.markdown(
            f'<div style="text-align:center;"><span class="status-badge"><span class="status-online"></span>{mode_text} • Pronta</span></div>',
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
        <p>L'assistente analizzerà il documento per rispondere alle tue domande</p>
        <div class="example-question">📋 Fai un riassunto del documento</div>
        <div class="example-question">🔍 Quali sono i concetti principali?</div>
        <div class="example-question">📊 Cerca informazioni su [argomento]</div>
        <div class="example-question">❓ Spiega [argomento specifico]</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== INPUT CHAT ==================== #
if st.session_state.vectorstore:
    user_input = st.chat_input("💭 Scrivi la tua domanda qui...")
    
    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Analisi in corso..."):
                try:
                    # Determina numero di chunks
                    if any(word in user_input.lower() for word in ["riassumi", "riassunto", "sintesi", "sommario", "overview"]):
                        k_chunks = 8
                    else:
                        k_chunks = 5
                    
                    # Cerca nel documento
                    docs = st.session_state.vectorstore.similarity_search(user_input, k=k_chunks)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Genera risposta
                    if st.session_state.ai_mode == "advanced":
                        # Prova con Groq
                        response = query_groq_api(user_input, context)
                        if not response:
                            # Fallback a modalità base
                            response = generate_smart_response(context, user_input)
                            response = "⚠️ *AI avanzata temporaneamente non disponibile. Risposta in modalità base:*\n\n" + response
                    else:
                        # Modalità base (sempre funzionante)
                        response = generate_smart_response(context, user_input)
                    
                    st.markdown(response)
                    
                    # Mostra contesto (opzionale)
                    with st.expander("📄 Vedi il contesto estratto dal documento"):
                        st.text(context[:1500] + "..." if len(context) > 1500 else context)
                    
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    
                except Exception as e:
                    error_msg = f"❌ Errore: {str(e)}"
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
if st.session_state.ai_mode == "advanced":
    st.markdown("""
    <div style="text-align: center; color: #FFFFFF; padding: 20px;">
        <p><b>🚀 Modalità Avanzata</b> • Powered by Groq Llama 3.3 70B</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; color: #FFFFFF; padding: 20px;">
        <p><b>⚡ Modalità Base</b> • Analisi semantica con ricerca vettoriale</p>
        <p style="font-size: 0.9em;">Nessuna API key richiesta • Funziona sempre • Privacy totale</p>
    </div>
    """, unsafe_allow_html=True)
