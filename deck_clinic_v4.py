import streamlit as st
import google.generativeai as genai
import os
import tempfile
import json
import pandas as pd
import csv
import datetime
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path 

# --- HELPER FUNCTIONS ---
def log_feedback(session_id, rating, comment, doc_type):
    feedback_file = "feedback_logs.csv"
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Session ID", "Rating", "Comment", "Doc Type"])
        writer.writerow([datetime.datetime.now(), session_id, rating, comment, doc_type])

def log_interaction(session_id, filename, doc_type, scores, exec_summary):
    log_file = "clinic_logs.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Session ID", "Filename", "Doc Type", "Logic Score", "Clarity Score", "Impact Score", "Summary"])
        writer.writerow([
            datetime.datetime.now(), session_id, filename, doc_type,
            scores.get('Logic', 0), scores.get('Clarity', 0), scores.get('Impact', 0),
            exec_summary
        ])

def get_score_context(score):
    """Return tier, color, and advice based on score"""
    if score >= 85:
        return "EXCELLENT", "#10b981", "🏆 Top 10%"
    elif score >= 70:
        return "GOOD", "#3b82f6", "✓ Above Average"
    elif score >= 50:
        return "NEEDS WORK", "#f59e0b", "⚠ Improvements Needed"
    else:
        return "CRITICAL", "#ef4444", "⚡ Major Issues"

# --- ADVANCED RAG FUNCTIONS ---
def advanced_rag_pipeline(draft_text, doc_type, vector_db):
    """Advanced RAG with query expansion and reciprocal rank fusion"""
    
    query_expansion_prompt = f"""
    Generate 5 focused search queries to find relevant examples for this {doc_type}:
    
    DRAFT EXCERPT:
    {draft_text[:1500]}
    
    Return JSON array: ["query1", "query2", "query3", "query4", "query5"]
    """
    
    model = genai.GenerativeModel('gemini-flash-latest')
    try:
        response = model.generate_content(
            query_expansion_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        queries = json.loads(response.text)
    except:
        queries = [draft_text[:500]]
    
    all_results = {}
    
    for i, query in enumerate(queries):
        try:
            results = vector_db.similarity_search(query, k=5)
            
            for rank, doc in enumerate(results):
                doc_id = hash(doc.page_content[:200])
                if doc_id not in all_results:
                    all_results[doc_id] = {'doc': doc, 'ranks': []}
                all_results[doc_id]['ranks'].append(rank + 1)
        except:
            continue
    
    k = 60
    for doc_id in all_results:
        ranks = all_results[doc_id]['ranks']
        rrf_score = sum(1 / (k + r) for r in ranks)
        all_results[doc_id]['rrf_score'] = rrf_score
    
    sorted_docs = sorted(
        all_results.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )
    
    final_docs = [item['doc'] for item in sorted_docs[:5]]
    
    knowledge_context = ""
    for i, doc in enumerate(final_docs, 1):
        knowledge_context += f"\n\n━━━ EXAMPLE {i} ━━━\n{doc.page_content}"
    
    return knowledge_context, queries, final_docs

def enhanced_document_indexing(uploaded_file):
    """Index documents with metadata extraction"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    raw_docs = loader.load()
    
    sample_text = "\n".join([doc.page_content for doc in raw_docs[:3]])
    
    metadata_prompt = f"""
    Analyze this document briefly:
    
    {sample_text[:2000]}
    
    Return JSON:
    {{
        "doc_type": "Strategy Deck | Product Spec | Fundraising Pitch",
        "quality_tier": "Excellent | Good | Average"
    }}
    """
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(
            metadata_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        metadata = json.loads(response.text)
    except:
        metadata = {"doc_type": "Unknown", "quality_tier": "Unknown"}
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    docs = text_splitter.split_documents(raw_docs)
    
    for doc in docs:
        doc.metadata.update({
            'doc_type': metadata.get('doc_type', 'Unknown'),
            'quality_tier': metadata.get('quality_tier', 'Unknown'),
            'upload_date': datetime.datetime.now().isoformat(),
            'filename': uploaded_file.name
        })
    
    return docs, metadata

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Deck Clinic",
    page_icon="🎠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODERN DASHBOARD CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;500;600;700;800&display=swap');
    
    /* GLOBAL RESET */
    * {
        font-family: 'Work Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    html, body, [class*="css"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* MAIN CONTAINER */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* CARD STYLES */
    .dashboard-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 10px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1), 0 16px 32px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    /* HEADERS */
    h1 {
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        font-weight: 700;
        color: #1f2937;
        letter-spacing: -0.5px;
    }
    
    /* SCORE CARDS */
    div[data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 800;
        color: #667eea;
    }
    
    div[data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.7rem;
        font-weight: 700;
        color: #6b7280;
        margin-bottom: 8px;
    }
    
    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 32px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        border-radius: 8px;
        color: #6b7280;
        font-weight: 600;
        padding: 0 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* EXPANDER */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        font-weight: 600;
        color: #374151;
        padding: 16px;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    
    /* ALERTS */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
        padding: 16px;
    }
    
    /* INFO BOXES */
    .element-container div[data-testid="stMarkdownContainer"] > div > div {
        border-radius: 10px;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e5e7eb !important;
    }
    
    /* Sidebar Selectbox */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox input {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: white !important;
    }
    
    /* Sidebar File Uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        color: #e5e7eb !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar Text Input */
    [data-testid="stSidebar"] input[type="text"],
    [data-testid="stSidebar"] input[type="password"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar Expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] details[open] > .streamlit-expanderHeader {
        border-bottom: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar Caption */
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Sidebar Metrics */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Sidebar JSON */
    [data-testid="stSidebar"] [data-testid="stJson"] {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar Divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* DIVIDER */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e5e7eb;
        opacity: 0.5;
    }
    
    /* BADGE STYLES */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    
    /* ISSUE/FIX TAGS */
    .issue-tag {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #b91c1c;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 12px;
        border-left: 4px solid #dc2626;
    }
    
    .fix-tag {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 12px;
        border-left: 4px solid #059669;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #93c5fd;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    
    /* ANIMATIONS */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .dashboard-card {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SECURITY & SETUP ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 SYSTEM ERROR: API Key Missing.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

if not os.path.exists("user_uploads"):
    os.makedirs("user_uploads")

# --- 4. CORE ENGINE ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ CONFIGURATION")
    doc_type = st.selectbox("Protocol Type", ["Strategy Deck (McKinsey/Amazon)", "Product Spec (Technical)"])
    
    st.markdown("---")
    
    st.markdown("### 📚 KNOWLEDGE BASE")
    st.caption("Upload exemplar decks to teach the system")
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")
    if uploaded_file and st.button("🔄 INDEX DOCUMENT", use_container_width=True):
        with st.spinner("Analyzing and indexing..."):
            docs, metadata = enhanced_document_indexing(uploaded_file)
            vector_db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIRECTORY)
            try: vector_db.persist()
            except: pass
            st.success(f"✅ Indexed {len(docs)} chunks")
            st.json(metadata)

    st.markdown("---")
    
    # ADMIN PANEL
    with st.expander("🔐 ADMIN ACCESS"):
        admin_pass = st.text_input("Access Key", type="password", label_visibility="collapsed")
        if admin_pass == "gemini2025": 
            st.success("✓ Authenticated")
            
            if os.path.exists("clinic_logs.csv"):
                try:
                    df_system = pd.read_csv("clinic_logs.csv", on_bad_lines='skip')
                    st.metric("Total Analyses", len(df_system))
                    st.metric("Avg Logic Score", f"{df_system['Logic Score'].mean():.0f}")
                    
                    if st.button("📥 Export Data", use_container_width=True):
                        csv = df_system.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "deck_clinic_export.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    if st.button("🗑️ Reset Database", use_container_width=True):
                        if os.path.exists("clinic_logs.csv"): os.remove("clinic_logs.csv")
                        if os.path.exists("feedback_logs.csv"): os.remove("feedback_logs.csv")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.info("No data yet")
    
    # RAG DEBUG
    with st.expander("🔬 RAG DEBUG"):
        if 'rag_queries' in st.session_state:
            for i, query in enumerate(st.session_state.rag_queries, 1):
                st.caption(f"{i}. {query[:50]}...")

# --- 6. MAIN INTERFACE ---
st.markdown("# 🎠 Deck Clinic")
st.markdown(f"**AI-Powered Strategy Deck Analyzer** · `{doc_type}` · *Built by Olivia Li*")

# Hero Card
st.markdown("""
<div class="dashboard-card">
    <h3 style="margin-top: 0;">How It Works</h3>
    <p style="color: #6b7280; margin-bottom: 0;">
        Upload your draft deck → AI analyzes logic, clarity & impact using multimodal vision + RAG → 
        Get surgical feedback with specific rewrites → Iterate and improve
    </p>
</div>
""", unsafe_allow_html=True)

# Upload Section
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("### 📄 Upload Your Deck")
    target_pdf = st.file_uploader("Drop your PDF here", type="pdf", key="target")
    
    if not target_pdf:
        st.info("👆 Upload a deck to get started")
    
    analyze_btn = st.button("🚀 RUN DIAGNOSTIC", type="primary", use_container_width=True)

with col_info:
    st.markdown("### 📊 What You'll Get")
    st.markdown("""
    - **3-Dimensional Scoring** (Logic, Clarity, Impact)
    - **Chain-of-Thought Reasoning** (See AI's analysis process)
    - **Slide-by-Slide Critique** (Specific issues + fixes)
    - **Narrative Flow Audit** (Story structure review)
    """)

# Session Management
if target_pdf and 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = target_pdf.name
    st.session_state.analysis_data = None
    st.session_state.images = None
    st.session_state.session_id = str(uuid.uuid4())[:8] 
elif target_pdf and st.session_state.get('last_uploaded') != target_pdf.name:
    st.session_state.last_uploaded = target_pdf.name
    st.session_state.analysis_data = None
    st.session_state.images = None
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Main Logic Flow
if (target_pdf and analyze_btn) or (target_pdf and st.session_state.get('analysis_data')):
    
    # PHASE A: GENERATION
    if not st.session_state.get('analysis_data'):
        
        session_id = st.session_state.session_id
        safe_filename = f"{session_id}_{target_pdf.name}"
        save_path = os.path.join("user_uploads", safe_filename)
        
        with open(save_path, "wb") as f:
            f.write(target_pdf.getbuffer())
        
        # Vision Processing
        with st.spinner("🔍 Processing slides..."):
            try:
                images = convert_from_path(save_path)
                st.session_state.images = images
            except Exception as e:
                st.warning("⚠️ Vision processing unavailable (Poppler not installed). Continuing with text-only analysis.")
                st.session_state.images = None
        
        # Text Extraction
        loader = PyPDFLoader(save_path)
        draft_docs = loader.load()
        draft_text = ""
        for i, doc in enumerate(draft_docs):
            draft_text += f"\n\n--- [PAGE {i+1}] ---\n{doc.page_content}"

        # Advanced RAG
        with st.spinner("🧠 Retrieving knowledge..."):
            try:
                vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
                knowledge_context, search_queries, retrieved_chunks = advanced_rag_pipeline(
                    draft_text=draft_text,
                    doc_type=doc_type,
                    vector_db=vector_db
                )
                st.session_state.rag_queries = search_queries
            except Exception as e:
                knowledge_context = "Standard Top Tech Company Protocols"

        # Prompt Construction (YOUR ORIGINAL PROMPT - UNCHANGED)
        base_instruction = ""
        if "Strategy" in doc_type:
            base_instruction = "ROLE: Head of Product Manager in Tech Company. FRAMEWORK: Amazon Clarity, McKinsey Structure.BLUF, Extreme Brevity."
        elif "Product" in doc_type:
            base_instruction = "ROLE: Head of Technical PM. FRAMEWORK: Feasibility checks, Spec strictness."

        prompt = f"""
{base_instruction}

### GOLD STANDARD CONTEXT:
{knowledge_context}

### DRAFT TEXT:
{draft_text[:500000]} 

### VISUAL INPUT:
I have provided images of the slides. Please analyze them alongside the text.

### ANALYSIS FRAMEWORK (THINK STEP-BY-STEP):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1: DEEP REASONING (Think Out Loud)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before scoring, analyze each dimension thoroughly:

**LOGIC REASONING:**
Step 1a: Identify the main claim/hypothesis in the deck
Step 1b: List the evidence provided (data, examples, charts)
Step 1c: Evaluate the logical chain - Does the evidence PROVE the claim? Are there gaps?
Step 1d: Check for circular reasoning or correlation≠causation errors
Step 1e: Write your reasoning (3-4 sentences explaining your logic assessment)

**CLARITY REASONING:**
Step 2a: Count abstract/vague terms (synergies, optimization, leverage, etc.)
Step 2b: Assess cognitive load - Word count per slide, concepts per slide, jargon
Step 2c: Test the "10-second rule" - Can an exec understand the slide quickly?
Step 2d: Check for visual clutter
Step 2e: Write your reasoning (3-4 sentences explaining clarity assessment)

**IMPACT REASONING:**
Step 3a: Identify the "so what?" moment
Step 3b: Check for specificity - Vague vs concrete goals
Step 3c: Assess emotional resonance - Is there urgency? Clear problem?
Step 3d: Write your reasoning (3-4 sentences explaining impact assessment)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2: SCORING (Based on Your Reasoning Above)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOW, based on the reasoning you just wrote:
- Logic Score: 0-100
- Clarity Score: 0-100
- Impact Score: 0-100

**CRITICAL:** Your scores MUST align with your reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3: EXTRACTION & RECONSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. Extract the current headlines
4. Critique the narrative flow
5. Scan for vague claims

### EXAMPLES OF GOOD CRITIQUES:

input_text: "The KSP is enable Shopee buyers to see an AI generated summary of available promotions and encourage them to buy."
critique: "1. Grammar: 'is enable' is broken. 2. Weak Metrics: 'encourage to buy' is vague; use 'conversion'."
rewrite: "Objective: Increase Shopee conversion rates by displaying AI-generated promotion summaries."

### JSON STRUCTURE:
{{
    "chain_of_thought": {{
        "logic_reasoning": "<string>",
        "clarity_reasoning": "<string>",
        "impact_reasoning": "<string>"
    }},
    "scores": {{
        "Logic": <int 0-100>,
        "Clarity": <int 0-100>,
        "Impact": <int 0-100>
    }},
    "executive_summary": "<string>",
    "narrative_check": {{
         "original_headlines": ["<string>"],
         "critique": "<string>",
         "revised_headlines": ["<string>"]
    }},
   "section_deep_dive": [
        {{
            "page_number": "<int>",
            "target_section": "<string>",
            "issue": "<string>",
            "improved_version": "<string>",
            "why": "<string>"
        }}
    ]
}}
"""

        # Generation
        with st.spinner("⚡ Analyzing with AI..."):
            try:
                model = genai.GenerativeModel('gemini-flash-latest')
                
                content_list = [prompt]
                if st.session_state.images:
                    content_list.extend(st.session_state.images)
                
                response = model.generate_content(
                    content_list, 
                    generation_config={"response_mime_type": "application/json"}
                )
                
                try:
                    st.session_state.analysis_data = json.loads(response.text)
                except json.JSONDecodeError:
                    cleaned = response.text.replace("```json", "").replace("```", "").strip()
                    st.session_state.analysis_data = json.loads(cleaned)
                
                log_interaction(
                    session_id=session_id,
                    filename=safe_filename,
                    doc_type=doc_type,
                    scores=st.session_state.analysis_data.get('scores', {}),
                    exec_summary=st.session_state.analysis_data.get('executive_summary', 'N/A')
                )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.stop()

    # PHASE B: RENDERING
    data = st.session_state.analysis_data
    
    st.markdown("---")
    
    # Scorecard Section
    st.markdown(f"### 📊 Analysis Results · Session `{st.session_state.session_id}`")
    
    logic_score = data.get('scores', {}).get('Logic', 0)
    clarity_score = data.get('scores', {}).get('Clarity', 0)
    impact_score = data.get('scores', {}).get('Impact', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 LOGIC", f"{logic_score}/100")
        tier, color, label = get_score_context(logic_score)
        st.markdown(f'<div class="status-badge" style="background-color: {color}20; color: {color}; border: 2px solid {color};">{label}</div>', unsafe_allow_html=True)
        
    with col2:
        st.metric("💬 CLARITY", f"{clarity_score}/100")
        tier, color, label = get_score_context(clarity_score)
        st.markdown(f'<div class="status-badge" style="background-color: {color}20; color: {color}; border: 2px solid {color};">{label}</div>', unsafe_allow_html=True)
        
    with col3:
        st.metric("⚡ IMPACT", f"{impact_score}/100")
        tier, color, label = get_score_context(impact_score)
        st.markdown(f'<div class="status-badge" style="background-color: {color}20; color: {color}; border: 2px solid {color};">{label}</div>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown("---")
    st.markdown(f"""
    <div class="info-box">
        <strong>📋 Executive Summary:</strong><br>
        {data.get('executive_summary', 'No summary generated.')}
    </div>
    """, unsafe_allow_html=True)
    
    # Chain of Thought
    with st.expander("🧠 See AI's Reasoning (Chain-of-Thought)", expanded=False):
        cot_data = data.get('chain_of_thought', {})
        
        st.markdown("#### 🔍 Logic Assessment")
        st.info(cot_data.get('logic_reasoning', 'No reasoning provided'))
        
        st.markdown("#### 📖 Clarity Assessment")
        st.info(cot_data.get('clarity_reasoning', 'No reasoning provided'))
        
        st.markdown("#### 💥 Impact Assessment")
        st.info(cot_data.get('impact_reasoning', 'No reasoning provided'))
    
    # Feedback Loop
    st.markdown("---")
    st.markdown("### 💬 Was this helpful?")
    
    fb_col1, fb_col2, fb_col3 = st.columns([2, 1, 1])
    with fb_col1:
        user_comment = st.text_input("", placeholder="Share your feedback (optional)...", label_visibility="collapsed")
    with fb_col2:
        if st.button("👍 Helpful", use_container_width=True):
            log_feedback(st.session_state.session_id, "Positive", user_comment, doc_type)
            st.success("Thanks!")
    with fb_col3:
        if st.button("👎 Not Helpful", use_container_width=True):
            log_feedback(st.session_state.session_id, "Negative", user_comment, doc_type)
            st.info("Noted")
    
    # Tabs
    st.markdown("---")
    tab1, tab2 = st.tabs(["📖 NARRATIVE FLOW", "🔬 DEEP DIVE"])
    
    with tab1:
        nav_data = data.get('narrative_check', {})
        
        st.markdown("#### Headline & Story Analysis")
        st.markdown(f"> {nav_data.get('critique', 'No critique available.')}")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**🔴 Original Flow**")
            for line in nav_data.get('original_headlines', []):
                st.markdown(f"• {line}")
        
        with col_b:
            st.markdown("**🟢 Optimized Flow**")
            for line in nav_data.get('revised_headlines', []):
                st.markdown(f"• **{line}**")
        
        if logic_score < 75:
            st.error("⚠️ Narrative thread has gaps")
        else:
            st.success("✅ Narrative flows logically")

    with tab2:
        st.markdown("#### Surgical Improvements")
        
        images = st.session_state.images
        
        for i, item in enumerate(data.get('section_deep_dive', [])):
            page_num = item.get('page_number', '?')
            target = item.get('target_section', 'General')
            
            st.markdown(f"##### 📄 Page {page_num}")
            
            if images:
                try:
                    p_idx = int(page_num) - 1
                    if 0 <= p_idx < len(images):
                        with st.expander(f"👁️ View Slide {page_num}"):
                            st.image(images[p_idx], use_container_width=True)
                except:
                    pass
            
            col_issue, col_fix = st.columns([1, 2])
            
            with col_issue:
                st.markdown('<div class="issue-tag">⚠️ ISSUE</div>', unsafe_allow_html=True)
                st.markdown(f"**{item.get('issue', 'N/A')}**")
                st.caption(f"💡 {item.get('why', 'N/A')}")
            
            with col_fix:
                st.markdown('<div class="fix-tag">✓ SOLUTION</div>', unsafe_allow_html=True)
                rewrite_text = item.get('improved_version', 'N/A')
                if len(rewrite_text) < 300:
                    st.success(rewrite_text)
                else:
                    st.success(rewrite_text[:300] + "...")
                    with st.expander("Show full rewrite"):
                        st.code(rewrite_text, language="text")
            
            st.markdown("---")