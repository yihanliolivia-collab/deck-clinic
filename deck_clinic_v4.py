import streamlit as st
import google.generativeai as genai
import os
import tempfile
import json
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. CONFIGURATION & CYBER-TECH UI ---
st.set_page_config(
    page_title="Deck Clinic V5",
    page_icon="🧬",
    layout="wide"
)

# Custom "High-Tech" CSS - NOW WITH FIXED TEXT COLORS
st.markdown("""
<style>
    /* 1. Force Background and Text Colors */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0; /* Light Gray Text for readability */
    }
    
    /* 2. Global Font: Monospace for that 'Terminal' feel */
    body, .stMarkdown, p, h1, h2, h3, li, .stMetricLabel {
        font-family: 'Courier New', Courier, monospace !important;
        color: #E0E0E0 !important;
    }
    
    /* 3. Neon Accents for Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #00FFC2 !important; /* Neon Cyan */
        text-shadow: 0 0 10px rgba(0, 255, 194, 0.5);
        font-weight: 700;
    }
    
    /* 4. Card/Container Styling */
    .stExpander {
        border: 1px solid #333;
        border-radius: 4px;
        background-color: #161B22; /* Slightly lighter dark for contrast */
    }
    
    /* 5. Fix Text Inside Expanders (This was likely the black-on-black issue) */
    .streamlit-expanderContent p, .streamlit-expanderContent li {
        color: #E0E0E0 !important;
    }
    
    /* 6. Custom Buttons */
    div.stButton > button {
        background-color: #1E1E1E;
        color: #00FFC2;
        border: 1px solid #00FFC2;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00FFC2;
        color: #000000 !important; /* Black text on hover */
        box-shadow: 0 0 15px rgba(0, 255, 194, 0.6);
    }
    
    /* 7. Code Blocks */
    code {
        color: #FF79C6 !important; /* Pink for code highlights */
        background-color: #282A36 !important;
    }
    
    /* 8. Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. SECURITY & SETUP ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 SYSTEM HALTED: API Key Missing.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# --- 3. CORE ENGINE ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 4. SIDEBAR: CONTROL PANEL ---
with st.sidebar:
    st.title("🎛️ CONTROL PANEL")
    
    # Context Selector
    doc_type = st.selectbox(
        "Protocol Selection",
        ["Strategy Deck (McKinsey/Amazon)", "Product Spec (Technical)", "Exec Update (Brief)"]
    )
    
    st.divider()
    
    # Knowledge Base Uploader
    st.caption("SYSTEM: Neural Knowledge Base")
    uploaded_file = st.file_uploader("Upload 'Gold Standard' PDF", type="pdf")
    
    if uploaded_file and st.button("INITIATE TRAINING"):
        with st.spinner("...Ingesting Data Streams..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            raw_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            docs = text_splitter.split_documents(raw_docs)
            
            vector_db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIRECTORY)
            try: vector_db.persist()
            except: pass
            st.toast(f"✅ System Upgraded: {len(docs)} chunks added.")

# --- 5. MAIN INTERFACE ---
st.title("🧬 DECK CLINIC: V5")
st.markdown(f"**active_protocol:** `{doc_type}`")
st.caption(f"**model_core:** `gemini-flash-latest`") 

col1, col2 = st.columns([2, 3]) 

with col1:
    st.markdown("### 📂 INPUT SOURCE")
    target_pdf = st.file_uploader("Upload Draft PDF", type="pdf", key="target")
    analyze_btn = st.button(">> EXECUTE DIAGNOSTIC", type="primary", use_container_width=True)

if target_pdf and analyze_btn:
    # A. File Processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(target_pdf.read())
        draft_path = tmp_file.name
    
    loader = PyPDFLoader(draft_path)
    draft_docs = loader.load()
    draft_text = " ".join([d.page_content for d in draft_docs])

    # B. RAG Retrieval
    with st.spinner("...Accessing Neural Database..."):
        try:
            vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            results = vector_db.similarity_search(draft_text, k=3)
            knowledge_context = "\n".join([doc.page_content for doc in results])
        except:
            knowledge_context = "Standard Top Tech Company Protocols"

    # --- C. DYNAMIC PROMPT LOGIC ---
    base_instruction = ""
    
    if "Strategy" in doc_type:
        base_instruction = """
        ROLE: VP of Strategy (Amazon/McKinsey background).
        FRAMEWORK:
        1. **Amazon Clarity:** Grade 8 reading level. No big words. No long sentences (>2 commas).
        2. **McKinsey Structure:** MECE principle. Pyramid Principle. "Golden Thread" logic.
        """
    elif "Product" in doc_type:
        base_instruction = """
        ROLE: Senior Technical PM.
        FRAMEWORK:
        1. **Feasibility:** Identify missing edge cases or technical risks.
        2. **Specs:** Ensure metrics and success criteria are defined strictly.
        """
    else: # Exec Update
        base_instruction = """
        ROLE: CEO.
        FRAMEWORK:
        1. **BLUF:** Bottom Line Up Front.
        2. **Brevity:** If it can be said in 5 words, do not use 10.
        """

    prompt = f"""
    {base_instruction}
    
    ### GOLD STANDARD CONTEXT:
    {knowledge_context}
    
    ### DRAFT TEXT:
    {draft_text[:50000]} 
    
    ### INSTRUCTIONS:
    1. Analyze the text based on the FRAMEWORK above.
    2. **CRITICAL STEP:** Extract the 'Narrative Flow'. Identify the Headline/Topic Sentence of every distinct section. List them in order to test the story flow.
    3. Output in JSON only.

    ### JSON STRUCTURE:
    {{
        "scores": {{
            "Logic": <int 0-100>,
            "Clarity": <int 0-100>,
            "Impact": <int 0-100>
        }},
        "executive_summary": "<string: Brutal one-sentence summary>",
        "narrative_check": [
             "<string: Slide/Section 1 Headline>",
             "<string: Slide/Section 2 Headline>",
             "<string: Slide/Section 3 Headline>"
        ],
        "critical_issues": [
            {{ "section": "<string>", "issue": "<string>", "fix": "<string>" }}
        ],
        "rewrite_showcase": {{
            "original_text": "<string>",
            "improved_version": "<string>",
            "why": "<string>"
        }}
    }}
    """

    # D. Generation
    with st.spinner("...Processing Logic Matrix..."):
        # UPDATED: Using the latest alias as requested
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
    # E. Rendering
    try:
        data = json.loads(response.text)
        
        with col2:
            st.markdown("### 📊 DIAGNOSTICS")
            s1, s2, s3 = st.columns(3)
            s1.metric("LOGIC", f"{data['scores'].get('Logic',0)}")
            s2.metric("CLARITY", f"{data['scores'].get('Clarity',0)}")
            s3.metric("IMPACT", f"{data['scores'].get('Impact',0)}")
        
        st.divider()
        
        # --- NEW TABS LAYOUT ---
        tab1, tab2, tab3 = st.tabs(["🔗 STORY FLOW", "🛑 CRITICAL GAPS", "✨ REWRITE SHOWCASE"])
        
        with tab1:
            st.markdown("#### The Narrative Check (Pyramid Principle)")
            st.caption("Does the story hold together reading ONLY these lines?")
            with st.container(border=True):
                for i, line in enumerate(data.get('narrative_check', [])):
                    st.code(f"{i+1}. {line}", language="markdown")
            
            if data['scores'].get('Logic', 0) < 75:
                st.error("⚠️ ALERT: Narrative thread appears broken. Re-order sections.")
            else:
                st.success("✅ Narrative thread is coherent.")
        
        with tab2:
            st.info(f"**EXECUTIVE SUMMARY:** {data['executive_summary']}")
            for item in data['critical_issues']:
                with st.expander(f"📍 {item['section']}"):
                    st.write(f"**Error:** {item['issue']}")
                    st.markdown(f"**Correction:** `{item['fix']}`")
        
        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("ORIGINAL")
                st.code(data['rewrite_showcase']['original_text'], language="text")
            with c2:
                st.caption("OPTIMIZED")
                st.code(data['rewrite_showcase']['improved_version'], language="text")
            st.markdown(f"> **Why:** {data['rewrite_showcase']['why']}")

    except Exception as e:
        st.error(f"Data Stream Parsing Error: {e}")
        with st.expander("DEBUG DATA"):
            st.text(response.text)