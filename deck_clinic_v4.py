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

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Deck Clinic: Approval Accelerator",
    page_icon="💾",
    layout="wide"
)

# --- 2. CSS STYLING (Gemini / Google Style) ---
st.markdown("""
<style>
    /* A. Gemini-Style Font (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* B. Clean Headers */
    h1, h2, h3, h4, h5 {
        font-weight: 700;
        color: #202124; /* Google Dark Grey */
        letter-spacing: -0.5px;
    }
    
    /* C. Metric Cards (Clean & Minimal) */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        background-color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* D. Buttons (Pill Shape / Google Style) */
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid #dadce0;
        font-weight: 600;
        transition: all 0.2s;
        background-color: #ffffff;
        color: #3c4043;
    }
    div.stButton > button:hover {
        background-color: #f1f3f4;
        border-color: #dadce0;
        transform: translateY(-1px);
        color: #202124;
    }
    
    /* E. CLINIC CARD DESIGN (Deep Dive) */
    .issue-tag {
        background-color: #fce8e6; /* Google Red Light */
        color: #c5221f; /* Google Red Dark */
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .fix-tag {
        background-color: #e6f4ea; /* Google Green Light */
        color: #137333; /* Google Green Dark */
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .logic-footer {
        font-size: 0.85rem;
        color: #5f6368;
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
        border: 1px solid #f1f3f4;
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

# --- 4. CORE ENGINE ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 5. SIDEBAR: CONTROL PANEL ---
with st.sidebar:
    st.title("🎛️ SETTINGS")
    
    # Context Selector
    doc_type = st.selectbox(
        "DIAGNOSTIC PROTOCOL",
        ["Strategy Deck (McKinsey/Amazon)", "Product Spec (Technical)", "Exec Update (Brief)"]
    )
    
    st.divider()
    
    # Knowledge Base Uploader
    st.caption("📂 KNOWLEDGE BASE")
    uploaded_file = st.file_uploader("Upload 'Gold Standard' PDF: Deck best practice if you have one.Don't worry, we have added best example in our database.", type="pdf")
    
    if uploaded_file and st.button("TRAIN SYSTEM"):
        with st.spinner("Indexing..."):
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
            st.success(f"System Index Updated: {len(docs)} chunks.")

# --- 6. MAIN INTERFACE ---
st.title(" 🎠DECK Playground: Your Powerful Deck GP")
st.caption(f"PROTOCOL: {doc_type} | CORE: gemini-flash-latest") 

col1, col2 = st.columns([2, 3]) 

with col1:
    # ✅ These lines are indented exactly 4 spaces
    st.markdown("### INPUT FEED")
    target_pdf = st.file_uploader("Upload Draft PDF", type="pdf", key="target")
    analyze_btn = st.button("RUN DIAGNOSTIC", type="primary", use_container_width=True)

if target_pdf and analyze_btn:
    # A. File Processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(target_pdf.read())
        draft_path = tmp_file.name
    
    loader = PyPDFLoader(draft_path)
    draft_docs = loader.load()  # <--- Make sure this line exists!
    
    # ✅ FIX: This loop must be INDENTED to match the line above
    draft_text = ""
    for i, doc in enumerate(draft_docs):
        draft_text += f"\n\n--- [PAGE {i+1}] ---\n{doc.page_content}"
    # B. RAG Retrieval
    with st.spinner("Retrieving Context..."):
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

    # --- V6 PROMPT STRUCTURE ---
    prompt = f"""
    {base_instruction}
    
    ### GOLD STANDARD CONTEXT:
    {knowledge_context}
    
    ### DRAFT TEXT:
    {draft_text[:50000]} 
    
    ### INSTRUCTIONS:
    1. **STEP 1 (HIDDEN BRAINSTORM):** Read the text. Specifically look for logical gaps. Ask yourself: "Does the problem prove the solution?" "Is the data specific?"
    2. **STEP 2 (SCORING):** Only assign scores AFTER you have written the critique.
    3. **STEP 3 (EXTRACTION):** Extract the current headlines to identify the existing narrative.
    4. **STEP 4 (Headline & Narrative Audit):**
       - Critique the current headlines: Do they tell a story if read in isolation? Are they descriptive or generic?
       - Suggest a **"Revised Headline Flow"**: A list of rewritten headlines that guide the reader logically from the problem to the solution.
    5. **STEP 5 (CONTENT RIGOR):** Scan the **body paragraphs and bullet points** for vague claims (e.g., "significant growth", "optimized synergies"). Ignore the headlines for this step.
   
    ### EXAMPLES OF GOOD CRITIQUES (FEW-SHOT):
    
    input_text: "The KSP is enable Shopee buyers to see an AI generated summary of available promotions and encourage them to buy. In this deck, we will discuss the logic of the input of promotion summary first, then show the front end demo and share the examples of different generated example in words."
    critique: "Low Clarity. The first sentence is grammatically broken. 'Encourage them to buy' is weak. The storyline lacks a clear reasoning flow."
    rewrite: "Objective: Increase Shopee conversion rates by displaying AI-generated promotion summaries.This deck follows a three-part structure: 1. Core Logic (How inputs drive summaries), 2. Output Validation (Reviewing generated text examples), and 3. User Experience (Frontend demo)."

    input_text: "We will leverage synergies to optimize the flywheel."
    critique: "Jargon overload. Low clarity. No distinct meaning."
    rewrite: "We will migrate the Promotion admin to CMT to siginificantly improve efficience."

    # ADD THIS NEW "BODY CONTENT" EXAMPLE:
    input_text: "Slide Title: Strong User Growth. Body: We saw significant uplift in daily active users across various regions due to better performance."
    critique: "Vague Body Content. The headline is fine, but the bullet point lacks evidence. 'Significant uplift' needs a % or absolute number. 'Various regions' is lazy—specify which ones."
    rewrite: "Body: DAU increased by 15% (20k users) in SEA and LATAM, driven by a 200ms reduction in app load time."
    
    ### JSON STRUCTURE:
    {{
        "reasoning_log": "<string: Write a 3-sentence internal analysis of the logic flaws here FIRST.>",
        "scores": {{
            "Logic": <int 0-100>,
            "Clarity": <int 0-100>,
            "Impact": <int 0-100>
        }},
        "executive_summary": "<string: Brutal one-sentence summary based on the reasoning_log>",
        "narrative_check": {{
             "original_headlines": [
                 "<string: Extracted Headline 1>",
                 "<string: Extracted Headline 2>"
             ],
             "critique": "<string: Critique of the current storytelling flow>",
             "revised_headlines": [
                 "<string: Improved Headline 1>",
                 "<string: Improved Headline 2>"
             ]
        }},
       "section_deep_dive": [
            {{
                "page_number": "<int: The page number extracted from the [PAGE X] marker>",
                "target_section": "<string: Quote the specific BULLET POINT or SENTENCE (not the headline)>",
                "issue": "<string: Specific critique of the evidence/data (e.g. 'Claim lacks metric', 'Vague adjective')>",
                "improved_version": "<string: Rewrite the bullet point to be data-driven and specific>",
                "why": "<string: Why this is better>"
            }}
        ]
    }}
    """

    # D. Generation
    with st.spinner("Processing Logic Matrix..."):
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
    # E. Rendering
    try:
        data = json.loads(response.text)
        
        with col2:
            st.markdown("### SCORECARD")
            s1, s2, s3 = st.columns(3)
            s1.metric("LOGIC", f"{data.get('scores', {}).get('Logic', 0)}")
            s2.metric("CLARITY", f"{data.get('scores', {}).get('Clarity', 0)}")
            s3.metric("IMPACT", f"{data.get('scores', {}).get('Impact', 0)}")
        
        st.divider()
        st.info(f"**EXECUTIVE SUMMARY:** {data.get('executive_summary', 'No summary generated.')}")
        
        # --- NEW TABS LAYOUT ---
        tab1, tab2 = st.tabs(["STORY FLOW", "🔬 DEEP DIVE & REWRITES"])
        
        # --- TAB 1: NARRATIVE FLOW ---
        with tab1:
            st.markdown("#### The Narrative Check (Pyramid Principle)")
            nav_data = data.get('narrative_check', {})
            
            st.markdown(f"> *{nav_data.get('critique', 'No critique available.')}*")
            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("🔴 ORIGINAL FLOW")
                for line in nav_data.get('original_headlines', []):
                    st.text(f"• {line}")
            
            with col_b:
                st.caption("🟢 OPTIMIZED FLOW")
                for line in nav_data.get('revised_headlines', []):
                    st.markdown(f"**• {line}**")

            if data.get('scores', {}).get('Logic', 0) < 75:
                st.error("⚠️ NARRATIVE THREAD BROKEN")
            else:
                st.success("✅ NARRATIVE THREAD STABLE")
        
        # --- TAB 2: SECTIONS DEEP DIVE (SPLIT VIEW) ---
        with tab2:
            st.markdown("#### 🔬 Surgical Reconstruction")
            st.caption("Specific text edits to improve Logic, Clarity, and Impact.")
            
            deep_dive_items = data.get('section_deep_dive', [])
            
            if not deep_dive_items:
                st.info("✅ No critical issues found. Your deck is clean!")
            
            for i, item in enumerate(deep_dive_items):
                with st.container():
                    # 1. HEADER WITH PAGE NUMBER
                    page_num = item.get('page_number', 'General')
                    target = item.get('target_section', 'General Logic')
                    st.markdown(f"##### 📄 Page {page_num}: {target[:50]}..." if len(target) > 50 else f"##### 📄 Page {page_num}: {target}")
                    
                    c1, c2 = st.columns([1, 2], gap="large")
                    
                    with c1:
                        # DIAGNOSIS (RED)
                        st.markdown('<div class="issue-tag">THE SYMPTOM (ISSUE)</div>', unsafe_allow_html=True)
                        st.markdown(f"**{item.get('issue', 'N/A')}**")
                        st.markdown(f"<div class='logic-footer'><b>💡 ROOT CAUSE:</b><br>{item.get('why', 'N/A')}</div>", unsafe_allow_html=True)
                    
                    with c2:
                        # PRESCRIPTION (GREEN)
                        st.markdown('<div class="fix-tag">THE PRESCRIPTION (REWRITE)</div>', unsafe_allow_html=True)
                        
                        rewrite_text = item.get('improved_version', 'N/A')
                        
                        # TRUNCATION LOGIC
                        # If text is short (< 300 chars), show it all.
                        # If text is long, show preview + Expander.
                        if len(rewrite_text) < 300:
                            st.info(rewrite_text, icon="✍️")
                        else:
                            st.info(rewrite_text[:300] + "...", icon="✍️")
                            with st.expander("Show Full Rewrite"):
                                st.code(rewrite_text, language="text") # Use code block inside expander for easy copying
                    
                    st.divider()

    except Exception as e:
        st.error(f"Data Stream Parsing Error: {e}")
        with st.expander("DEBUG DATA"):
            st.code(response.text)