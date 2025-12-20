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

# --- HELPER FUNCTIONS (LOGGING WITH SESSION ID) ---
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

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Deck Clinic: Data Lake Edition",
    page_icon="💾",
    layout="wide"
)

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5 { font-weight: 700; color: #202124; letter-spacing: -0.5px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    div.stButton > button { border-radius: 20px; border: 1px solid #dadce0; font-weight: 600; transition: all 0.2s; background-color: #ffffff; color: #3c4043; }
    div.stButton > button:hover { background-color: #f1f3f4; border-color: #dadce0; transform: translateY(-1px); color: #202124; }
    .issue-tag { background-color: #fce8e6; color: #c5221f; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.75rem; display: inline-block; margin-bottom: 8px; }
    .fix-tag { background-color: #e6f4ea; color: #137333; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.75rem; display: inline-block; margin-bottom: 8px; }
    .logic-footer { font-size: 0.85rem; color: #5f6368; background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin-top: 10px; border: 1px solid #f1f3f4; }
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

# Ensure "user_uploads" folder exists
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
    st.title("🎛️ SETTINGS")
    doc_type = st.selectbox("DIAGNOSTIC PROTOCOL", ["Strategy Deck (McKinsey/Amazon)", "Product Spec (Technical)", "Exec Update (Brief)"])
    st.divider()
    
    st.caption("📂 KNOWLEDGE BASE")
    uploaded_file = st.file_uploader("Upload 'Gold Standard' PDF", type="pdf")
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

    st.divider()
    
    # 🔐 MASTER ADMIN PANEL
    with st.expander("🔐 ADMIN PANEL (MASTER VIEW)"):
        admin_pass = st.text_input("Enter Admin Key", type="password")
        if admin_pass == "gemini2025": 
            st.success("ACCESS GRANTED")
            
            # --- THE DATA LAKE MERGE ---
            if os.path.exists("clinic_logs.csv") and os.path.exists("feedback_logs.csv"):
                df_system = pd.read_csv("clinic_logs.csv")
                df_feedback = pd.read_csv("feedback_logs.csv")
                
                # Clean headers just in case
                df_system.columns = df_system.columns.str.strip()
                df_feedback.columns = df_feedback.columns.str.strip()
                
                try:
                    # LEFT JOIN System Logs with Feedback on Session ID
                    df_master = pd.merge(df_system, df_feedback[['Session ID', 'Rating', 'Comment']], on='Session ID', how='left')
                    
                    st.markdown("### 🏆 Master Performance Table")
                    st.caption("Combined View: Input (File) + Output (Score) + Feedback (Rating)")
                    st.dataframe(df_master)
                    
                    # Highlight Conflicts
                    st.markdown("### 🚨 Conflict Detector")
                    conflicts = df_master[(df_master['Logic Score'] > 80) & (df_master['Rating'] == 'Negative')]
                    if not conflicts.empty:
                        st.error(f"Found {len(conflicts)} cases where AI was confident but User was unhappy!")
                        st.dataframe(conflicts)
                    else:
                        st.info("No obvious conflicts found.")
                        
                except Exception as e:
                    st.error(f"Merge Error: {e}. Check CSV headers.")
            
            elif os.path.exists("clinic_logs.csv"):
                st.dataframe(pd.read_csv("clinic_logs.csv"))
            
            if st.button("Clear ALL Data"):
                if os.path.exists("clinic_logs.csv"): os.remove("clinic_logs.csv")
                if os.path.exists("feedback_logs.csv"): os.remove("feedback_logs.csv")
                for f in os.listdir("user_uploads"):
                    file_path = os.path.join("user_uploads", f)
                    if os.path.isfile(file_path): os.remove(file_path)
                st.rerun()

        elif admin_pass:
            st.error("Access Denied")

# --- 6. MAIN INTERFACE ---
st.title(" 🎠DECK Playground")
st.caption(f"PROTOCOL: {doc_type} | CORE: gemini-flash-latest") 

col1, col2 = st.columns([2, 3]) 

with col1:
    st.markdown("### UPLOAD DRAFT DECK")
    target_pdf = st.file_uploader("Upload Draft PDF", type="pdf", key="target")
    analyze_btn = st.button("RUN DIAGNOSTIC", type="primary", use_container_width=True)

# 1. Reset Session if NEW file uploaded
if target_pdf and 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = target_pdf.name
    st.session_state.analysis_data = None
    # GENERATE NEW SESSION ID
    st.session_state.session_id = str(uuid.uuid4())[:8] 
elif target_pdf and st.session_state.get('last_uploaded') != target_pdf.name:
    st.session_state.last_uploaded = target_pdf.name
    st.session_state.analysis_data = None
    st.session_state.session_id = str(uuid.uuid4())[:8]

# 2. Main Logic Flow
if (target_pdf and analyze_btn) or (target_pdf and st.session_state.get('analysis_data')):
    
    # PHASE A: GENERATION (Only runs if data is missing!)
    if not st.session_state.get('analysis_data'):
        
        # A. File Processing & SAVING TO VAULT
        session_id = st.session_state.session_id
        safe_filename = f"{session_id}_{target_pdf.name}"
        save_path = os.path.join("user_uploads", safe_filename)
        
        with open(save_path, "wb") as f:
            f.write(target_pdf.getbuffer())
        
        loader = PyPDFLoader(save_path)
        draft_docs = loader.load()
        
        # Inject Page Markers
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

        # C. Prompt Construction
        base_instruction = ""
        if "Strategy" in doc_type:
            base_instruction = "ROLE: VP of Strategy. FRAMEWORK: Amazon Clarity, McKinsey Structure."
        elif "Product" in doc_type:
            base_instruction = "ROLE: Senior Technical PM. FRAMEWORK: Feasibility checks, Spec strictness."
        else:
            base_instruction = "ROLE: CEO. FRAMEWORK: BLUF, Extreme Brevity."

        # ✅ CHECKED: THIS CONTAINS THE FULL USER PROMPT
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
        critique: "1. Grammar: 'is enable' is broken. 2. Weak Metrics: 'encourage to buy' is vague; use 'conversion'. 3. Illogical Flow: The proposed agenda jumps from 'Input Logic' to 'Frontend Demo' before validating the output quality."
        rewrite: "Objective: Increase Shopee conversion rates by displaying AI-generated promotion summaries. This deck follows a three-part structure: 1. Core Logic (How inputs drive summaries), 2. Output Validation (Reviewing generated text examples), and 3. User Experience (Frontend demo)."
    
        input_text: "We will leverage synergies to optimize the flywheel."
        critique: "Jargon overload. Low clarity. No distinct meaning."
        rewrite: "We will migrate the Promotion admin to CMT to significantly improve efficiency."
    
        input_text: "Slide Title: Strong User Growth. Body: We saw significant uplift in daily active users across various regions due to better performance."
        critique: "Vague Body Content. The headline is fine, but the bullet point lacks evidence. 'Significant uplift' needs a % or absolute number."
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
                 "original_headlines": [ "<string: Extracted Headline 1>", "<string: Extracted Headline 2>" ],
                 "critique": "<string: Critique of the current storytelling flow>",
                 "revised_headlines": [ "<string: Improved Headline 1>", "<string: Improved Headline 2>" ]
            }},
           "section_deep_dive": [
                {{
                    "page_number": "<int: The page number extracted from the [PAGE X] marker>",
                    "target_section": "<string: Quote the specific BULLET POINT or SENTENCE (not the headline)>",
                    "issue": "<string: Specific critique of the evidence/data>",
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
            
            st.session_state.analysis_data = json.loads(response.text)
            
            # LOG WITH SESSION ID
            log_interaction(
                session_id=session_id,
                filename=safe_filename,
                doc_type=doc_type,
                scores=st.session_state.analysis_data.get('scores', {}),
                exec_summary=st.session_state.analysis_data.get('executive_summary', 'N/A')
            )

    # PHASE B: RENDERING
    data = st.session_state.analysis_data
    
    with col2:
        st.markdown(f"### SCORECARD (ID: `{st.session_state.session_id}`)")
        s1, s2, s3 = st.columns(3)
        # Safe getters
        s1.metric("LOGIC", f"{data.get('scores', {}).get('Logic', 0)}")
        s2.metric("CLARITY", f"{data.get('scores', {}).get('Clarity', 0)}")
        s3.metric("IMPACT", f"{data.get('scores', {}).get('Impact', 0)}")
    
    st.divider()
    
    # FEEDBACK LOOP
    with st.container():
        st.markdown("#### 💬 Was this helpful?")
        fb_col1, fb_col2 = st.columns([3, 1])
        with fb_col1:
            user_comment = st.text_input("Feedback (Optional)", placeholder="E.g. The logic check was too harsh...")
        with fb_col2:
            st.write("") 
            st.write("")
            b1, b2 = st.columns(2)
            if b1.button("👍"):
                log_feedback(st.session_state.session_id, "Positive", user_comment, doc_type)
                st.toast("Feedback Saved!", icon="👍")
            if b2.button("👎"):
                log_feedback(st.session_state.session_id, "Negative", user_comment, doc_type)
                st.toast("Feedback Saved!", icon="📉")

    st.divider()
    st.info(f"**EXECUTIVE SUMMARY:** {data.get('executive_summary', 'No summary generated.')}")
    
    # TABS RENDER
    tab1, tab2 = st.tabs(["STORY FLOW", "🔬 DEEP DIVE & REWRITES"])
    
    with tab1:
        st.markdown("#### The Narrative Check (Pyramid Principle)")
        nav_data = data.get('narrative_check', {})
        st.markdown(f"> *{nav_data.get('critique', 'No critique available.')}*")
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("🔴 ORIGINAL FLOW")
            for line in nav_data.get('original_headlines', []): st.text(f"• {line}")
        with col_b:
            st.caption("🟢 OPTIMIZED FLOW")
            for line in nav_data.get('revised_headlines', []): st.markdown(f"**• {line}**")
        
        if data.get('scores', {}).get('Logic', 0) < 75: st.error("⚠️ NARRATIVE THREAD BROKEN")
        else: st.success("✅ NARRATIVE THREAD STABLE")

    with tab2:
        st.markdown("#### 🔬 Surgical Reconstruction")
        st.caption("Specific text edits to improve Logic, Clarity, and Impact.")
        for i, item in enumerate(data.get('section_deep_dive', [])):
            with st.container():
                page_num = item.get('page_number', '?')
                target = item.get('target_section', 'General Logic')
                
                header_text = f"📄 Page {page_num}: {target}"
                if len(header_text) > 60: header_text = header_text[:60] + "..."
                st.markdown(f"##### {header_text}")
                
                c1, c2 = st.columns([1, 2], gap="large")
                with c1:
                    st.markdown('<div class="issue-tag">THE SYMPTOM (ISSUE)</div>', unsafe_allow_html=True)
                    st.markdown(f"**{item.get('issue', 'N/A')}**")
                    st.markdown(f"<div class='logic-footer'><b>💡 ROOT CAUSE:</b><br>{item.get('why', 'N/A')}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="fix-tag">THE PRESCRIPTION (REWRITE)</div>', unsafe_allow_html=True)
                    rewrite_text = item.get('improved_version', 'N/A')
                    if len(rewrite_text) < 300: st.info(rewrite_text, icon="✍️")
                    else:
                        st.info(rewrite_text[:300] + "...", icon="✍️")
                        with st.expander("Show Full Rewrite"): st.code(rewrite_text, language="text")
                st.divider()