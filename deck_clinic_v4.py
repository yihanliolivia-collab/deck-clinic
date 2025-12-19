import streamlit as st
import google.generativeai as genai
import os
import tempfile
import json
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# Updated import for new LangChain versions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. Basic Page Configuration ---
st.set_page_config(
    page_title="Deck Clinic V3: PM Edition",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for scorecard
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .big-font {
        font-size:20px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Security & API Connection ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 Critical Error: API Key not found. Please check Secrets settings.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# --- 3. Core Engine (Cached) ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 4. Sidebar: Knowledge Base ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.caption("Role: Head of Product Management")
    
    uploaded_file = st.file_uploader("Upload 'Gold Standard' PDF", type="pdf")
    
    if uploaded_file and st.button("Train Knowledge Base"):
        with st.spinner("Processing Reference Material..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            raw_docs = loader.load()
            
            # Splitting text for vector retrieval
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            docs = text_splitter.split_documents(raw_docs)
            
            vector_db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            # Persist call is automatic in newer Chroma versions, but keeping for safety
            try:
                vector_db.persist()
            except:
                pass
            st.success(f"✅ Learned from {len(docs)} chunks of knowledge!")

# --- 5. Main Interface: Deck Review ---
st.title("🏥 Deck Clinic: Strategic Review")
st.markdown("Your **AI Product Manager** will review your draft and generate a structured scorecard.")

col1, col2 = st.columns([2, 3]) 

with col1:
    st.subheader("📄 Input Draft")
    target_pdf = st.file_uploader("Upload Proposal Draft", type="pdf", key="target")
    analyze_btn = st.button("🚀 Run PM Review", type="primary", use_container_width=True)

if target_pdf and analyze_btn:
    # A. Prepare File
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(target_pdf.read())
        draft_path = tmp_file.name
    
    loader = PyPDFLoader(draft_path)
    draft_docs = loader.load()
    draft_text = " ".join([d.page_content for d in draft_docs])

    # B. Retrieve Context (RAG)
    with st.spinner("1/3 Retrieving Knowledge..."):
        try:
            vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            results = vector_db.similarity_search(draft_text, k=3)
            knowledge_context = "\n".join([doc.page_content for doc in results])
        except:
            knowledge_context = "General Top Tech Company Standards"

    # C. Core Prompt
    prompt = f"""
    You are the Head of Product Management in a Top Tech company reviewing a proposal draft.
    
    ### REFERENCE (Gold Standard Examples):
    {knowledge_context}
    
    ### DRAFT TO REVIEW:
    {draft_text[:50000]} 
    
    ### INSTRUCTIONS:
    1. Compare the Draft against the style, tone, and logic of the Reference.
    2. Output the result in **STRICT JSON FORMAT** only. Do not add markdown or intro text.
    
    ### JSON STRUCTURE (Fill this in):
    {{
        "scores": {{
            "Strategic_Fit": <int 0-100>,
            "Clarity": <int 0-100>,
            "Persuasion": <int 0-100>
        }},
        "executive_summary_feedback": "<string: Summary of the biggest gap>",
        "critical_issues": [
            {{
                "section": "<string: Identify Section Header>",
                "issue": "<string: What is wrong>",
                "fix": "<string: Actionable advice>"
            }},
            {{
                "section": "<string>", "issue": "<string>", "fix": "<string>"
            }}
        ],
        "rewrite_showcase": {{
            "original_text": "<string: Quote a weak paragraph>",
            "improved_version": "<string: Rewrite it in Top Tech PM style>",
            "why": "<string: Explain the reasoning>"
        }}
    }}
    """

    # D. Generate & Parse
    with st.spinner("2/3 Analyzing Logic Flow..."):
        # Corrected model name to ensure JSON mode works
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
    with st.spinner("3/3 Rendering Dashboard..."):
        try:
            data = json.loads(response.text)
            
            # --- Display Area (Right Column) ---
            with col2:
                st.subheader("📊 PM Scorecard")
                
                # 1. Metrics
                s1, s2, s3 = st.columns(3)
                s1.metric("Strategy", f"{data['scores']['Strategic_Fit']}/100")
                s2.metric("Clarity", f"{data['scores']['Clarity']}/100")
                s3.metric("Persuasion", f"{data['scores']['Persuasion']}/100")
                
                # 2. Bar Chart
                chart_df = pd.DataFrame({
                    "Metric": list(data['scores'].keys()),
                    "Score": list(data['scores'].values())
                })
                st.bar_chart(chart_df, x="Metric", y="Score", color="#FF4B4B")

            # --- Detailed Report (Bottom) ---
            st.divider()
            
            # Tab 1: Critical Issues
            st.subheader("🛑 Critical Gaps & Fixes")
            st.info(f"**PM Summary:** {data['executive_summary_feedback']}")
            
            for item in data['critical_issues']:
                with st.expander(f"📍 Issue in: {item['section']}"):
                    st.write(f"**Problem:** {item['issue']}")
                    st.success(f"**Action:** {item['fix']}")

            # Tab 2: Rewrite Showcase
            st.divider()
            st.subheader("✨ Before vs After Showcase")
            c_old, c_new = st.columns(2)
            with c_old:
                st.warning("🔴 Original Draft")
                st.write(data['rewrite_showcase']['original_text'])
            with c_new:
                st.success("🟢 PM Optimized")
                st.write(data['rewrite_showcase']['improved_version'])
            st.caption(f"💡 Logic: {data['rewrite_showcase']['why']}")

        except json.JSONDecodeError:
            st.error("🚨 JSON Parsing Error. The AI response was not valid JSON.")
            with st.expander("Raw Response Debug"):
                st.text(response.text)