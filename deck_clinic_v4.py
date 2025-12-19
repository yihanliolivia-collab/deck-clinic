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

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="Deck Cgoogle.api_core.exceptions.NotFound: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/deck-clinic/deck_clinic_v4.py", line 155, in <module>
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
File "/home/adminuser/venv/lib/python3.13/site-packages/google/generativeai/generative_models.py", line 331, in generate_content
    response = self._client.generate_content(
        request,
        **request_options,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/google/ai/generativelanguage_v1beta/services/generative_service/client.py", line 835, in generate_content
    response = rpc(
        request,
    ...<2 lines>...
        metadata=metadata,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/google/api_core/gapic_v1/method.py", line 131, in __call__
    return wrapped_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
        target,
    ...<3 lines>...
        on_error=on_error,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
        exc,
    ...<6 lines>...
        timeout,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
File "/home/adminuser/venv/lib/python3.13/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
File "/home/adminuser/venv/lib/python3.13/site-packages/google/api_core/timeout.py", line 130, in func_with_timeout
    return func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/google/api_core/grpc_helpers.py", line 77, in error_remapped_callable
    raise exceptions.from_grpc_error(exc) from exclinic V3: PM Edition", 
    page_icon="🧠", 
    layout="wide"
)

# 自定义 CSS：让评分卡看起来更专业
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

# --- 2. 安全连接 Gemini ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 严重错误：未找到 API Key，请检查 Secrets 设置。")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# --- 3. 核心引擎 (带缓存) ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 4. 侧边栏：知识库 (Reference) ---
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
            
            # 这里的切片是为了向量检索，不是给 LLM 读的
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            docs = text_splitter.split_documents(raw_docs)
            
            vector_db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            vector_db.persist()
            st.success(f"✅ Learned from {len(docs)} chunks of knowledge!")

# --- 5. 主界面：提案诊断 ---
st.title("🏥 Deck Clinic: Strategic Review")
st.markdown("Your **AI Product Manager** will review your draft and generate a structured scorecard.")

col1, col2 = st.columns([2, 3]) # 左边传文件，右边看图表

with col1:
    st.subheader("📄 Input Draft")
    target_pdf = st.file_uploader("Upload Proposal Draft", type="pdf", key="target")
    analyze_btn = st.button("🚀 Run PM Review", type="primary", use_container_width=True)

if target_pdf and analyze_btn:
    # A. 准备文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(target_pdf.read())
        draft_path = tmp_file.name
    
    loader = PyPDFLoader(draft_path)
    draft_docs = loader.load()
    # 拼接全文
    draft_text = " ".join([d.page_content for d in draft_docs])

    # B. 检索参考资料 (RAG)
    with st.spinner("1/3 Retrieving Knowledge..."):
        try:
            vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            results = vector_db.similarity_search(draft_text, k=3)
            knowledge_context = "\n".join([doc.page_content for doc in results])
        except:
            knowledge_context = "General Top Tech Company Standards"

    # C. 核心 Prompt (结合了你的 Head of PM 人设 + JSON 强制格式)
    prompt = f"""
    You are the Head of Product Management in a Top Tech company reviewing a proposal draft.
    
    ### REFERENCE (Gold Standard Examples):
    {knowledge_context}
    
    ### DRAFT TO REVIEW:
    {draft_text[:500000]} 
    
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

    # D. 生成与解析
    with st.spinner("2/3 Analyzing Logic Flow..."):
        model = genai.GenerativeModel('gemini-flash-latest')
        # 强制请求 JSON (Gemini 1.5 特性)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
    with st.spinner("3/3 Rendering Dashboard..."):
        try:
            # 解析 JSON
            data = json.loads(response.text)
            
            # --- 展示区域 (Right Column) ---
            with col2:
                st.subheader("📊 PM Scorecard")
                
                # 1. 数字指标
                s1, s2, s3 = st.columns(3)
                s1.metric("Strategy", f"{data['scores']['Strategic_Fit']}/100")
                s2.metric("Clarity", f"{data['scores']['Clarity']}/100")
                s3.metric("Persuasion", f"{data['scores']['Persuasion']}/100")
                
                # 2. 柱状图 (Visual)
                chart_df = pd.DataFrame({
                    "Metric": list(data['scores'].keys()),
                    "Score": list(data['scores'].values())
                })
                st.bar_chart(chart_df, x="Metric", y="Score", color="#FF4B4B")

            # --- 详细报告区 (Bottom) ---
            st.divider()
            
            # Tab 1: 关键问题
            st.subheader("🛑 Critical Gaps & Fixes")
            st.info(f"**PM Summary:** {data['executive_summary_feedback']}")
            
            for item in data['critical_issues']:
                with st.expander(f"📍 Issue in: {item['section']}"):
                    st.write(f"**Problem:** {item['issue']}")
                    st.success(f"**Action:** {item['fix']}")

            # Tab 2: 重写示范
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