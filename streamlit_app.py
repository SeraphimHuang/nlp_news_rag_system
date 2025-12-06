import streamlit as st
import os
import sys
import time

# 导入后端逻辑
# 确保当前目录在 path 中以便导入 main.py
sys.path.append(os.getcwd())
from main import NewsRAGSystem, generate_answer_with_ollama, list_ollama_models, check_ollama_connection

# 页面配置
st.set_page_config(
    page_title="NewsRAG Assistant",
    page_icon="📰",
    layout="wide"
)

# 1. 资源缓存：加载 RAG 系统
@st.cache_resource
def load_rag_engine():
    """初始化并加载 RAG 索引"""
    checkpoint_dir = './newsrag_checkpoint'
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(os.path.join(checkpoint_dir, 'faiss.index')):
        return None
    
    rag = NewsRAGSystem()
    try:
        rag.load(checkpoint_dir)
        return rag
    except Exception as e:
        st.error(f"加载索引失败: {e}")
        return None

# 初始化系统
rag_system = load_rag_engine()

# 2. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # 状态检查
    if rag_system is None:
        st.error("❌ 未找到索引文件")
        st.info("请先运行 main.py 构建索引: `python main.py --data ...`")
        st.stop()
    else:
        st.success(f"✅ 索引已加载 ({len(rag_system.documents)} 篇文档)")

    # Ollama 连接检查
    ollama_status = check_ollama_connection()
    if ollama_status:
        st.success("✅ Ollama 服务在线")
        # 获取可用模型
        available_models = list_ollama_models()
        if not available_models:
            available_models = ['mistral'] # 默认回退
        
        selected_model = st.selectbox(
            "选择模型 (Ollama)",
            available_models,
            index=0 if available_models else None
        )
    else:
        st.error("❌ Ollama 服务未运行")
        st.info("请在终端运行 `ollama serve`")
        selected_model = "mistral" # 避免变量未定义
        
    # 参数调整
    k_retrieval = st.slider("检索文章数量 (K)", min_value=1, max_value=10, value=3)
    
    st.divider()
    st.markdown("### 关于")
    st.markdown("这是一个基于本地 Ollama 和 FAISS 的新闻问答系统。")

# 3. 主界面聊天逻辑
st.title("📰 NewsRAG 智能问答")
st.caption("基于本地知识库的新闻助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的新闻助手。有什么我可以帮你的吗？"}]

# 显示历史消息
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # 如果是助手消息且包含来源信息，可以尝试渲染（这里简单处理，最新回复单独渲染来源）

# 处理用户输入
if prompt := st.chat_input("输入关于新闻的问题..."):
    # 1. 显示用户消息
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if not ollama_status:
            st.error("无法连接到 Ollama 服务，请检查后台是否运行。")
        else:
            with st.spinner('正在检索新闻并思考...'):
                try:
                    # 检索
                    retrieved_docs = rag_system.retrieve(prompt, k=k_retrieval)
                    
                    # 生成
                    result = generate_answer_with_ollama(
                        prompt, 
                        retrieved_docs, 
                        model=selected_model
                    )
                    
                    answer = result['answer']
                    sources = result['sources']
                    
                    # 显示答案
                    message_placeholder.markdown(answer)
                    
                    # 记录到历史
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    
                    # 展示来源 (使用 Expander)
                    if sources:
                        with st.expander("📚 参考新闻来源"):
                            for src in sources:
                                st.markdown(f"**[{src['rank']}] 相似度: N/A**") # 原始接口没返回 score 到 sources list，这里简化
                                st.markdown(f"_{src['passage']}..._")
                                if src.get('url'):
                                    st.markdown(f"🔗 [阅读原文]({src['url']})")
                                st.divider()
                                
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")

