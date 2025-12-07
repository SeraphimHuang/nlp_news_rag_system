import streamlit as st
import os
import sys
import time

# Import backend logic
sys.path.append(os.getcwd())
from main import NewsRAGSystem, generate_answer_with_ollama, list_ollama_models, check_ollama_connection, load_and_preprocess_data

# Page configuration
st.set_page_config(
    page_title="NewsRAG Assistant",
    page_icon="📰",
    layout="wide"
)

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Strategy Selection (Just a UI toggle now, backend has all indices)
    retrieval_strategy = st.radio(
        "Retrieval Strategy",
        ["Simple (Single Index)", "Weighted (Dual Index)"],
        index=0,
        help="Simple: Standard RAG.\nWeighted: Separates headline/content for better precision."
    )
    
    strategy_code = 'weighted' if 'Weighted' in retrieval_strategy else 'simple'
    
    # Re-ranking Toggle
    use_reranker = st.checkbox(
        "Enable Cross-Encoder Re-ranking",
        value=True,
        help="Re-ranks top candidates for higher precision. Uncheck to rely solely on vector similarity."
    )

# 1. Resource Caching: Load RAG System
@st.cache_resource
def load_rag_engine():
    """Initialize and load RAG index (loads ALL indices at once)"""
    checkpoint_dir = './newsrag_checkpoint'
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_dir) and os.path.exists(os.path.join(checkpoint_dir, 'metadata.pkl')):
        try:
            rag = NewsRAGSystem()
            rag.load(checkpoint_dir)
            return rag
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            return None
    
    return None

# Initialize system
rag_system = load_rag_engine()

# Sidebar Status Check
with st.sidebar:
    # Status Check
    if rag_system is None:
        st.error("❌ Index Not Found")
        st.info("Please run `python main.py --save-index ./newsrag_checkpoint` to build the knowledge base.")
        st.stop()
    else:
        # Show which indices are active
        idx_count = 0
        if rag_system.faiss_index: idx_count += 1
        if rag_system.faiss_index_headline: idx_count += 1
        
        st.success(f"✅ System Ready ({len(rag_system.documents)} docs)")
        if strategy_code == 'weighted' and not rag_system.faiss_index_headline:
            st.warning("⚠️ Weighted index missing! Re-run build.")

    # Ollama Connection Check
    ollama_status = check_ollama_connection()
    
    if ollama_status:
        st.success("✅ Ollama Service Online")
        available_models = list_ollama_models()
        if not available_models:
            available_models = ['mistral'] 
        
        selected_model = st.selectbox(
            "Select Model (Ollama)",
            available_models,
            index=0 if available_models else None
        )
    else:
        st.error("❌ Ollama Service Not Running")
        st.info("Please run `ollama serve` in terminal")
        selected_model = "mistral"
        
    # Parameter Adjustment
    k_retrieval = st.slider("Number of Articles (K)", min_value=1, max_value=10, value=3)
    
    st.divider()
    st.markdown("### About")
    st.markdown("This is a News QA System based on local Ollama and FAISS.")

# 3. Main Chat Interface Logic
st.title("📰 NewsRAG Intelligent QA")
st.caption("News Assistant based on Local Knowledge Base")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am your news assistant. How can I help you?"}]

# Display History Messages
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle User Input
if prompt := st.chat_input("Enter your question about news..."):
    # 1. Display User Message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not ollama_status:
            st.error("Cannot connect to Ollama service.")
        else:
            with st.spinner(f'Retrieving news ({strategy_code} strategy) and thinking...'):
                try:
                    # Retrieve (Pass the selected strategy here!)
                    retrieved_docs = rag_system.retrieve(
                        prompt, 
                        strategy=strategy_code, 
                        use_reranker=use_reranker,
                        k=k_retrieval
                    )
                    
                    # Generate
                    result = generate_answer_with_ollama(
                        prompt, 
                        retrieved_docs, 
                        model=selected_model
                    )
                    
                    answer = result['answer']
                    sources = result['sources']
                    
                    # Display Answer
                    message_placeholder.markdown(answer)
                    
                    # Record to History
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    
                    # Display Sources
                    if sources:
                        with st.expander("📚 Reference News Sources"):
                            for src in sources:
                                date_str = f" - {src.get('date', 'N/A')}" if src.get('date') else ""
                                st.markdown(f"**[{src['rank']}] Source{date_str}**") 
                                st.markdown(f"_{src['passage']}..._")
                                if src.get('url'):
                                    st.markdown(f"🔗 [Read Original]({src['url']})")
                                st.divider()
                                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
