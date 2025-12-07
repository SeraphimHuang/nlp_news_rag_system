import streamlit as st
import os
import sys
import time

# Import backend logic
# Ensure current directory is in path to import main.py
sys.path.append(os.getcwd())
from main import NewsRAGSystem, generate_answer_with_ollama, list_ollama_models, check_ollama_connection

# Page configuration
st.set_page_config(
    page_title="NewsRAG Assistant",
    page_icon="📰",
    layout="wide"
)

# 1. Resource Caching: Load RAG System
@st.cache_resource
def load_rag_engine():
    """Initialize and load RAG index"""
    checkpoint_dir = './newsrag_checkpoint'
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(os.path.join(checkpoint_dir, 'faiss.index')):
        return None
    
    rag = NewsRAGSystem()
    try:
        rag.load(checkpoint_dir)
        return rag
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None

# Initialize system
rag_system = load_rag_engine()

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Status Check
    if rag_system is None:
        st.error("❌ Index file not found")
        st.info("Please run main.py first to build the index: `python main.py --data ...`")
        st.stop()
    else:
        st.success(f"✅ Index loaded ({len(rag_system.documents)} documents)")

    # Ollama Connection Check
    # No parameters passed, main.py will handle env var check
    ollama_status = check_ollama_connection()
    
    if ollama_status:
        st.success("✅ Ollama Service Online")
        # Get available models
        available_models = list_ollama_models()
        if not available_models:
            available_models = ['mistral'] # Default fallback
        
        selected_model = st.selectbox(
            "Select Model (Ollama)",
            available_models,
            index=0 if available_models else None
        )
    else:
        st.error("❌ Ollama Service Not Running")
        st.info("Please run `ollama serve` in terminal")
        selected_model = "mistral" # Avoid undefined variable
        
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
        full_response = ""
        
        if not ollama_status:
            st.error("Cannot connect to Ollama service. Please check if it is running in the background.")
        else:
            with st.spinner('Retrieving news and thinking...'):
                try:
                    # Retrieve
                    retrieved_docs = rag_system.retrieve(prompt, k=k_retrieval)
                    
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
                    
                    # Display Sources (Using Expander)
                    if sources:
                        with st.expander("📚 Reference News Sources"):
                            for src in sources:
                                st.markdown(f"**[{src['rank']}] Relevance: N/A**") 
                                st.markdown(f"_{src['passage']}..._")
                                if src.get('url'):
                                    st.markdown(f"🔗 [Read Original]({src['url']})")
                                st.divider()
                                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
