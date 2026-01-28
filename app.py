import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Research Paper Analysis Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# Initialize LLM and embeddings
@st.cache_resource
def initialize_models():
    """Initialize LLM and embeddings (cached)"""
    llm = OllamaLLM(model="llama3.2", temperature=0)
    embeddings = OllamaEmbeddings(model="llama3.2")
    return llm, embeddings

def format_chat_history(history):
    """Formats chat history into a readable string for the LLM"""
    if not history:
        return "No previous conversation"
    return "\n".join([f"Human: {q}\nAI: {a}" for q, a in history])

def get_recent_history(history, max_pairs=5):
    """Returns only the most recent conversation pairs"""
    return history[-max_pairs:] if len(history) > max_pairs else history

def process_pdf(uploaded_file):
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        with st.spinner("📥 Loading PDF..."):
            pdf_reader = PyPDFLoader(tmp_file_path)
            documents = pdf_reader.load()
        
        if not documents:
            st.error("No content found in PDF")
            return False
        
        st.success(f"✅ Loaded {len(documents)} pages")
        
        # Split text into chunks
        with st.spinner("✂️ Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            st.session_state.chunks = chunks
        
        st.success(f"✅ Created {len(chunks)} chunks")
        
        # Create embeddings and vector store
        with st.spinner("🧠 Creating embeddings (this may take a moment)..."):
            _, embeddings = initialize_models()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Create retriever with MMR
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20}
            )
            
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
        
        st.success("✅ Vector store created successfully!")
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False

def format_docs(docs):
    """Combines multiple retrieved chunks into a single text string"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_context_with_hybrid_search(question):
    """Retrieve context with smart strategy for research papers"""
    structure_keywords = ['abstract', 'introduction', 'title', 'authors', 'keywords']
    is_structure_query = any(keyword in question.lower() for keyword in structure_keywords)
    
    if is_structure_query:
        # For structure queries: get first 3 chunks + semantic search
        all_chunks = st.session_state.vectorstore.similarity_search(question, k=6)
        first_chunks = st.session_state.chunks[:3]
        combined = first_chunks + [c for c in all_chunks if c not in first_chunks]
        docs = combined[:8]
    else:
        # Normal semantic search
        docs = st.session_state.retriever.invoke(question)
    
    return format_docs(docs)

def get_answer(question):
    """Get answer from the RAG system"""
    llm, _ = initialize_models()
    
    # Prompt template
    answer_template = """You are a helpful research assistant analyzing academic research papers. 
Provide accurate, detailed answers based strictly on the information in the research paper context below.

Research Paper Context:
{context}

Previous Conversation (for reference only):
{chat_history}

Current Question: {question}

Instructions:
- Answer based ONLY on the research paper context provided
- Be precise and cite specific findings, methodologies, or results when available
- If the answer isn't in the context, say "This information is not available in the provided research paper"
- For follow-up questions, use previous conversation for context but maintain accuracy
- Use academic tone appropriate for research paper discussion

Answer:"""
    
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    
    # Create chain
    context = get_context_with_hybrid_search(question)
    chat_history = format_chat_history(get_recent_history(st.session_state.chat_history))
    
    # Generate answer
    chain = answer_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })
    
    return result

# Main UI
st.title("📄 Research Paper Analysis Bot")
st.markdown("Ask questions about your research paper using AI-powered analysis")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Research Paper")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a research paper in PDF format"
    )
    
    if uploaded_file is not None:
        if not st.session_state.pdf_processed or st.button("🔄 Process New PDF"):
            if process_pdf(uploaded_file):
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []
                st.rerun()
    
    st.divider()
    
    # Stats
    if st.session_state.pdf_processed:
        st.subheader("📊 Document Stats")
        st.metric("Chunks Created", len(st.session_state.chunks) if st.session_state.chunks else 0)
        st.metric("Conversations", len(st.session_state.chat_history))
    
    st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Info
    st.subheader("ℹ️ About")
    st.markdown("""
    This tool uses:
    - **Ollama (Llama 3.2)** for local LLM
    - **FAISS** for vector search
    - **LangChain** for RAG pipeline
    - **Streamlit** for UI
    """)

# Main chat area
if not st.session_state.pdf_processed:
    st.info("👈 Please upload a research paper PDF to get started")
else:
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br/>
            {question}
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 AI:</strong><br/>
            {answer}
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    st.divider()
    
    # Use columns for better layout
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question about the research paper:",
            key="user_input",
            placeholder="e.g., What is the abstract of this paper?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("Send 🚀", use_container_width=True)
    
    # Process question
    if ask_button and user_question:
        with st.spinner("🤖 Analyzing research paper..."):
            try:
                answer = get_answer(user_question)
                st.session_state.chat_history.append((user_question, answer))
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure Ollama is running: `ollama serve`")
    
    # Quick question buttons
    if len(st.session_state.chat_history) == 0:
        st.markdown("### 💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 What is the abstract?"):
                st.session_state.temp_question = "What is the abstract of this paper?"
                st.rerun()
        
        with col2:
            if st.button("🔬 What is the methodology?"):
                st.session_state.temp_question = "What methodology was used in this research?"
                st.rerun()
        
        with col3:
            if st.button("📊 What are the key findings?"):
                st.session_state.temp_question = "What are the main findings and results?"
                st.rerun()
        
        # Handle quick question
        if hasattr(st.session_state, 'temp_question'):
            with st.spinner("🤖 Analyzing research paper..."):
                try:
                    answer = get_answer(st.session_state.temp_question)
                    st.session_state.chat_history.append((st.session_state.temp_question, answer))
                    delattr(st.session_state, 'temp_question')
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Built with LangChain, Ollama, and Streamlit | Keep your research papers private with local processing</small>
</div>
""", unsafe_allow_html=True)
