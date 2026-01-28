from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#import os

#load_dotenv()

#Initialize LLM
llm = OllamaLLM(model="llama3.2", temperature=0) #deterministic responses  


#Loads your PDF from given path
pdf_path = r"C:\Users\gourig\Downloads\IJSDR2303219.pdf"

try:
    print("📥 Loading PDF...")
    pdf_reader = PyPDFLoader(pdf_path)
    #Extracts all text from the PDF
    documents = pdf_reader.load()
    
    if not documents:
        raise ValueError("No content found in PDF")
    
    print(f"✅ Loaded {len(documents)} pages")
    
    # Debug: Check if pages have content
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"📄 Total characters: {total_chars}")
    
    if total_chars == 0:
        raise ValueError("PDF has no text content (might be image-based)")
    
    #Breaks the text into 500-character chunks with 100-character overlap (smaller chunks for better precision)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        raise ValueError("No chunks created from PDF - try a different PDF with text content")
    
    print(f"✂️ Created {len(chunks)} chunks")

except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    print(f"💡 Try using a different PDF or check if '{pdf_path}' exists")
    exit(1)



#Converts text chunks into numerical vectors using OllamaEmbeddings
print("🧠 Creating embeddings...")
embeddings = OllamaEmbeddings(model="llama3.2")

try:
    #Creates a FAISS vector store from the document chunks and their embeddings
    vectorstore = FAISS.from_documents(chunks, embeddings)
    #Sets up a retriever to fetch relevant document chunks based on user queries
    # Use MMR (Maximal Marginal Relevance) for diverse chunks including early pages
    # fetch_k=20 considers more candidates, k=6 returns top 6 diverse chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )
    print("✅ Vector store created successfully!")
except Exception as e:
    print(f"❌ Error creating vector store: {e}")
    print("💡 Make sure Ollama is running: ollama serve")
    exit(1)



#=== MEMORY MANAGEMENT ===
MAX_HISTORY = 5  # Keep only last 5 Q&A pairs to limit memory usage

def format_chat_history(history):
    """Formats chat history into a readable string for the LLM"""
    if not history:
        return "No previous conversation"
    return "\n".join([f"Human: {q}\nAI: {a}" for q, a in history])

def get_recent_history(history, max_pairs=MAX_HISTORY):
    """Returns only the most recent conversation pairs to prevent token overflow"""
    return history[-max_pairs:] if len(history) > max_pairs else history



#=== RAG CHAIN FOR RESEARCH PAPERS ===
# Research-paper focused prompt that maintains accuracy without rephrasing user questions
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

def format_docs(docs):
    """Combines multiple retrieved chunks into a single text string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Chain that retrieves context and generates answer with chat history
def get_context_with_debug(question):
    """Retrieve context with smart strategy for research papers"""
    # Keywords that indicate user wants early paper sections
    structure_keywords = ['abstract', 'introduction', 'title', 'authors', 'keywords']
    
    # Check if query is about paper structure
    is_structure_query = any(keyword in question.lower() for keyword in structure_keywords)
    
    if is_structure_query:
        # For structure queries: get first 3 chunks + semantic search
        all_chunks = vectorstore.similarity_search(question, k=6)
        # Also get first chunks from the document
        first_chunks = chunks[:3]
        # Combine and deduplicate
        combined = first_chunks + [c for c in all_chunks if c not in first_chunks]
        docs = combined[:8]  # Take top 8 total
        print(f"\n🔍 Retrieved {len(docs)} chunks (including early document sections):")
    else:
        # Normal semantic search
        docs = retriever.invoke(question)
        print(f"\n🔍 Retrieved {len(docs)} chunks:")
    
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"   Chunk {i}: {preview}...")
    return format_docs(docs)

answer_chain = (
    {
        "context": lambda x: get_context_with_debug(x),
        "chat_history": lambda x: format_chat_history(get_recent_history(chat_history)),
        "question": lambda x: x
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)



#=== CONVERSATIONAL RAG SYSTEM ===
chat_history = []

print("=" * 60)
print("Research Paper Analysis Bot - Ask questions about your research paper")
print("=" * 60)
print("\n💡 Commands:")
print("  - Type your question to get an answer")
print("  - 'clear' or 'reset' - Clear conversation history")
print("  - 'history' - Show conversation history")
print("  - 'pdf' - Change PDF file")
print("  - 'help' - Show commands")
print("  - 'exit', 'quit', 'bye' - End conversation")
print("=" * 60)

def load_new_pdf():
    """Dynamically load a new PDF file"""
    global vectorstore, retriever, chunks
    
    pdf_path = input("\n📄 Enter PDF path: ").strip().strip('"')
    
    try:
        print("📥 Loading PDF...")
        pdf_reader = PyPDFLoader(pdf_path)
        documents = pdf_reader.load()
        
        print("✂️ Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        print("🧠 Creating embeddings...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20}
        )
        
        print("✅ PDF loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return False

while True:
    query = input("\n🧑 You: ").strip()
    
    # Handle empty input
    if not query:
        continue
    
    # Handle commands
    if query.lower() in ['exit', 'quit', 'bye']:
        print("\n👋 Goodbye!")
        break
    
    elif query.lower() in ['clear', 'reset']:
        chat_history = []
        print("🗑️ Chat history cleared!")
        continue
    
    elif query.lower() == 'history':
        if not chat_history:
            print("📭 No conversation history yet")
        else:
            print("\n📜 Conversation History:")
            print("-" * 60)
            for i, (q, a) in enumerate(chat_history, 1):
                print(f"\n{i}. Q: {q}")
                print(f"   A: {a[:150]}{'...' if len(a) > 150 else ''}")
            print("-" * 60)
        continue
    
    elif query.lower() == 'pdf':
        if load_new_pdf():
            chat_history = []  # Clear history when new PDF is loaded
        continue
    
    elif query.lower() == 'help':
        print("\n💡 Available Commands:")
        print("  • Type your question to chat with the PDF")
        print("  • 'clear'/'reset' - Start a fresh conversation")
        print("  • 'history' - View all previous Q&A")
        print("  • 'pdf' - Load a different PDF file")
        print("  • 'help' - Show this help message")
        print("  • 'exit'/'quit'/'bye' - Close the application")
        continue
    
    # Process actual questions
    try:
        # Use original question without rephrasing to maintain accuracy
        print("🤖 Analyzing research paper...")
        result = answer_chain.invoke(query)
        
        print(f"\n🤖 AI: {result}")
        
        # Store in history (original question + answer)
        chat_history.append((query, result))
        
        # Show memory status
        print(f"\n💾 Memory: {len(chat_history)} exchanges stored (max: {MAX_HISTORY})")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try rephrasing your question or type 'help' for commands")
