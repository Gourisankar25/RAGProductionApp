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
    
    #Breaks the text into 1000-character chunks with 200-character overlap (helps maintain context between chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
    retriever = vectorstore.as_retriever()
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



#=== STEP 1: CONDENSE QUESTION CHAIN ===
# Converts follow-up questions into standalone questions using chat history
condense_template = """Given the chat history and a follow-up question, 
rephrase the follow-up question to be a standalone question that can be understood without the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

condense_prompt = ChatPromptTemplate.from_template(condense_template)

# Chain that takes question + history and outputs standalone question
condense_chain = (
    {
        "chat_history": lambda x: format_chat_history(get_recent_history(chat_history)),
        "question": RunnablePassthrough()
    }
    | condense_prompt
    | llm
    | StrOutputParser()
)



#=== STEP 2: MAIN RAG CHAIN ===
# Answers the standalone question using retrieved context
answer_template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)

def format_docs(docs):
    """Combines multiple retrieved chunks into a single text string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Chain that retrieves context and generates answer
answer_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | answer_prompt
    | llm
    | StrOutputParser()
)



#=== CONVERSATIONAL RAG SYSTEM ===
chat_history = []

print("=" * 60)
print("RAG Chatbot - Ask questions about your PDF")
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        print("🧠 Creating embeddings...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()
        
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
        # If this is a follow-up (chat history exists), condense the question first
        if chat_history:
            print("🔄 Processing follow-up question...")
            standalone_question = condense_chain.invoke(query)
            print(f"📝 Standalone: {standalone_question}")
        else:
            # First question - use as is
            standalone_question = query
        
        # Get answer using the standalone question
        print("🤖 Thinking...")
        result = answer_chain.invoke(standalone_question)
        
        print(f"\n🤖 AI: {result}")
        
        # Store in history (original question + answer)
        chat_history.append((query, result))
        
        # Show memory status
        print(f"\n💾 Memory: {len(chat_history)} exchanges stored (max: {MAX_HISTORY})")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try rephrasing your question or type 'help' for commands")
