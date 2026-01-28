# 📄 Research Paper Analysis Bot - RAG Application

An intelligent chatbot that analyzes academic research papers using Retrieval-Augmented Generation (RAG). Built with LangChain, Ollama, and Streamlit for local, privacy-focused document analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

This application enables users to upload research papers in PDF format and interact with them through natural language questions. The bot provides accurate, context-aware answers based solely on the paper's content, making it ideal for researchers, students, and academics.

## ✨ Key Features

- 🔒 **100% Local Processing** - All data stays on your machine (Ollama-powered)
- 📚 **Smart Document Chunking** - Optimized text splitting for research papers
- 🎯 **Hybrid Retrieval Strategy** - Combines semantic search with structural understanding
- 💬 **Conversational Memory** - Maintains context across follow-up questions
- 🎨 **Modern UI** - Clean, intuitive Streamlit interface with dark theme
- 🔍 **Research-Focused Prompts** - Tailored for academic paper analysis
- ⚡ **Fast Vector Search** - FAISS-powered similarity matching
- 🎯 **Quick Question Buttons** - Pre-made queries for common research questions

## 🏗️ Architecture

```
User Query → Hybrid Retrieval → Vector Store (FAISS) → Context + History → LLM (Llama 3.2) → Answer
```

**Key Components:**
- **LangChain**: RAG pipeline orchestration
- **Ollama**: Local LLM inference (Llama 3.2)
- **FAISS**: Vector similarity search
- **Streamlit**: Web UI framework

## 📸 Screenshots

### Main Interface
![Main Interface](screenshots/pic_1.png)
*Clean, modern interface with sidebar navigation and chat area*

### PDF Upload & Processing
![PDF Processing](screenshots/pic_2.png)
*Upload and process research papers with real-time status updates - showing pages loaded, chunks created, and embedding generation*

### Ready-to-Use Interface
![Ready Interface](screenshots/pic_3.png)
*Interface after PDF processing is complete, showing:*
- Document uploaded and processed (IJSDR2303219.pdf - 244.2KB)
- 26 chunks created from the research paper
- Quick question buttons ("What is the abstract?", "What is the methodology?", "What are the key findings?")
- Document statistics sidebar showing 0 conversations (fresh start)
- Text input ready for custom questions

### Question & Answer Interaction
![Q&A Interaction](screenshots/pic_4.png)
*Complete Q&A interaction showing:*
- **User Question**: "What is the drowsy detection device mentioned in the paper?"
- **AI Response**: Detailed answer explaining the drowsiness detection system that combines face detection and eye detection to detect driver drowsiness
- **System Details**: Uses Raspberry Pi3 programmed in Python
- **Process Steps**: 
  1. Recording video
  2. Face detection
  3. Eye detection
  4. Drowsiness detection (combination of steps 2 and 3)
- **Document Stats Updated**: Shows 1 conversation completed
- Chat history preserved with clear formatting (blue for user, white for AI)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Gourisankar25/RAG-App.git
cd RAG-App
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install and start Ollama**
```bash
# Install Ollama from https://ollama.ai/

# Pull the Llama 3.2 model
ollama pull llama3.2

# Start Ollama (in a separate terminal)
ollama serve
```

### Running the Application

#### Streamlit UI (Recommended)
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

#### CLI Version
```bash
python main.py
```

## 📖 Usage

1. **Upload PDF**: Click "Browse files" or drag & drop in the sidebar
2. **Wait for Processing**: The app will:
   - Load PDF pages
   - Split into chunks (500 characters each)
   - Create embeddings using Ollama
   - Build FAISS vector store
3. **Ask Questions**: 
   - Use quick question buttons for common queries
   - Type custom questions in the text input
   - Click "Send 🚀" to submit
4. **Get Answers**: Receive accurate, context-based responses with:
   - Academic tone appropriate for research papers
   - Specific findings and details from the paper
   - Structured responses (lists, steps, explanations)
   - Clear indication when information isn't available

### Example Questions

- "What is the abstract of this paper?"
- "What methodology was used in this research?"
- "What are the main findings and results?"
- "Who are the authors of this paper?"
- "What are the key contributions?"
- "What are the limitations mentioned?"
- "What future work is suggested?"
- "What datasets were used?"
- "What is the drowsy detection device mentioned in the paper?"

## 🛠️ Technical Details

### Retrieval Strategy

**Hybrid Approach:**
- **Structural Queries** (abstract, introduction, authors, keywords, title): 
  - Retrieves first 3 chunks (document start)
  - Combines with semantic search results
  - Returns up to 8 diverse chunks
  - Ensures critical paper metadata is always available
- **Content Queries**: 
  - MMR (Maximal Marginal Relevance) for diverse, non-redundant results
  - Fetches 6 most relevant chunks
  - Fetch_k=20 candidates for better diversity

### Prompt Engineering

Research-focused prompt that:
- Identifies as a research assistant analyzing academic papers
- Enforces strict context-only responses
- Maintains formal academic tone
- Handles follow-up questions with conversation history (last 5 Q&A pairs)
- Explicitly admits when information isn't available
- Requests citations and specific findings from the paper
- Structures answers clearly (lists, steps, explanations)

### Configuration

Key parameters in code:
```python
chunk_size = 500          # Characters per chunk
chunk_overlap = 100       # Overlap between chunks (20%)
k = 6                     # Number of chunks to retrieve
fetch_k = 20              # Candidates for MMR algorithm
temperature = 0           # Deterministic responses (no randomness)
max_history = 5           # Conversation pairs to remember
```

## 📁 Project Structure

```
RAG-App/
├── app.py                      # Streamlit UI application
├── main.py                     # CLI version
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── screenshots/                # Screenshots folder
    ├── pic_1.png              # Main interface
    ├── pic_2.png              # PDF processing
    ├── pic_3.png              # Ready interface with quick questions
    └── pic_4.png              # Q&A interaction with detailed answer
```

## 🧪 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming language |
| **LangChain** | RAG framework & orchestration |
| **Ollama** | Local LLM inference |
| **Llama 3.2** | Language model (3B parameters) |
| **FAISS** | Vector similarity search |
| **Streamlit** | Web UI framework |
| **PyPDF** | PDF text extraction |

## 🔧 Configuration Options

Modify these in the code for different behavior:

- **Change LLM Model**: Update `model="llama3.2"` in `OllamaLLM()` and `OllamaEmbeddings()`
  - Try: `llama3.1`, `mistral`, `phi3`, `gemma2`, etc.
- **Adjust Chunk Size**: Modify `chunk_size` in `RecursiveCharacterTextSplitter`
  - Smaller chunks (300-400): More precise but may miss context
  - Larger chunks (800-1000): More context but less precise
- **Change Retrieval Count**: Update `k` value in retriever configuration
  - Higher k (8-10): More context but slower
  - Lower k (3-4): Faster but may miss relevant info
- **Modify Temperature**: Change `temperature` parameter
  - 0: Deterministic (same answer every time) - Current setting
  - 0.3-0.5: Slightly varied but still focused
  - 0.7-1.0: More creative/varied responses

## 🚨 Troubleshooting

**Ollama not responding?**
```bash
# Start Ollama service
ollama serve

# In another terminal, verify model is available
ollama list

# If model not found, pull it
ollama pull llama3.2
```

**Out of memory errors?**
- Use a smaller model: `ollama pull llama3.2:1b`
- Reduce chunk count: Set `k=3` instead of `k=6`
- Close other applications to free up RAM

**PDF not processing?**
- Ensure PDF is not password-protected
- Check file size (limit: 200MB)
- Verify PDF contains extractable text (not just scanned images)
- Try re-uploading the file

**Slow response times?**
- First query is always slower (model loading)
- Reduce `fetch_k` from 20 to 10
- Use a smaller/faster model like `phi3`

## 🎯 Dual Interface

This project includes **two interfaces** for different use cases:

### Streamlit UI ([`app.py`](app.py))
- **For**: End users, demos, presentations
- **Features**: 
  - Visual PDF upload with drag & drop
  - Chat interface with message history
  - Quick question buttons
  - Document statistics (chunks, conversations)
  - Clear chat history functionality
  - Real-time processing status
- **Run**: `streamlit run app.py`

### CLI Version ([`main.py`](main.py))
- **For**: Developers, debugging, automation
- **Features**:
  - Command-line interaction
  - Text-based Q&A
  - Debug output with chunk previews
  - No dependencies on web frameworks
  - Lightweight and fast
- **Run**: `python main.py`

Both share the same core RAG logic and produce identical results.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Gourisankar G**
- GitHub: [@Gourisankar25](https://github.com/Gourisankar25)
- LinkedIn: [Gourisankar G](https://www.linkedin.com/in/gourisankar-g-274aa2246)
- Email: kgourisankar01@gmail.com

## 🙏 Acknowledgments

- LangChain team for the excellent RAG framework
- Ollama team for making local LLMs accessible
- Streamlit for the intuitive UI framework
- Meta AI for Llama 3.2 model
- Facebook AI Research for FAISS vector search

## 📧 Contact

For questions or feedback, please open an issue or contact me at [kgourisankar01@gmail.com](mailto:kgourisankar01@gmail.com)

---

⭐ **Star this repo** if you find it helpful!
