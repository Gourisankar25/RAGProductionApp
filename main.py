from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#import os

#load_dotenv()

#Initialize LLM
llm = Ollama(model="llama3.2", temperature=0) #deterministic responses  


#Loads your PDF from given path
pdf_reader = PyPDFLoader(r"C:\Users\gourig\Downloads\Gap-Analysis-2026-01-27-14-54-32.pdf")
#Extracts all text from the PDF
documents = pdf_reader.load()
#Breaks the text into 1000-character chunks with 200-character overlap (helps maintain context between chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)



#Converts text chunks into numerical vectors using OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama3.2")
#Creates a FAISS vector store from the document chunks and their embeddings
vectorstore = FAISS.from_documents(chunks, embeddings)
#Sets up a retriever to fetch relevant document chunks based on user queries
retriever = vectorstore.as_retriever()



#Create a simple RAG chain
#Defines how to format the question and context for the LLM
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    #Combines multiple retrieved chunks into a single text string
    return "\n\n".join(doc.page_content for doc in docs)



'''
Chains are sequences of operations that link together 
components like prompts, LLMs, and other tools 
to create a workflow.
'''

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What is this document all about?"

result = chain.invoke(query)

print("Answer:", result)
