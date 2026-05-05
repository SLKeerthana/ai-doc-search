import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"

def process_document(file_path):
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split into chunks (Important for LLM context limits)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings and Store in ChromaDB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore

def get_vectorstore(documents=None):
    # Load the persisted vectorstore from disk
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Check if the database already exists on disk
    if os.path.exists(CHROMA_PATH) and documents is None:
        print("--- Loading existing Vector DB ---")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
    else:
        print("--- Creating new Vector DB ---")
        vectorstore = Chroma.from_documents(
            documents=documents, 
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )

    return vectorstore

def get_answer(vectorstore, query):
    # 4. Setup RetrievalQA Chain
    llm = ChatOllama(model="phi3", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain.invoke(query)