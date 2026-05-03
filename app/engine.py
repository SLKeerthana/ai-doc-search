import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def process_document(file_path):
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split into chunks (Important for LLM context limits)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings and Store in ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def get_answer(vectorstore, query):
    # 4. Setup RetrievalQA Chain
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain.invoke(query)