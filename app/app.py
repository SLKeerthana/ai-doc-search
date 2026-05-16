import streamlit as st
import os
from engine import get_vectorstore, get_answer

# --- Page Config ---
st.set_page_config(page_title="AI Doc Search", layout="wide")
st.title("📄 Professional AI Document Search")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    # Load the persistent DB if it exists
    st.session_state.vectorstore = get_vectorstore()

# --- Sidebar: File Management ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if st.button("Process Document") and uploaded_file:
        with st.spinner("Analyzing PDF..."):
            # 1. Save temp file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Process and Update Persistence
            # (Note: engine.py needs a loader/splitter logic we discussed earlier)
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            st.session_state.vectorstore = get_vectorstore(documents=documents)
            st.success("Indexed & Saved to Disk!")
            os.remove(temp_path)

# --- Chat Interface ---
# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            response = get_answer(st.session_state.vectorstore, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and process a document first.")