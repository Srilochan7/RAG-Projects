# All your imports here...
import streamlit as st
import tempfile

@st.cache_resource
def create_vector_store(_uploaded_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load the PDF from the temporary path
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Split the documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings with the correct model name
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the FAISS vector store
    store = FAISS.from_documents(chunks, embeddings)
    
    return store