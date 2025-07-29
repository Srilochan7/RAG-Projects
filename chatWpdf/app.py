import streamlit as st
import tempfile
# Also good practice to update langchain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Page Configuration and Title ---
st.set_page_config(page_title="Chat with PDF", layout="centered")
st.title("💬 Chat with Your PDF")

# --- PDF File Uploader (on the main page) ---
uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

# --- Spacer for visual separation ---
st.write("---") 

# --- Initialize Chat History ---
# This makes sure the chat messages are saved when the app reruns
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a PDF and ask me anything about it."}]

# --- Display Existing Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Box ---
if prompt := st.chat_input("Ask a question..."):
    # Add and display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Generate and Display Assistant's Response ---
    with st.chat_message("assistant"):
        if uploaded_file is None:
            response = "Please upload a PDF file first so I can answer your questions."
        else:
            # Placeholder for actual backend logic
            # In a real app, this is where you'd call your AI model
            response = f"Thinking about '{prompt}' from the file '{uploaded_file.name}'..."
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})