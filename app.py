import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile
import time

# --- Set up the Streamlit UI ---
st.set_page_config(page_title="Docu-Query AI", page_icon="📄")
st.title("📄 Docu-Query AI: Chat with Your PDFs")
st.markdown("Upload a PDF and ask questions about its content.")

# Access the API key from the secrets file
try:
    google_api_key = st.secrets["google_api_key"]
except KeyError:
    st.error("Google AI API Key not found in `.streamlit/secrets.toml`. Please configure your secrets.")
    st.stop() # Stop the app if the key isn't found

# --- User input for PDF upload ---
with st.sidebar:
    
    pdf_file = st.file_uploader("Upload your PDF document", type="pdf")
    # Removed the text input for the API key

def process_pdf(pdf_file, google_api_key):
    # ... rest of your code remains the same ...
    # The rest of the function `process_pdf` does not need to change since it already accepts the key as an argument.
    # The `google_api_key` is now passed to this function.
    if not google_api_key:
        st.error("Please provide a Google AI API key.")
        return None

    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save PDF to temporary file
        status_text.text("Saving document...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        progress_bar.progress(20)

        # Load and split document
        status_text.text("Loading and splitting document...")
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = text_splitter.split_documents(documents)
        progress_bar.progress(40)

        # Set up embedding model (local, free)
        status_text.text("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        progress_bar.progress(60)

        # Create vector store with FAISS (faster than Chroma)
        status_text.text("Building search database...")
        db = FAISS.from_documents(docs, embeddings)
        progress_bar.progress(80)

        # Set up Google Gemini AI (FREE and FAST)
        status_text.text("Connecting to AI...")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=1024
        )
        
        # Create custom prompt for better results
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        progress_bar.progress(100)
        status_text.text("Ready to answer questions!")
        time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return qa

    except Exception as e:
        st.error(f"An error occurred: {e}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None

# --- Main application logic ---
if pdf_file and google_api_key:
    if "qa_chain" not in st.session_state or st.session_state.uploaded_pdf != pdf_file.name:
        st.session_state.uploaded_pdf = pdf_file.name
        with st.spinner("Setting up your document... This will be fast!"):
            st.session_state.qa_chain = process_pdf(pdf_file, google_api_key)

    if st.session_state.qa_chain:
        st.success(f"Document '{pdf_file.name}' ready for questions!")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about your document..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Getting answer..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    response = result["result"]
                except Exception as e:
                    response = f"Error: {str(e)}"
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif pdf_file and not google_api_key:
    st.warning("Please configure your Google AI API key in the `.streamlit/secrets.toml` file.")