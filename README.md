# 📄 Docu-Query AI

A Streamlit application that lets you chat with your PDF documents using Google's Gemini AI.

## ✨ Features

- **PDF Upload**: Easily upload and process PDF documents
- **AI-Powered Q&A**: Ask questions about your document content
- **Fast Processing**: Quick setup with efficient text processing
- **Chat Interface**: Intuitive conversation-style interaction

## 🚀 Setup

1. **Install Dependencies**:
   ```bash
   pip install streamlit langchain-community langchain-google-genai faiss-cpu sentence-transformers
   ```

2. **Get API Key**:
   - Obtain a Google AI API key from [Google AI Studio](https://makersuite.google.com/)

3. **Configure Secrets**:
   - Create `.streamlit/secrets.toml` file
   - Add your API key:
     ```toml
     google_api_key = "your-api-key-here"
     ```

## 🎯 Usage

1. **Launch Application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF**:
   - Use the sidebar to upload your PDF document

3. **Ask Questions**:
   - Type questions in the chat interface
   - Get instant answers based on your document content

## 🔧 Technical Details

- **Text Processing**: Uses recursive text splitting for optimal chunking
- **Embeddings**: Employs HuggingFace's sentence transformers
- **Vector Store**: FAISS for efficient similarity search
- **AI Model**: Google's Gemini 2.5 Flash for fast, accurate responses

## 📋 Requirements

- Python 3.7 or higher
- Valid Google AI API key
- PDF files to analyze

-