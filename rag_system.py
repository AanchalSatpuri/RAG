from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
UPLOAD_FOLDER = '/Users/aanchal.satpuri/Desktop/projects/Rag/files/'
pages = []

# Load files from folder
for file in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        pages.extend(loader.load_and_split())
    elif file.endswith('.docx') or file.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
        pages.append(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        pages.extend(loader.load())
    elif file.endswith('.json'):
        loader = JSONLoader(file_path)
        pages.extend(loader.load())

# ✅ Load web content (add your URLs here)
web_urls = [
    "https://wework.co.in/",
    "https://wework.co.in/workspaces/private-office-space/"
]

for url in web_urls:
    try:
        loader = WebBaseLoader(url)
        pages.extend(loader.load())
    except Exception as e:
        print(f"Failed to load {url}: {e}")

# Flatten the documents
flat_pages = []
for page in pages:
    if isinstance(page, list):
        flat_pages.extend(page)
    else:
        flat_pages.append(page)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
split_docs = text_splitter.split_documents(flat_pages)

# Create vector store
vectorstore = DocArrayInMemorySearch.from_documents(split_docs, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# Sample questions
questions = [
    "discount on PO",
    "What do you get with your WeWork Private Office?"
]

for question in questions:
    print(f"Question: {question}")
    retrieved_chunks = retriever.invoke(question)
    print("Retrieved Chunks:")
    for chunk in retrieved_chunks:
        print(chunk.page_content)
        print("-" * 80)
