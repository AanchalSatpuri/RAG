from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
)
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------
# 1. Embedding model
# ---------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')

# ---------------------------------------------
# 2. Load files from local folder
# ---------------------------------------------
UPLOAD_FOLDER = '/Users/aanchal.satpuri/Desktop/projects/Rag/files/'
pages = []

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

# ---------------------------------------------
# 3. Load URLs using Playwright instead of WebBaseLoader
# ---------------------------------------------
def fetch_rendered_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=10000)
        page.wait_for_load_state('domcontentloaded')
        content = page.content()
        browser.close()
        return content

def load_url_with_playwright(url: str) -> list[Document]:
    html = fetch_rendered_html(url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url})]

# ---------------------------------------------
# 4. Read URLs from file and load them
# ---------------------------------------------
URL_FOLDER = '/Users/aanchal.satpuri/Desktop/projects/Rag/url_files'
URL_FILE = 'urls.txt'

url_file_path = os.path.join(URL_FOLDER, URL_FILE)
with open(url_file_path, 'r') as f:
    web_urls = f.read().splitlines()

# Function to fetch and process a single URL
def fetch_and_process_url(url):
    try:
        docs = load_url_with_playwright(url)
        return docs
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

# Use ThreadPoolExecutor with more workers
with ThreadPoolExecutor(max_workers=20) as executor:
    results = list(executor.map(fetch_and_process_url, web_urls))

# Flatten the results
for docs in results:
    pages.extend(docs)

# ---------------------------------------------
# 5. Flatten and split into chunks
# ---------------------------------------------
flat_pages = []
for page in pages:
    if isinstance(page, list):
        flat_pages.extend(page)
    else:
        flat_pages.append(page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
split_docs = text_splitter.split_documents(flat_pages)

# ---------------------------------------------
# 6. Vector store and retrieval
# ---------------------------------------------
vectorstore = DocArrayInMemorySearch.from_documents(split_docs, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# ---------------------------------------------
# 7. Sample Questions
# ---------------------------------------------
questions = [
    "discount on all access plus",
    "discount on private office",
    "What do you get with your WeWork Private Office?"
]

# Define the number of chunks to retrieve
num_chunks_to_retrieve = 5  # Adjust this number as needed

for question in questions:
    print(f"Original Question: {question}")
    # Pass the parameter to limit the number of retrieved chunks
    retrieved_chunks = retriever.invoke(question, top_k=num_chunks_to_retrieve)
    print("Retrieved Chunks:")
    for chunk in retrieved_chunks:
        print(chunk.page_content)
        print("-" * 80)
