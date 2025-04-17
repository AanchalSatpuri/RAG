# Retrieval-Augmented Generation (RAG) System

This project implements a Retrieval-Augmented Generation (RAG) system that processes various document types and web content to enable semantic search and retrieval. The system uses the `langchain` library for document loading and vector storage, and `sentence-transformers` for generating embeddings.

## Features
- Load and process documents from PDF, DOCX, TXT, and JSON files.
- Retrieve and process web content from specified URLs.
- Split documents into manageable chunks for processing.
- Generate embeddings using a pre-trained model from `sentence-transformers`.
- Perform semantic search and retrieval based on user queries.

## Setup

### Prerequisites
- Python 3.7+
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Directory Structure
- Place your documents in the `random_files_directory` located at `/Users/random_files_directory/`.
- Update the `UPLOAD_FOLDER` variable in `rag_system.py` if your directory structure is different.

## Usage

1. **Run the Script**:
   Execute the script to process documents and perform retrieval:
   ```bash
   python rag_system.py
   ```

2. **Sample Queries**:
   The script includes sample questions to demonstrate retrieval capabilities. Modify the `questions` list in `rag_system.py` to test with different queries.

## Notes
- Ensure that the URLs in the `web_urls` list are accessible and valid.
- Adjust the `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` to optimize document splitting for your use case.
- Adjust the chunk size and overlapping according to your needs to improve retrieval performance and accuracy.

## License
This project is licensed under the MIT License. 
