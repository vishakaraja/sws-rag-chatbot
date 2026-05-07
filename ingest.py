import os
import sys
from dotenv import load_dotenv

print(" Starting ingest script...")
sys.stdout.flush()

load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: Load all PDFs
print("📄 Loading PDFs...")
sys.stdout.flush()

documents = []
docs_path = "documents"

for filename in os.listdir(docs_path):
    if filename.endswith(".pdf"):
        print(f"  Loading: {filename}")
        sys.stdout.flush()
        loader = PyMuPDFLoader(os.path.join(docs_path, filename))
        docs = loader.load()
        documents.extend(docs)
        print(f"  Loaded: {filename}")
        sys.stdout.flush()

print(f"\n📊 Total pages loaded: {len(documents)}")

# Step 2: Split into chunks
print("\n Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f" Total chunks created: {len(chunks)}")

# Step 3: Store in ChromaDB
print("\n Creating embeddings and storing in ChromaDB...")
print(" This may take 2-3 minutes first time...")
sys.stdout.flush()

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("\n Done! All documents stored in ChromaDB.")
print("You can now run: uvicorn main:app --reload")
