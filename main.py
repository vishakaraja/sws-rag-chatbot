import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

class Question(BaseModel):
    question: str

@app.post("/api/chat")
async def chat(body: Question):
    results = vectorstore.similarity_search(body.question, k=4)
    
    context = "\n\n".join([doc.page_content for doc in results])
    
    sources = list(set([
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in results
    ]))
    
    prompt = f"""You are an HR assistant for SWS AI company.
Answer the question using ONLY the context below.
If the answer is not in the context, say:
I don't have that information in the company documents.

Context:
{context}

Question: {body.question}

Answer:"""
    
    response = llm.invoke(prompt)
    
    return {
        "answer": response.content,
        "sources": sources
    }

@app.get("/")
def root():
    return {"message": "SWS AI RAG Chatbot is running!"}
