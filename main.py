from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

#from sentence_transformers import SentenceTransformer
#import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict
import threading
import os

llm_pipeline = None
llm_lock = threading.Lock()

load_dotenv()  # reads .env file
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

#api_key = os.getenv("OPENAI_API_KEY")

#if not api_key:
#    raise ValueError("OPENAI_API_KEY not found. Did you set it in your .env file?")

# FastAPI: Framework web
# uvicorn: El servidor que ejecutará FastAPI.

# --- 1. Application Initialization ---

app = FastAPI()
# Inicializa ChromaDB en modo de cliente (en memoria por defecto)
# ¡Nota!: Los datos se perderán cada vez que reinicies la aplicación.

# SOLUTION 1: Use persistent storage instead of in-memory
# This will create a local directory to store the database

# PERSIST_DIRECTORY = "chroma_db"
# client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Initialize Qdrant client
print (f"URL: {QDRANT_URL}")
print (f"KEY: {QDRANT_API_KEY}")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Crea una colección para nuestras incrustaciones.
# Usamos un modelo de Hugging Face para generar las incrustaciones.
# 'all-MiniLM-L6-v2' es un modelo pequeño y muy eficiente.

COLLECTION_NAME = "document_collection"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# --- 2. Helper Functions ---

def get_or_create_collection():
    """Get or create the collection safely"""
    try:
        collection = client.get_collection(
            collection_name=COLLECTION_NAME
        )
        print(f"Collection '{COLLECTION_NAME}' found with {collection.count()} documents")
        return collection
    except Exception as e:
        print(f"Collection not found, creating new one: {e}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # embedding dimension for all-MiniLM-L6-v2
                distance=Distance.COSINE
            )
        )
        collection = client.get_collection(
            collection_name=COLLECTION_NAME
        )
        print(f"Collection '{COLLECTION_NAME}' created")
        return collection

# Initialize collection at startup
#collection = get_or_create_collection()




# collection = client.get_or_create_collection(
#     name="document_collection",
#     embedding_function=embedding_function
# )

#llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)

# Load a local model (downloads the first time, then caches)
# hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small") small model

#sentence-transformers: Un paquete para usar modelos de incrustación de Hugging Face.

#pypdf: Una librería para leer archivos PDF.



def read_pdf(file: UploadFile) -> str:
    """Lee el texto de un archivo PDF."""
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or "[No text on page]"
        return text
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Divide un texto largo en fragmentos más pequeños con solapamiento."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def reset_collection():
    """Reset the collection safely"""
    global collection
    try:
        print(f"Collection to delete: '{COLLECTION_NAME}' ")
        #client.delete_collection(name=COLLECTION_NAME)
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted")
    except Exception as e:
        print(f"Error deleting collection: {e}")
    # collection = client.create_collection(
    #     name=COLLECTION_NAME,
    #     embedding_function=embedding_function
    # )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE) 
    )
    collection = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' recreated")
    return collection

def get_llm_pipeline():
    global llm_pipeline
    with llm_lock:
        if llm_pipeline is None:
            llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return llm_pipeline

def query_collection(query_text: str, top_k: int = 5):
    query_vector = embedding_function(query_text)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return results
#hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-large") #bigger model
#hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small") #smaller model
#hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
                       
#llm = HuggingFacePipeline(pipeline=hf_pipeline)

#template = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])
#chain = LLMChain(llm=llm, prompt=template)

class QueryRequest(BaseModel):
    query: str

class SearchRequest(BaseModel):
    query: str

# --- 3. API Endpoints ---
@app.post("/train")
async def train_with_document(file: UploadFile = File(...)):
    """
    Endpoint to train the model with a PDF file.
    - Upload PDF file
    - Split PDF text into chunks  
    - Convert chunks to vector embeddings
    - Save embeddings to vector database
    """
    global collection
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser un PDF."
        )

    
    try:
        # Reset collection for new training
        collection = reset_collection()
        # 1. Read PDF
        document_text = read_pdf(file)
        if not document_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF."
            )

        # 2. Split text into chunks
        chunks = split_text_into_chunks(document_text)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks generated from PDF."
            )
        # 3. Prepare data for ChromaDB
        #documents = chunks
        #metadatas = [{"source": file.filename} for _ in chunks]
        #ids = [f"id{i}" for i in range(len(chunks))]

        # 4. Add embeddings to database in batches (to avoid memory issues)
        # missing work here:
        # collection.add(
        #     documents=documents,
        #     metadatas=metadatas,
        #     ids=ids
        # )
        embeddings = [embedding_function(x) for x in chunks]
        points = [
            models.PointStruct(
                id=i,
                vector=embeddings[i],
                payload={"text": chunks[i], "source": file.filename}
            )
            for i in range(len(chunks))
        ]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        count = collection.count()
        print(f"Collection now has {count} documents")

        return {
            "message": "Document processed and trained successfully.",
            "chunks_count": len(chunks),
            "total_documents_in_db": count
        }
    
    except Exception as e:
        print(f"Training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@app.post("/search")
async def search_in_document(query: Dict[str, str]):
    """
    Endpoint to search in trained documents.
    - Receives text query
    - Searches for most similar chunks in vector database
    - Returns text of found chunks
    """
    global collection
    #search_query = req.query
    search_query = query.get("query")
    if not search_query:
        raise HTTPException(
            status_code=400,
            detail="'query' field required in request body."
        )
    try:
        # Verify collection exists and has data
        if collection is None:
            collection = get_or_create_collection()
    
        count = collection.count()
        print(f"Collection has {count} documents")
        if count == 0:
            return {
                "message": "No documents found in database. Please train first.",
                "results": []
                }
        # Perform vector search in ChromaDB
        results = collection.query(
            query_texts=[search_query],
            n_results=min(5, count)  # Don't request more results than available
        )
        
        # Extract texts and distances from results
        found_documents = []
        if results['documents']:
            for i, document_texts in enumerate(results['documents'][0]):
                found_documents.append({
                    "text": document_texts,
                    "distance": results['distances'][0][i]
                })

        return {
            "message": "Search completed.",
            "results": found_documents
        }
    
    except Exception as e:
        print(f"Search error: {e}")
        # Try to reinitialize collection if it's corrupted
        try:
            collection = get_or_create_collection()
            count = collection.count()
            if count == 0:
                return {
                    "message": "Database was reset. Please train again.",
                    "results": []
                }
        except Exception as reinit_error:
            print(f"Could not reinitialize collection: {reinit_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
@app.post("/ask")
async def ask(req: QueryRequest):
    
    
    
    try:
        #Option1:Retrieve relevant chunks
        #answer = chain.run(question=req.query)  #Works fine. Retrieve chuncks
        #return {"answer": answer}
        
        #Option2: 
        #2.1 Retrieve relevant chunks
        #results = collection.query(
        #query_texts=[req.query],
        #n_results=5
        #)
        results = query_collection(req.query,5)
        documents = results['documents'][0]
        
        #2.2 Create a prompt with the retrieved context
        context = " ".join(documents)
        prompt = f"I am a full stack software engineer (since December 2018) with a portafolio website showcasing my skills. Your goal as a chatbot embedded in such website is to answer questions of recruiters. Please answer the following question in a concise, clear and professional way, using the details below from my curriculum:\n\nContext: {context}\n\nQuestion: {req.query}\nAnswer:"

        #2.3 Run the LLM
        hf_pipeline = get_llm_pipeline()
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        answer = llm.invoke(prompt)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    """Get database status"""
    try:
        count = collection.count() if collection else 0
        return {
            "status": "active",
            "collection_name": COLLECTION_NAME,
            "documents_count": count,
            "persist_directory": "cloud.qdrant"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
@app.get("/")
def read_root():
    return {"msg": "RAG Application Active", "status": "OK"}

# --- 4. Startup Event ---
# on_event is deprecated in FASTAPI
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global collection
    try:
        collection = get_or_create_collection()
        print(f"Application started successfully. Collection has {collection.count()} documents.")
    except Exception as e:
        print(f"Startup error: {e}")
#http://127.0.0.1:8000/docs#
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)