from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from sentence_transformers import SentenceTransformer
#import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
#from chromadb.utils import embedding_functions
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict
import threading
import asyncio
import os



load_dotenv()  # reads .env file
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "document_collection"

#api_key = os.getenv("OPENAI_API_KEY")

#if not api_key:
#    raise ValueError("OPENAI_API_KEY not found. Did you set it in your .env file?")

# FastAPI: Framework web
# uvicorn: El servidor que ejecutará FastAPI.

# --- 1. Application Initialization ---

app = FastAPI(title="RAG Application", version="1.0.0")
# Inicializa ChromaDB en modo de cliente (en memoria por defecto)
# ¡Nota!: Los datos se perderán cada vez que reinicies la aplicación.

# SOLUTION 1: Use persistent storage instead of in-memory
# This will create a local directory to store the database

# PERSIST_DIRECTORY = "chroma_db"
# client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Global variables
client = None
embedding_model = None
llm_pipeline = None
llm_lock = asyncio.Lock()

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


model_name = "sentence-transformers/all-MiniLM-L6-v2"
#embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# --- 2. Helper Functions ---
def initialize_qdrant_client():
    """Initialize Qdrant client safely"""
    global client
    try:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"Qdrant client initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize Qdrant client: {e}")
        return False

def initialize_embedding_model():
    """Initialize embedding model safely"""
    global embedding_model
    try:
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Embedding model initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        return False

def get_or_create_collection():
    """Get or create the collection safely"""
    try:
        if not client:
            raise ValueError("Qdrant client not initialized")
        if client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists!!")
            collection = client.get_collection(
            collection_name=COLLECTION_NAME
            )
            return collection
        else: 
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
        
    except Exception as e:
        print(f"Error during collection setup: {e}")
        return False

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
        file.file.seek(0)  # Reset file pointer
        reader = PdfReader(file.file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {e}")
                continue
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Divide un texto largo en fragmentos más pequeños con solapamiento."""
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def reset_collection():
    """Reset the collection safely"""
    global collection
    try:
        print(f"Collection to delete: '{COLLECTION_NAME}' ")
        #client.delete_collection(name=COLLECTION_NAME)
        if client.collection_exists(collection_name=COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE) 
        )
        collection = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' recreated")
        return collection
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False
    

async def get_llm_pipeline():
    global llm_pipeline
    async with llm_lock:
        if llm_pipeline is None:
            llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
    return llm_pipeline

# def query_collection(query_text: str, top_k: int = 5):
#     query_vector = embedding_function(query_text)
#     results = client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=top_k
#     )
#     return results

def get_collection_count():
    try:
        count_response = client.count(collection_name=COLLECTION_NAME)
        count = count_response.count
        return count
    except Exception as e:
        print(f"Error getting collection count: {e}")
        return 0

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
    if not client:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")
    
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")
    
    try:
        # Reset collection for new training
        reset_collection()
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
        # Create embeddings and upsert
        points = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = embedding_model.encode(chunk).tolist()
                point = PointStruct(
                    id=i,
                    vector=embedding,
                    payload={"text": chunk, "source": file.filename}
                )
                points.append(point)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        count = get_collection_count()
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
    """Ask question with RAG"""
    
    try:
        if not client:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        #Option1:Retrieve relevant chunks
        #answer = chain.run(question=req.query)  #Works fine. Retrieve chuncks
        #return {"answer": answer}
        
        #Option2: 
        #2.1 Retrieve relevant chunks
        #results = collection.query(
        #query_texts=[req.query],
        #n_results=5
        #)
        # Get relevant documents
        query_embedding = embedding_model.encode(req.query).tolist()
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )
        print(f"Results:{results}")
        if not results:
            return {"answer": "No relevant documents found. Please train the system first."}
        
        # Create context
        # context = " ".join([result.payload["text"] for result in results])
        clean_texts = [
            " ".join(result.payload["text"].split())  # collapses all whitespace to single spaces
            for result in results
        ]
        context = " ".join(clean_texts)
        #prompt = f"""Based on the following context, answer the question concisely and professionally:
        prompt = f""" You are a helpful assistant trained to answer questions about my experience that appears in the context.
        Please provide a detailed, professional, and comprehensive paragraph summarizing the following context.
        Context: {context}

        Question: {req.query}
        Answer professionally:
        """
        
        # Get LLM response
        hf_pipeline = await get_llm_pipeline()
        #llm = HuggingFacePipeline(pipeline=hf_pipeline)
        llm = HuggingFacePipeline(pipeline=hf_pipeline, pipeline_kwargs={"max_new_tokens": 900,
        "temperature": 0.9,
        "min_new_tokens": 1200,
        "num_beams": 9,
        "do_sample": True,
        "top_p": 0.9})
        
        #answer = llm.invoke(prompt)
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, llm.invoke, prompt)
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    """Get database status"""
    """Health check endpoint"""
    collection_exists = False
    doc_count =0
    if(client):
        try:
            count_response = client.count(collection_name=COLLECTION_NAME)
            collection_exists=True
            doc_count = count_response.count
        except Exception as e:
            print(f"Error in status: {e}")
            collection_exists = False
            doc_count =0
    return {
        "status": "healthy",
        "qdrant_connected": client is not None,
        "collection_exists": collection_exists,
        "document_count": doc_count
    }

@app.get("/")
def read_root():
    return {"msg": "RAG Application Active", "status": "OK"}

# --- 4. Startup Event ---
# on_event is deprecated in FASTAPI
@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    print("Starting RAG application...")
    
    # Initialize services
    if not initialize_qdrant_client():
        print("WARNING: Qdrant client initialization failed")
    
    if not initialize_embedding_model():
        print("WARNING: Embedding model initialization failed")
    
    # Create collection if needed
    # if client:
    #     get_or_create_collection()
    
    print("RAG application startup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)