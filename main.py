# from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import re
#from sentence_transformers import SentenceTransformer
#import chromadb

# from qdrant_client.models import Distance, VectorParams, PointStruct
# from qdrant_client.http import models
# from chromadb.utils import embedding_functions
from pypdf import PdfReader
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

from typing import List, Dict
import threading
import asyncio
import os
import requests


load_dotenv()  # reads .env file
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "document_collection"
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
#MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L6-v2"  #"sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

#MODEL_ID = "sentence-transformers/all-mpnet-base-v2"

#MODEL_ID = "text-embedding-ada-002"  #NOT FOUND
#MODEL_ID = "gpt2" #NOT FOUND
#MODEL_ID = "dbmdz/bert-large-cased-finetuned-conll03-english"  #INPUT expected = numbers
#MODEL_ID = "YaYaB/yb_test_inference_clip_embedding" #NOT FOUND
#MODEL_ID = "deerslab/llama-7b-embeddings" #NOT FOUND
#MODEL_ID = "BAAI/bge-large-en-v1.5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
#MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
#HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"
HF_HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

print("QDRANT_URL: " + QDRANT_URL)
print("QDRANT_API_KEY:", "loaded" if QDRANT_API_KEY else "missing")
print("HUGGING_FACE_TOKEN:", "loaded" if HUGGING_FACE_TOKEN else "missing")
print("HF_API_URL: " + HF_API_URL)
print("HEADERS: " + str(HF_HEADERS))

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
HGClient = InferenceClient(api_key=HUGGING_FACE_TOKEN)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
#QDRANT: Is an online vector database

# Initialize Qdrant client
# print (f"URL: {QDRANT_URL}")
# print (f"KEY: {QDRANT_API_KEY}")
# client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY
# )

# Crea una colección para nuestras incrustaciones.
# Usamos un modelo de Hugging Face para generar las incrustaciones.
# 'all-MiniLM-L6-v2' es un modelo pequeño y muy eficiente.


# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# --- 2. Helper Functions ---
def initialize_qdrant_client():
    """Initialize Qdrant client safely"""
    global client
    try:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # client = QdrantClient(
        #     url="https://b15776a3-3430-4d25-bd2e-d85f6ea4a3cc.us-east4-0.gcp.cloud.qdrant.io",
        #     api_key="YOUR_API_KEY"
        # )
        print(client.get_collections())
        print(f"Qdrant client initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize Qdrant client: {e}")
        return False

# --- Call Hugging Face API:
def get_hf_embedding(text: str):
    """Call Hugging Face API to get embedding for one text chunk"""
    max_len = 500  # keep under safe limit (tokens, approx chars)
    if len(text) > max_len:
        text = text[:max_len]
    
    if not text.strip():
        # Raise an exception so the loop's try/except block skips this chunk.
        raise ValueError("Input text chunk is empty or whitespace-only.")
    
    # FIX: The standard Feature Extraction payload requires the input text to be in a list.
    #payload = {"inputs": text}
    #payload = {"sentences": [text]}
    try:
        result = HGClient.feature_extraction(text, model=MODEL_ID)
        return result  # list of floats (vector)
    except:
        raise ValueError("Error in get_hf_embedding.")
    # response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
    # if response.status_code != 200:
    #     raise Exception(f"HF API error {response.status_code}: {response.text}")
    # #response.raise_for_status()
    # #return response.json()[0]  # returns a list of floats (vector size 384)
    # data = response.json()

    # # Some models return [[...]], some just [...]
    # if isinstance(data, list) and isinstance(data[0], list):
    #     return data[0]  # take first embedding
    # elif isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
    #     # Fallback in case a list of floats is returned directly, though less common
    #     return data
    # else:
    #     raise Exception(f"Unexpected HF response format: {data}")

def split_question(question: str):
    # lowercase, remove punctuation
    question = re.sub(r'[^\w\s]', '', question.lower())
    # split words
    words = question.split()
    # remove trivial words
    stopwords = {"the","is","do","have","with","a","an","of","to"}
    return [w for w in words if w not in stopwords]

def retrieve_relevant_chunks(question: str, top_k=5):
    words = split_question(question)
    results = []

    for w in words:
        embedding = get_hf_embedding(w)
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k
        )
        results.extend(hits)

    # remove duplicates & rank by score
    unique = {}
    for hit in results:
        text = hit.payload["text"]
        score = hit.score
        if text not in unique or score > unique[text]:
            unique[text] = score
    
    # sort by best score
    sorted_hits = sorted(unique.items(), key=lambda x: x[1], reverse=True)
    return [text for text, score in sorted_hits[:top_k]]

def summarize_chunks(question, chunks, max_chunk_tokens=800):
    try:
        context = " ".join(chunks)
        # Truncate context if it is too long for the model
        # Tokenize and truncate properly
        inputs = tokenizer(
            context,
            max_length=max_chunk_tokens,
            truncation=True,
            return_tensors="pt"
        )
        text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        #prompt = f"Answer this question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        raise ValueError(f"Error in: summarize_chunks: {str(e)}")

# def initialize_embedding_model():
#     """Initialize embedding model safely"""
#     global embedding_model
#     try:
#         embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#         print("Embedding model initialized successfully")
#         return True
#     except Exception as e:
#         print(f"Failed to initialize embedding model: {e}")
#         return False

# def get_or_create_collection():
#     """Get or create the collection safely"""
#     try:
#         if not client:
#             raise ValueError("Qdrant client not initialized")
#         if client.collection_exists(collection_name=COLLECTION_NAME):
#             print(f"Collection '{COLLECTION_NAME}' already exists!!")
#             collection = client.get_collection(
#             collection_name=COLLECTION_NAME
#             )
#             return collection
#         else: 
#             print(f"Collection not found, creating new one: {e}")
#             client.create_collection(
#                 collection_name=COLLECTION_NAME,
#                 vectors_config=VectorParams(
#                 size=384,  # embedding dimension for all-MiniLM-L6-v2
#                 distance=Distance.COSINE
#                 )
#             )   
#             collection = client.get_collection(
#                 collection_name=COLLECTION_NAME
#             )
#             print(f"Collection '{COLLECTION_NAME}' created")
#             return collection
        
#     except Exception as e:
#         print(f"Error during collection setup: {e}")
#         return False

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
        print(f"Collection to recreate: '{COLLECTION_NAME}' ")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' recreated")
        return True
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
        return count_response.count
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
            detail="The file should have PDF format."
        )
    if not client:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")
  
    try:
    #     # Reset collection for new training
        reset_collection()
    # 1. Read PDF
        document_text = read_pdf(file)
        if not document_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF."
            )

    #     # 2. Split text into chunks
        chunks = split_text_into_chunks(document_text)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks generated from PDF."
            )
    
        points = []
        
 
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_hf_embedding(chunk)  # 👈 Hugging Face call
                point = {
                    "id": i,
                    "vector": embedding,
                    "payload": {"text": chunk, "source": file.filename}
                }
                points.append(point)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        if points:
            # Ensure collection exists
            try:
                collections = [c.name for c in client.get_collections().collections]
                if COLLECTION_NAME not in collections:
                    print(f"Creating new collection '{COLLECTION_NAME}'")
                    client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                else:
                    print(f"Collection '{COLLECTION_NAME}' already exists")
            except Exception:
                print(f"Creating new collection '{COLLECTION_NAME}'")
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        return {
            "message": "Document processed and trained successfully.",
             "chunks_count": len(chunks),
            # "total_documents_in_db": count
        }
    
    except Exception as e:
        print(f"Training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )
    #   Foreach chunk Calls an external free-tier API(Hugging face)
    #   To get the vector embedding.


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

@app.post("/search")
async def search_in_document(query: Dict[str, str]):
    """
    Endpoint to search in trained documents.
    - Receives text query
    - Searches for most similar chunks in vector database
    - Returns text of found chunks
    """
    # global collection
    # #search_query = req.query
    # search_query = query.get("query")
    # if not search_query:
    #     raise HTTPException(
    #         status_code=400,
    #         detail="'query' field required in request body."
    #     )
    # try:
    #     # Verify collection exists and has data
    #     if collection is None:
    #         collection = get_or_create_collection()
    
    #     count = collection.count()
    #     print(f"Collection has {count} documents")
    #     if count == 0:
    #         return {
    #             "message": "No documents found in database. Please train first.",
    #             "results": []
    #             }
    #     # Perform vector search in ChromaDB
    #     results = collection.query(
    #         query_texts=[search_query],
    #         n_results=min(5, count)  # Don't request more results than available
    #     )
        
    #     # Extract texts and distances from results
    #     found_documents = []
    #     if results['documents']:
    #         for i, document_texts in enumerate(results['documents'][0]):
    #             found_documents.append({
    #                 "text": document_texts,
    #                 "distance": results['distances'][0][i]
    #             })

    return {
        "message": "Search completed.",
        # "results": found_documents
    }
    
    # except Exception as e:
    #     print(f"Search error: {e}")
    #     # Try to reinitialize collection if it's corrupted
    #     try:
    #         collection = get_or_create_collection()
    #         count = collection.count()
    #         if count == 0:
    #             return {
    #                 "message": "Database was reset. Please train again.",
    #                 "results": []
    #             }
    #     except Exception as reinit_error:
    #         print(f"Could not reinitialize collection: {reinit_error}")
        
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"Search failed: {str(e)}"
    #     )
@app.post("/extract")
async def extract(req: QueryRequest):
    """Ask question with RAG"""
    
    try:
        if not client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        # 1. Get query embedding from Hugging Face API
        query_embedding = get_hf_embedding(req.query)
        
        #Option1:Retrieve relevant chunks
        #answer = chain.run(question=req.query)  #Works fine. Retrieve chuncks
        #return {"answer": answer}
        
        # 2. Search Qdrant for top-k relevant chunks
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )

        if not results:
            return {"answer": "No relevant documents found. Please train the system first."}

        # 3. Build context from retrieved chunks
        context = " ".join([hit.payload["text"] for hit in results])

        # 4. Build prompt
        prompt = f"""Based on the following context, provide a long and professional answer to the question.

        Context: {context}

        Question: {req.query}
        Answer:"""

        # 5. Call Hugging Face LLM API for final answer (example: flan-t5-small)
        #hf_generation_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
        #hf_generation_url = "https://api-inference.huggingface.co/models/sshleifer/tiny-gpt2"
        #hf_generation_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
        # hf_generation_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        
        #successful
        #hf_generation_url = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"  #successful
        hf_generation_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"  #successful

        response = requests.post(
            hf_generation_url,
            headers=HF_HEADERS,
            json={"inputs": {"question": req.query, "context": context}},
            timeout=120
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Extract failed: {response.text}")
        data = response.json()

        #For: deepset/roberta-base-squad2
        answer = data.get("answer", "").strip()
        score = data.get("score", 0.0)

        # Handle summarization vs generation
        # answer = None
        # if isinstance(data, list):
        #     if "generated_text" in data[0]:
        #         answer = data[0]["generated_text"]
        #     elif "summary_text" in data[0]:
        #         answer = data[0]["summary_text"]
        # elif isinstance(data, dict):
        #     answer = data.get("generated_text") or data.get("summary_text")

        # if not answer:
        #     raise HTTPException(status_code=500, detail=f"Unexpected HF response: {data}")
        if not answer:
            return {"answer": "No confident answer found.", "context": context, "score": score}
        return {"answer": answer.strip(), "context": context}
    
        #Option2: 
        #2.1 Retrieve relevant chunks
        #results = collection.query(
        #query_texts=[req.query],
        #n_results=5
        #)
        # Get relevant documents
        # query_embedding = embedding_model.encode(req.query).tolist()
        # results = client.search(
        #     collection_name=COLLECTION_NAME,
        #     query_vector=query_embedding,
        #     limit=3
        # )
        
        #if not results:
        # return {"answer": "No relevant documents found. Please train the system first."}
        
        # # Create context
        # context = " ".join([result.payload["text"] for result in results])
        # prompt = f"""Based on the following context, answer the question concisely and professionally:

        # Context: {context}

        # Question: {req.query}
        # Answer:"""
        
        # Get LLM response
        hf_pipeline = await get_llm_pipeline()
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        answer = llm.invoke(prompt)
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extract failed: {str(e)}")

@app.post("/summarize")
async def summarize(req: QueryRequest):
    """summarize question with RAG"""
    
    try:
        if not client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        # 1. Get query embedding from Hugging Face API
        query_embedding = get_hf_embedding(req.query)
        
        #Option1:Retrieve relevant chunks
        
        # 2. Search Qdrant for top-k relevant chunks
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )

        if not results:
            return {"answer": "No relevant documents found. Please train the system first."}

        # 3. Build context from retrieved chunks
        context = " ".join([hit.payload["text"] for hit in results])

        # 4. Build prompt
        prompt = f"""Summarize the answer to the question: '{req.query}' In the context you find the details of the work experience (jobs functions), provide a long answer to the question professionally.

        Context: {context}

        Question: {req.query}
        Answer:"""

        # 5. Call Hugging Face LLM API for final answer (example: flan-t5-small)
        #successful
        hf_generation_url = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"  #successful

        response = requests.post(
            hf_generation_url,
            headers=HF_HEADERS,
             json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 400, "temperature": 0.6}
            },
            timeout=120
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"summarize failed: {response.text}")
        data = response.json()

        # Handle summarization vs generation
        answer = None
        if isinstance(data, list):
            if "generated_text" in data[0]:
                answer = data[0]["generated_text"]
            elif "summary_text" in data[0]:
                answer = data[0]["summary_text"]
        elif isinstance(data, dict):
            answer = data.get("generated_text") or data.get("summary_text")

        if not answer:
            raise HTTPException(status_code=500, detail=f"Unexpected HF response: {data}")

        return {"answer": answer.strip(), "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"summarize failed: {str(e)}")

@app.post("/textgeneration")
async def ask(req: QueryRequest):
    """Ask question with RAG"""
    """Local Model"""
    try:
        if not client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        # 1. Get query embedding from Hugging Face API
        query_embedding = get_hf_embedding(req.query)
        
        #Option1:Retrieve relevant chunks
        
        # 2. Search Qdrant for top-k relevant chunks
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )

        if not results:
            return {"answer": "No relevant documents found. Please train the system first."}

        # 3. Build context from retrieved chunks
        context = " ".join([hit.payload["text"] for hit in results])

        # 4. Build prompt
        pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        result = pipe("Answer professionally: '{req.query}' Context: ...", max_new_tokens=200)
        return result[0]["generated_text"]
        
        # prompt = f"""Summarize the answer to the question: '{req.query}' In the context you find the details of the work experience (jobs functions), answer the question professionally.

        # Context: {context}

        # Question: {req.query}
        # Answer:"""

        # 5. Call Hugging Face LLM API for final answer (example: flan-t5-small)
        #successful
        # hf_generation_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"  #successful

        # response = requests.post(
        #     hf_generation_url,
        #     headers=HF_HEADERS,
        #     json={
        #         "inputs": prompt,
        #         "parameters": {
        #         "max_new_tokens": 400,
        #         "temperature": 0.7,
        #         "top_p": 0.9
        #         }
        #     },
        #     timeout=120
        # )
        # print("HF status code:", response.status_code)
        # print("HF raw response:", response.text)
        
        # if response.status_code != 200:
        #     raise HTTPException(status_code=500, detail=f"Ask failed: {response.text}")
        # data = response.json()

        # # Handle summarization vs generation
        # answer = None
        # if isinstance(data, list) and len(data) > 0:
        #     # Most generation models return a list with {"generated_text": "..."}
        #     answer = data[0].get("generated_text") or data[0].get("summary_text")
        # elif isinstance(data, dict):
        #     # Sometimes errors or different structures come as dict
        #     if "generated_text" in data:
        #         answer = data["generated_text"]
        #     elif "summary_text" in data:
        #         answer = data["summary_text"]
        #     elif "error" in data:
        #         raise HTTPException(status_code=500, detail=f"HuggingFace API error: {data['error']}")

        # if not answer:
        #     raise HTTPException(status_code=500, detail=f"Unexpected HF response: {data}")

        # return {"answer": answer.strip(), "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")

@app.post("/ask")
async def ask(req: QueryRequest):
    """Ask question with RAG"""
    """Local Model"""
    try:
        if not client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        # 1. Split question
        words = split_question(req.query)

        # 2. Retrieve relevant chunks from Qdrant
        relevant_chunks = retrieve_relevant_chunks(req.query, top_k=5)

        if not relevant_chunks:
            return {"answer": "No relevant documents found. Please train the system first."}

        # 3. Summarize results
        answer = summarize_chunks(req.query, relevant_chunks)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")

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

    # if not initialize_embedding_model():
    #     print("WARNING: Embedding model initialization failed")
    
    # # Create collection if needed
    # if client:
    #     get_or_create_collection()
    
    print("RAG application startup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)