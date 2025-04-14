import os
import shutil
import time
import re
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import faiss
# import fitz  # PyMuPDF
import pymupdf  # Use pymupdf directly instead of fitz
import spacy
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from unstructured.partition.pdf import partition_pdf
from rank_bm25 import BM25Okapi

# Load environment variables and set Gemini API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Initialize spaCy and the Sentence Transformer model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure logging
logging.basicConfig(
    filename="gemini_benchmark.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Enable CORS for the React frontend (adjust origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global storage for processed PDF data
pdf_data = {
    "filename": None,
    "raw_text": None,
    "tagged_sections": None,
    "chunks": None,
    "index": None,
}

# ----------------------------
# Advanced Model Functions
# ----------------------------

def extract_text_from_pdf(file_path: str) -> str:
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_structured_content(file_path: str):
    elements = partition_pdf(filename=file_path)
    structured_data = []
    for element in elements:
        structured_data.append({
            "type": element.type,
            "text": element.text.strip(),
        })
    return structured_data

def tag_sections_technical(structured_elements):
    section_pattern = re.compile(
        r"(Abstract|Introduction|Related Work|Background|Methodology|Approach|Experiments|Results|Discussion|Conclusion|Encoding|CLIP|Text Encoder|Embedding)",
        re.IGNORECASE
    )
    tagged_sections = {}
    current_section = None
    for element in structured_elements:
        element_type = element.get("type", "").lower()
        text = element.get("text", "")
        if element_type in ["heading", "title"] or section_pattern.search(text):
            match = section_pattern.search(text)
            new_section = match.group(0).strip() if match else text.strip()
            current_section = new_section
            if current_section not in tagged_sections:
                tagged_sections[current_section] = []
        elif current_section:
            tagged_sections[current_section].append(text)
        else:
            tagged_sections.setdefault("Body", []).append(text)
    for section in tagged_sections:
        tagged_sections[section] = "\n".join(tagged_sections[section]).strip()
    return tagged_sections

def robust_extract_text(file_path: str):
    try:
        structured_elements = extract_structured_content(file_path)
        tagged_sections = tag_sections_technical(structured_elements)
        combined_text = "\n\n".join([f"{section}: {content}" for section, content in tagged_sections.items()])
        if combined_text.strip():
            return combined_text, tagged_sections
        else:
            raise Exception("No structured content extracted.")
    except Exception as e:
        logging.info("Structured extraction failed; using fallback extraction. Error: " + str(e))
        fallback_text = extract_text_from_pdf(file_path)
        return fallback_text, {}

def adaptive_chunk_text_dynamic(text: str, min_threshold: int = None, factor: float = 1.5, transition_words=None):
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    token_counts = [len(sent.split()) for sent in sentences]
    if not token_counts:
        return [text]
    avg_tokens = sum(token_counts) / len(token_counts)
    if min_threshold is None:
        min_threshold = int(avg_tokens)
    threshold = int(max(min_threshold, avg_tokens * factor))
    if transition_words is None:
        transition_words = ["however", "moreover", "furthermore", "in conclusion", "finally", "additionally"]
    chunks = []
    current_chunk = ""
    current_token_count = 0
    for sent in sentences:
        sent_tokens = len(sent.split())
        if current_chunk:
            starts_with_transition = any(sent.lower().startswith(word) for word in transition_words)
        else:
            starts_with_transition = False
        if (current_token_count + sent_tokens > threshold) or (starts_with_transition and current_token_count > int(threshold * 0.7)):
            chunks.append(current_chunk.strip())
            current_chunk = sent
            current_token_count = sent_tokens
        else:
            current_chunk += " " + sent
            current_token_count += sent_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(chunks: list) -> np.ndarray:
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    return embeddings.astype("float32")

def build_hnsw_index(embeddings: np.ndarray, M: int = 32, efConstruction: int = 40):
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.add(embeddings)
    return index

def search_index(query: str, index, chunks: list, k: int = 5) -> list:
    start_time = time.time()
    query_embedding = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    search_duration = time.time() - start_time
    logging.info(f"HNSW Search Time: {search_duration:.4f} seconds")
    return results

def bm25_filter(query, candidate_chunks, threshold=1.0, top_n=2):
    """
    Use BM25 to select the top N most relevant chunks instead of just one.
    
    Args:
        query: User query
        candidate_chunks: List of candidate chunks from vector search
        threshold: Minimum score to consider
        top_n: Number of top chunks to return
        
    Returns:
        List of most relevant chunks
    """
    if not candidate_chunks:
        return []
    
    tokenized_corpus = [doc.lower().split() for doc in candidate_chunks if doc.strip()]
    if not tokenized_corpus:
        return []
    
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get indices sorted by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    # Select top N chunks that meet the threshold
    selected_chunks = []
    for idx in sorted_indices[:top_n]:
        if scores[idx] >= threshold and idx < len(candidate_chunks):
            selected_chunks.append(candidate_chunks[idx])
    
    return selected_chunks

def generate_answer_with_gemini(query: str, context: str) -> str:
    """Generates a well-reasoned answer using Gemini API with enhanced prompt formatting."""
    
    prompt = f"""
You are an advanced Retrieval-Augmented Generation (RAG) AI assistant specializing in legal document analysis. 
Your task is to accurately answer the given query based on the provided legal context.

**Context:**  
{context}

**Query:**  
{query}

**Instructions:**  
- Provide a **clear and direct answer first** (with a brief explanation).  
- Then, **explain your reasoning** in structured points.  
- If the correct context is not provided, **respond minimally** with:  
  - **"The provided context is insufficient to answer this question."**  
  - Do *not* provide unnecessary explanations or assumptions.  
- Ensure proper formatting with **clear paragraph separation**.  
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    return answer

def process_pdf_advanced(pdf_path):
    extracted_text, tagged_sections = robust_extract_text(pdf_path)
    chunks = adaptive_chunk_text_dynamic(extracted_text)
    embeddings = get_embeddings(chunks)
    faiss_index = build_hnsw_index(embeddings)
    return {
         "raw_text": extracted_text,
         "tagged_sections": tagged_sections,
         "chunks": chunks,
         "index": faiss_index,
    }

# ----------------------------
# Pydantic model for query requests
# ----------------------------
class QueryRequest(BaseModel):
    query: str
    model: str = "advanced"  # default to advanced; this is optional

# ----------------------------
# Endpoints
# ----------------------------

@app.post("/upload")
async def upload_pdf_route(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        advanced_data = process_pdf_advanced(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    pdf_data["filename"] = file_path
    pdf_data["raw_text"] = advanced_data["raw_text"]
    pdf_data["tagged_sections"] = advanced_data["tagged_sections"]
    pdf_data["chunks"] = advanced_data["chunks"]
    pdf_data["index"] = advanced_data["index"]
    return {"message": "PDF processed successfully.", "filename": filename}

@app.post("/query")
async def query_pdf(query_request: QueryRequest):
    start_time = time.time()
    
    if not pdf_data["chunks"] or not pdf_data["index"]:
        raise HTTPException(status_code=400, detail="No PDF uploaded or processed.")
    
    query = query_request.query
    
    # Log the incoming query
    logging.info(f"Query: {query}")
    
    # Search using hybrid approach (vector + keyword)
    candidate_chunks = search_index(query, pdf_data["index"], pdf_data["chunks"])
    
    logging.info("\nCandidate Chunks and Similarity Scores:")
    for i, chunk in enumerate(candidate_chunks):
        # Get vector similarity score
        query_embedding = embed_model.encode([query], convert_to_numpy=True).astype("float32")
        chunk_embedding = embed_model.encode([chunk], convert_to_numpy=True).astype("float32")
        similarity = np.dot(query_embedding[0], chunk_embedding[0]) / (np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk_embedding[0]))
        
        preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
        logging.info(f"Chunk {i+1}:\nSimilarity Score: {similarity}\nText (first 300 characters): {preview}\n")
    
    # Use BM25 to select the top 2 most relevant chunks instead of just one
    filtered_chunks = bm25_filter(query, candidate_chunks, threshold=0.5, top_n=2)
    
    if filtered_chunks:
        logging.info("BM25 Filtered Candidates Found.")
        # Combine the filtered chunks into a single context
        final_context = "\n\n".join(filtered_chunks)
    else:
        logging.info("No BM25 match found. Using top vector search result.")
        final_context = candidate_chunks[0] if candidate_chunks else ""
    
    logging.info(f"\nFinal Context for LLM Prompt:\n{final_context}\n")
    
    # Generate answer using the context
    answer = generate_answer_with_gemini(query, final_context)
    
    # Calculate reasoning time
    reasoning_time = round(time.time() - start_time, 2)
    
    return {"answer": answer, "reasoningTime": reasoning_time, "finalContext": final_context}

@app.delete("/delete-pdf")
async def delete_pdf_route():
    if not pdf_data["filename"]:
        raise HTTPException(status_code=400, detail="No PDF to delete.")
    
    try:
        # Delete the actual file
        if os.path.exists(pdf_data["filename"]):
            os.remove(pdf_data["filename"])
        
        # Reset the PDF data
        pdf_data["filename"] = None
        pdf_data["raw_text"] = None
        pdf_data["tagged_sections"] = None
        pdf_data["chunks"] = None
        pdf_data["index"] = None
        
        return {"message": "PDF deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def index_route():
    return "Gemini RAG Server is running!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True) 