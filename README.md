# PDF RAG Chatbot

> A practical implementation of Retrieval-Augmented Generation (RAG) that enables PDF document querying using Google's Gemini API. This project demonstrates how RAG systems can overcome the limitations of traditional language models by grounding responses in source documents.

![Demo Screenshot](frontend-preview.png)

## 📋 Contents

- [Overview](#overview)
- [Key Features](#features)
- [Technical Architecture](#technical-architecture)
- [Quick Start](#installation--setup)
- [Usage Guide](#how-to-use)
- [Development](#development--experimentation)
- [API Reference](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Future Roadmap](#future-enhancements)
- [Support](#support)

## Overview

### Why RAG Matters

Traditional language models (LLMs) are prone to hallucinations and struggle with domain-specific knowledge. This project demonstrates how Retrieval-Augmented Generation (RAG) addresses these limitations by:

1. **Reducing Hallucinations**: Responses are grounded in document content rather than model parameters
2. **Enabling Factual Accuracy**: The system retrieves specific sections from source documents before generating answers
3. **Facilitating Knowledge Updates**: New information can be added via documents without retraining the entire model
4. **Providing Source Attribution**: Responses can reference specific document sections, improving transparency

### Technical Implementation

This implementation follows RAG best practices with a hybrid retrieval approach:

1. **Document Processing**: PDFs are semantically chunked with consideration for document structure
2. **Dual-Stage Retrieval**: Combines dense vector search (FAISS) with sparse retrieval (BM25) for higher precision
3. **Context-Aware Generation**: Uses retrieved passages to provide the LLM with relevant context before generating responses
4. **Practical Web Interface**: Demonstrates the system with a React frontend and FastAPI backend

The project serves as both a functional tool and a demonstration of applied NLP techniques in a real-world scenario.

## Project Structure

This project consists of four main components:

1. **React Frontend** (`/frontend`): A modern, responsive web interface for interacting with the chatbot
2. **FastAPI Backend** (`/backend/advanced-model`): A high-performance API that handles PDF processing and query answering
3. **RAG Engine**: Advanced retrieval-augmented generation system integrated with the backend
4. **Development Notebook** (`Jupyter-Notebook.ipynb`): For experimentation and fine-tuning of RAG parameters

## Features

- **Smart PDF Processing:** 
  - Extracts text while preserving document structure (tables, headers, lists)
  - Uses advanced NLP for better text understanding
  - Maintains document formatting and layout

- **Intelligent Question Answering:**
  - Context-aware responses based on document content
  - Accurate information retrieval using hybrid search
  - Natural conversation flow

- **Technology Stack:**
  - FAISS HNSW for fast vector search
  - BM25 for keyword-based filtering
  - SentenceTransformer for text embeddings
  - Google's Gemini Flash for response generation
  - React for the frontend interface
  - FastAPI for the backend API

### Document Processing Pipeline

1. **PDF Text Extraction**
   - Primary: Uses Unstructured's `partition_pdf` for intelligent document parsing
   - Fallback: Uses PyMuPDF for raw text extraction when structured parsing fails

2. **Section Tagging**
   - Applies regex-based section identification for technical documents
   - Organizes content into semantic sections (Abstract, Introduction, etc.)

3. **Adaptive Chunking**
   - Dynamically adjusts chunk size based on document content
   - Uses sentence boundaries and transition words to create meaningful segments

### Vector Search Implementation

1. **Embedding Generation**
   - Uses SentenceTransformer's "all-MiniLM-L6-v2" model for text embeddings

2. **Hybrid Search**
   - FAISS HNSW index for efficient vector similarity search
   - BM25 re-ranking for precise keyword matching
   - Combined approach improves retrieval accuracy

3. **Response Generation**
   - Gemini Flash 1.5 model for fast, context-aware answer generation
   - Carefully crafted prompting to ensure answers are grounded in the document

### Performance Optimization

- HNSW index for sub-linear time complexity in vector search
- Caching mechanisms to reduce repeated processing
- Efficient memory management for large documents

## Prerequisites

- Python 3.8 or higher
- Node.js 14+ and npm
- Google Gemini API key (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python -m venv gemini-venv
# On Windows:
gemini-venv\Scripts\activate
# On macOS/Linux:
source gemini-venv/bin/activate

# Install dependencies
pip install -r gemini_requirements.txt

# Set up environment variables
# Option 1: Create .env files manually
# Create a .env file in the root directory with:
# GEMINI_API_KEY=your_gemini_api_key

# Create a .env file in backend/advanced-model with:
# GEMINI_API_KEY=your_gemini_api_key

# Option 2: Use these commands to create the .env files
echo "GEMINI_API_KEY=your_gemini_api_key" > .env
echo "GEMINI_API_KEY=your_gemini_api_key" > backend/advanced-model/.env
```

### 3. Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Return to the root directory
cd ..
```

## Running the Application

### 1. Start the Backend Server

```bash
# Make sure your virtual environment is activated
# Navigate to the backend directory
cd backend/advanced-model

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend Development Server

In a new terminal:

```bash
# Navigate to the frontend directory
cd frontend

# Start the React development server
npm start
```

The application should now be running at [http://localhost:3000](http://localhost:3000)

## How to Use

### Document Upload Process

1. **Upload Your PDF**
   - Click the "Upload" button in the top navigation bar
   - Select any PDF file (or Drag and Drop)

### Asking Questions

2. **Interact with the Document**
   - Type your questions in the chat interface
   - Examples of effective questions:
     - "What are the key findings in the conclusion section?"
     - "Summarize the methodology used in this document"
     - "What evidence supports the main argument?"
   - The system will search through the document and provide relevant answers

### Understanding Responses

3. **Interpret Results**
   - Get accurate answers based on your document's content
   - See the relevant sections from your PDF that support the answers
   - The system indicates which parts of the document were used to generate the response

## Development & Experimentation

### Configuration Options

The system can be configured through environment variables:

| Variable | Purpose | Default | 
|----------|---------|---------|
| CHUNK_SIZE | Controls text segmentation | 512 |
| EMBEDDING_MODEL | SentenceTransformer model | all-MiniLM-L6-v2 |

### Jupyter Notebook

The project includes a Jupyter notebook (`Jupyter-Notebook.ipynb`) specifically designed for:

- **RAG Model Tuning**: Easily modify parameters like chunk size, embedding model, or vector search configurations
- **Rapid Prototyping**: Test new retrieval approaches or prompting strategies without modifying the production backend
- **Performance Analysis**: Benchmark different configurations to optimize retrieval accuracy and speed
- **Iterative Improvement**: Make incremental changes to the RAG pipeline and immediately see their impact

To use the notebook for development:

```bash
# Activate the virtual environment
gemini-venv\Scripts\activate  # Windows
source gemini-venv/bin/activate  # macOS/Linux

# Start Jupyter
jupyter notebook

# Open Jupyter-Notebook.ipynb
```

## Deployment

### Docker Deployment (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Manual Production Deployment

For production environments:

1. Set up a reverse proxy (Nginx/Apache) in front of the FastAPI server
2. Configure proper SSL/TLS
3. Use a process manager like Gunicorn for the backend
4. Build a production version of the React app:
   ```bash
   cd frontend
   npm run build
   ```

## API Endpoints

The backend FastAPI server provides the following endpoints:

| Endpoint | Method | Description | Example Request |
|----------|--------|-------------|----------------|
| `/upload` | POST | Upload a PDF file | `curl -F "file=@document.pdf" http://localhost:8000/upload` |
| `/query` | POST | Query the uploaded document | `curl -X POST -H "Content-Type: application/json" -d '{"query":"What is the main conclusion?"}' http://localhost:8000/query` |
| `/delete-pdf` | DELETE | Delete the currently loaded PDF | `curl -X DELETE http://localhost:8000/delete-pdf` |
| `/` | GET | Health check endpoint | `curl http://localhost:8000/` |

## Security Considerations

- The application does not currently implement user authentication
- PDF documents are stored temporarily on the server
- API keys are stored in environment variables, not in the code
- Implement proper access controls in production environments

## Troubleshooting

### Common Issues

1. **PDF Upload Fails**
   - Ensure the PDF is not password-protected
   - Try reducing the file size if it's very large (>20MB)
   - Check the file isn't corrupted by opening it in another PDF reader

2. **Backend Connection Error**
   - Verify that the backend server is running
   - Check that the frontend is configured to connect to the correct backend URL
   - Ensure port 8000 isn't blocked by a firewall

3. **Empty or Irrelevant Responses**
   - The PDF might contain non-standard formatting
   - Try adjusting chunk size in the backend configuration
   - Some PDFs with heavy image content may not process correctly

## Performance Considerations

- Processing large PDFs (>100 pages) may take considerable time and memory
- The application is optimized for technical and academic documents
- Consider running on a machine with at least 4GB RAM for optimal performance

## Future Enhancements

- Multi-document querying support
- Domain-specific embedding optimizations
- Enhanced contextual retrieval
- Real-time document processing
- User authentication and document management
- Export conversation history
- Mobile application support

## Support

For issues or questions, please open an issue in the repository or contact the maintainer.

## License

MIT License 

