import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from openai import OpenAI
from langchain.document_loaders import YoutubeLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from typing import List
from fastapi.responses import StreamingResponse

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API keys from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENROUTER_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys are not set in the .env file")

# Initialize HuggingFace Embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("rag")

# Initialize PineconeVectorStore
vectorstore = PineconeVectorStore(index_name="rag", embedding=hf_embeddings)

# Initialize tokenizer for text splitting
tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=60,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# Initialize a list to store temporary context
temporary_context = []

def get_rag_context(query: str) -> str:
    # Load the embedding model
    model = hf_embeddings.model

    # Create the embedding for the query
    query_embedding = model.encode(query)

    # Query the Pinecone index using the embedding
    top_matches = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=10,
        include_metadata=True,
        namespace="pdf-documents"
    )

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    # Create the augmented query with context
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Add temporary context
    if temporary_context:
        augmented_query += "\n\n<TEMPORARY_CONTEXT>\n" + "\n\n".join(temporary_context) + "\n</TEMPORARY_CONTEXT>"

    return augmented_query

def perform_rag(query: str) -> str:
    global temporary_context
    augmented_query = get_rag_context(query)

    system_prompt = """You are a highly knowledgeable customer support. Please provide clear, concise, and accurate answers to any questions I have about the company in the PDF provided
and never provide any information that is not in the PDF. and dont come across like youre getting the info from the pdf just answer the question directly, professionally and accurately
"""

    res = openrouter_client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = res.choices[0].message.content
    
    # Add the response to temporary context
    temporary_context.append(response)
    if len(temporary_context) > 5:  # Keep only the last 5 contexts
        temporary_context.pop(0)

    return response

async def stream_rag(query: str):
    global temporary_context
    augmented_query = get_rag_context(query)

    system_prompt = """You are a company customer support. Please provide clear, concise, and accurate answers to any questions I have about the company in the PDF provided
rule 1. never mention the pdf when replying, just aanswer professionally and accurately
"""

    stream = openrouter_client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ],
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content

    # Add the full response to temporary context
    temporary_context.append(full_response)
    if len(temporary_context) > 5:  # Keep only the last 5 contexts
        temporary_context.pop(0)

class Query(BaseModel):
    query: str

class YoutubeURL(BaseModel):
    youtube_url: str

@app.post("/rag")
async def rag_endpoint(query: Query):
    try:
        response = perform_rag(query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_rag")
async def stream_rag_endpoint(query: Query):
    try:
        return StreamingResponse(stream_rag(query.query), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_youtube")
async def ingest_youtube(youtube_url: YoutubeURL):
    try:
        # Load YouTube transcript
        loader = YoutubeLoader.from_youtube_url(youtube_url.youtube_url, add_video_info=True)
        data = loader.load()

        # Split the transcript into chunks
        texts = text_splitter.split_documents(data)

        # Insert chunks into Pinecone
        PineconeVectorStore.from_texts(
            [f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts],
            hf_embeddings,
            index_name="rag",
            namespace="youtube-videos-2"
        )

        return {"message": f"Successfully ingested transcript from {youtube_url.youtube_url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Load PDF
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        
        # Split the PDF content into chunks
        texts = text_splitter.split_documents(data)

        # Insert chunks into Pinecone
        PineconeVectorStore.from_texts(
            [f"Source: PDF - {file.filename}, Page: {t.metadata['page']} \n\nContent: {t.page_content}" for t in texts],
            hf_embeddings,
            index_name="rag",
            namespace="pdf-documents"
        )
        
        # Remove temporary file
        os.remove(temp_file_path)
        
        return {"message": f"Successfully ingested PDF: {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_text")
async def ingest_text(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Load text file
        loader = TextLoader(temp_file_path)
        data = loader.load()

        # Split the text content into chunks
        texts = text_splitter.split_documents(data)

        # Insert chunks into Pinecone
        PineconeVectorStore.from_texts(
            [f"Source: Text file - {file.filename} \n\nContent: {t.page_content}" for t in texts],
            hf_embeddings,
            index_name="rag",
            namespace="text-documents"
        )
        
        # Remove temporary file
        os.remove(temp_file_path)
        
        return {"message": f"Successfully ingested text file: {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

