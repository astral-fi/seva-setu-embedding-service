import os
# --- KEY FIX: Set the cache directory BEFORE importing transformers ---
# This tells the library to use a writable directory inside the project.
os.environ['TRANSFORMERS_CACHE'] = '/code/cache'

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'krutrim-ai-labs/vyakyarth'

# --- Helper Function for Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# --- FastAPI App Initialization ---
app = FastAPI(title="Embedding Service")

# --- Load Model on Startup ---
# This dictionary will hold the loaded model and tokenizer
model_payload = {}

@app.on_event("startup")
def load_model():
    """Load the model and tokenizer when the server starts."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model_payload['tokenizer'] = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model_payload['model'] = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    model_payload['model'].eval()
    print("Model loaded successfully.")

# --- Pydantic Models for Request/Response ---
class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

# --- API Endpoint ---
@app.post("/embed", response_model=EmbeddingResponse)
def create_embedding(request: EmbeddingRequest):
    """Takes text and returns its vector embedding."""
    tokenizer = model_payload['tokenizer']
    model = model_payload['model']

    encoded_input = tokenizer(request.text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    embedding = mean_pooling(model_output, encoded_input['attention_mask']).tolist()[0]
    
    return {"embedding": embedding}

@app.get("/")
def read_root():
    return {"message": "Embedding Service is running. Use the /embed endpoint."}
