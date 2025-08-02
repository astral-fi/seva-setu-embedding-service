import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List

# --- Use a writable directory for the model cache ---
# This is crucial for environments like Render where the default directory isn't writable.
os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache'

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'krutrim-ai-labs/vyakyarth'

# --- Helper Function for Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling to get a single vector embedding for the entire sequence.
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# --- FastAPI App Initialization ---
app = FastAPI(title="Embedding Service")

# This dictionary will hold the loaded model and tokenizer
model_payload = {}

@app.on_event("startup")
def load_model():
    """Load the model and tokenizer when the server starts."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model_payload['tokenizer'] = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model_payload['model'] = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        model_payload['model'].eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # A more robust solution might include retries or logging to a dedicated service.

# --- Pydantic Models for Request/Response ---
class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

# --- API Endpoints ---
@app.post("/embed", response_model=EmbeddingResponse)
def create_embedding(request: EmbeddingRequest):
    """Takes text and returns its vector embedding."""
    if 'model' not in model_payload or 'tokenizer' not in model_payload:
        raise HTTPException(status_code=503, detail="Model is not available or is still loading.")
        
    tokenizer = model_payload['tokenizer']
    model = model_payload['model']

    # Tokenize the input text
    encoded_input = tokenizer(request.text, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    embedding = mean_pooling(model_output, encoded_input['attention_mask']).tolist()[0]
    
    # Return the embedding in the specified response format
    return EmbeddingResponse(embedding=embedding)

@app.get("/")
def read_root():
    """Root endpoint to check if the service is running."""
    return {"message": "Embedding Service is running. Use the /embed endpoint to get embeddings."}
