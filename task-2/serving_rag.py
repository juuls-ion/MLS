import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from threading import Thread
from queue import Queue
import uuid
from typing import Dict, List
import time

# Detect whether GPU is available, default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.set_device(0)
    print("Using GPU")
else:
    print("Using CPU")

# -----Preamble-----
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# -----Embeddings-----

# Load embedding model, tokenizer
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Generate embeddings using embedding model
def batched_get_embedding(texts: list) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(texts, return_tensors="pt",
                             truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Precompute document embeddings
doc_embeddings = np.vstack([batched_get_embedding([doc]) for doc in documents])

# Find top-k embeddings
# We can use our own method from task 1
def batched_retrieve_top_k(query_embeddings: np.ndarray, k: int = 2) -> List[List[str]]:
    doc_embeddings_tensor = torch.from_numpy(doc_embeddings)
    query_embeddings_tensor = torch.from_numpy(query_embeddings)

    sims = doc_embeddings_tensor.unsqueeze(0) @ query_embeddings_tensor.T
    _, indices = torch.topk(sims, k, dim=1)
    indices = indices.cpu().numpy()
    indices = indices.squeeze()
    documents_np = np.array(documents)
    selected_documents = documents_np[indices]
    print(selected_documents.shape)
    return selected_documents.T


# -----LLM-----
# chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")
# Qwen model: "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def generate_text_batch(
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
) -> List[str]:
    """
    Generates text responses for a batch of prompts.
    """
    start_time = time.time()
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        # Optional: set a max_length for tokenization if prompts can be very long
        # max_length=512
    )
    input_length = inputs["input_ids"].shape[1]

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[:, input_length:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    end_time = time.time()
    print(
        f"Batch generation took {end_time - start_time:.2f} seconds for {len(prompts)} prompts."
    )

    return generated_texts


# -----Output Formatting-----
# -----RAG Stuff-----
app = FastAPI()


def batched_rag_pipeline(queries: List[str], k: int = 2) -> List[str]:
    query_emb = batched_get_embedding(queries)
    retrieved_docs = batched_retrieve_top_k(query_emb, k)
    prompt = [
        f"Question: {query}\nContext:\n{context}\nAnswer:"
        for query, context in zip(queries, retrieved_docs)
    ]
    generated = generate_text_batch(prompt, a, b, c, d, e, f)
    return generated


# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    _id: str = ""


# Request Queue
request_queue: Queue[QueryRequest] = Queue()
responses: Dict[str, Queue[str]] = {}
MAX_BATCH_SIZE = 20
MAX_WAITING_TIME = 2


def process_requests():
    batch = []
    start = time.time()
    while True:
        if batch:
            if len(batch) == MAX_BATCH_SIZE or time.time() - start > MAX_WAITING_TIME:
                results = batched_rag_pipeline([req.query for req in batch])
                for i, req in enumerate(batch):
                    responses[req._id].put(results[i])
                batch = []
                start = time.time()
        try:
            # Wait for a request or timeout
            batch.append(request_queue.get(timeout=MAX_WAITING_TIME))
        except Exception:
            pass


# Start the background thread to process requests
Thread(target=process_requests).start()


@app.post("/rag")
def predict(payload: QueryRequest):
    # Create a unique ID and a response queue for this request.
    # This queue will only hold one item, but allows us to wait for it later.
    payload._id = str(uuid.uuid4())
    responses[payload._id] = Queue()

    # Put the request in the queue for processing.
    request_queue.put(payload)

    # Wait for the response, then delete the response queue and
    # return the response.
    response = responses[payload._id].get()
    del responses[payload._id]
    return response


# -----Hints-----
"""
### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests
"""

# -----Parameters-----
a = 200   # Max Length
b = True  # do_sample
c = 50    # top-k
d = 0.95  # top-p
e = 0.7   # temperature
f = 1.2   # repetition-penalty

# -------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
