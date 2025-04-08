import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import re
from threading import Thread
from queue import Queue
import uuid
from typing import Dict, List
import time

#-----Preamble-----
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

#-----Embeddings-----

# Load embedding model, tokenizer
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Generate embeddings using embedding model
def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

# Find top-k embeddings
# We can use our own method from task 1
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]


#-----LLM-----
#chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")

# Qwen model: "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_text(prompt,
                  max_length,
                  do_sample,
                  top_k,
                  top_p,
                  temperature,
                  repetition_penalty):
    inputs = tokenizer(prompt, return_tensors="pt")

    # When using GPU, move inputs to CUDA:
    #inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate output using LLM
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


#-----Output Formatting-----


#-----RAG Stuff-----
app = FastAPI()

def rag_pipeline(query: str, k: int = 2) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    #generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    generated = generate_text(
        prompt,
        max_length=a,
        do_sample=b,
        top_k=c,
        top_p=d,
        temperature=e,
        repetition_penalty=f
    )
    return generated

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2
    _id: str = None

# Request Queue
request_queues: List[Queue[QueryRequest]] = [Queue() for _ in range(5)]
responses: Dict[str, Queue[str]] = {}
MAX_BATCH_SIZE = 5
MAX_WAITING_TIME = 0.1

# Background thread to process requests
class RequestQueue:
    queue_index: int
    model: int

    def __init__(self, queue_index: int):
        self.queue_index = queue_index
        Thread(target=self.process_requests).start()

    def process_requests(self):
        print(self.queue_index, request_queues[self.queue_index])
        while True:
            batch = []
            for _ in range(MAX_BATCH_SIZE):
                try:
                    # Wait for a request or timeout
                    batch.append(request_queues[self.queue_index].get(timeout=MAX_WAITING_TIME))
                except Exception as e:
                    break

            if batch:
                # Process the batch of requests
                for payload in batch:
                    result = rag_pipeline(payload.query, payload.k)
                    responses.get(payload._id).put(result)

# Initialize request queues
for i in range(len(request_queues)):
    RequestQueue(i)

@app.post("/rag")
def predict(payload: QueryRequest):
    print("Received request")
    payload._id = str(uuid.uuid4())
    responses[payload._id] = Queue()
    min(request_queues, key=lambda q: q.qsize()).put(payload)
    response = responses[payload._id].get()
    del responses[payload._id]
    return response



#-----Hints-----
'''
### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests
'''
#-----Parameters-----
# Max Length
a = 200
# do_sample
b = True
# top-k
c = 50
# top-p
d = 0.95
#temperature
e = 0.7
# repetition-penalty
f = 1.2

#-------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
