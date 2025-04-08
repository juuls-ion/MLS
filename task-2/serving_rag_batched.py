import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import re

device = torch.device("cuda")

# -----Preamble-----
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings.",
]

# -----Embeddings-----

# Load embedding model, tokenizer
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
embed_model.to(device)


def batched_get_embedding(texts: list) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Generate embeddings using embedding model
def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Precompute document embeddings
# doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])


# Find top-k embeddings
# We can use our own method from task 1
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]


# -----LLM-----
# chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")

# Qwen model: "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

from typing import List
import time


device = "cpu"


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


def generate_text(
    prompt, max_length, do_sample, top_k, top_p, temperature, repetition_penalty
):
    inputs = tokenizer(prompt, return_tensors="pt")

    # When using GPU, move inputs to CUDA:
    # inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate output using LLM
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# -----Output Formatting-----


# -----RAG Stuff-----
app = FastAPI()


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


def batched_rag_pipeline(queries: List[str], k: int = 2) -> List[str]:
    query_emb = batched_get_embedding(queries)
    retrieved_docs = batched_retrieve_top_k(query_emb, k)
    prompt = [
        f"Question: {query}\nContext:\n{context}\nAnswer:"
        for query, context in zip(queries, retrieved_docs)
    ]
    generated = generate_text_batch(prompt, a, b, c, d, e, f)
    return generated


def rag_pipeline(query: str, k: int = 2) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding(query)

    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)

    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"

    # Step 3: LLM Output
    # generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    generated = generate_text(
        prompt,
        max_length=a,
        do_sample=b,
        top_k=c,
        top_p=d,
        temperature=e,
        repetition_penalty=f,
    )
    return generated


# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2


@app.post("/rag")
def predict(payload: QueryRequest):
    result = rag_pipeline(payload.query, payload.k)
    formatted = format_rag_output(result)

    return {result}


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
# Max Length
a = 200
# do_sample
b = True
# top-k
c = 50
# top-p
d = 0.95
# temperature
e = 0.7
# repetition-penalty
f = 1.2

# -------------------------------------------------

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    # print(rag_pipeline("Cats are", 2))
    prompt = [
        "What's the meaning of life?",
        "Dogs are",
        "Mice are",
        "Humming birds are",
    ]
    # print(batched_rag_pipeline(prompt, 2))
    print(generate_text_batch(prompt, a, b, c, d, e, f))
