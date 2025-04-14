import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
import asyncio
from asyncio import Queue, TimeoutError as AsyncTimeoutError
import uuid
from typing import Dict, List, Optional, AsyncGenerator
import time
import concurrent.futures
from contextlib import asynccontextmanager

# --- Configuration ---
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
LLM_MODEL_NAME = "facebook/opt-125m"
MAX_BATCH_SIZE = 128
MAX_WAITING_TIME = 1

MAX_NEW_TOKENS = 50
DO_SAMPLE = True
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.2
USE_ANN = True
USE_DOCS = True


# --- Global State ---
class AppState:
    """Global state for the FastAPI app."""

    def __init__(self):
        self.embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        self.embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_model.to(self.device)
        if self.device == "cuda":
            try:
                torch.cuda.set_device(0)
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(
                    f"Error setting GPU device: {e}. Falling back to default.")
                self.device = "cpu"
                self.llm_model.to(self.device)
        else:
            print("Using CPU")

        self.documents = None

        if USE_DOCS:
            import pandas as pd
            import os
            import requests

            url = "https://huggingface.co/datasets/enelpol/rag-mini-bioasq/resolve/main/question-answer-passages/train-00000-of-00001.parquet"
            filename = "train-00000-of-00001.parquet"

            # Download if needed
            if not os.path.exists(filename):
                print("Downloading parquet file...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print("Download complete.")
            else:
                print("File already exists.")

            # Load parquet
            df = pd.read_parquet(filename, engine="pyarrow")

            # Just get the passage column as a list of strings
            self.documents = df["answer"].tolist()
            print(len(self.documents))

        else:
            self.documents = [
                "Cats are small furry carnivores that are often kept as pets.",
                "Dogs are domesticated mammals, not natural wild animals.",
                "Hummingbirds can hover in mid-air by rapidly flapping their wings."
            ]
        self.embed_model.to(self.device)

        if USE_DOCS:
            if os.path.exists("doc_embeddings.pt"):
                self.doc_embeddings = torch.load(
                    "doc_embeddings.pt").to(self.device)
            else:
                self.doc_embeddings = self._precompute_doc_embeddings()
                torch.save(self.doc_embeddings.cpu(), "doc_embeddings.pt")
        else:
            self.doc_embeddings = self._precompute_doc_embeddings()

        self.request_queue: Queue[QueryRequest] = asyncio.Queue()
        self.responses: Dict[str, asyncio.Queue[str]] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor()
        # Store task handle here
        self.background_task: Optional[asyncio.Task] = None

    def _blocking_get_embedding(self, texts: list) -> torch.Tensor:
        """
        Internal blocking embedding function.

        :param texts: List of strings to embed.
        :returns: Tensor of embeddings.
        """
        inputs = self.embed_tokenizer(texts, return_tensors="pt",
                                      truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu()

    async def batched_get_embedding(self, texts: list) -> torch.Tensor:
        """
        Runs blocking embedding in executor.

        :param texts: List of strings to embed.
        :returns: Tensor of embeddings.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._blocking_get_embedding, texts)

    def _precompute_doc_embeddings(self) -> torch.Tensor:
        """
        Precompute document embeddings synchronously at startup.

        :returns: Tensor of document embeddings.
        """
        print("Precomputing document embeddings...")
        all_embeddings = []
        doc_batch_size = 32
        for i in range(0, len(self.documents), doc_batch_size):
            batch_docs = self.documents[i:i+doc_batch_size]
            batch_embeddings = self._blocking_get_embedding(batch_docs)
            all_embeddings.append(batch_embeddings)
        print("Document embeddings computed.")
        return torch.vstack(all_embeddings).to(self.device)

    async def batched_retrieve_top_k(self, query_embeddings: torch.Tensor, k: int = 2) -> List[List[str]]:
        """
        Find top-k documents based on embeddings. Assumes embeddings are tensors.

        :param query_embeddings: Tensor of query embeddings.
        :param k: Number of top documents to retrieve.
        :returns: List of lists of top-k documents.
        """
        query_embeddings_tensor = query_embeddings.to(self.device)
        actual_k = min(k, self.doc_embeddings.shape[0])
        sims = torch.matmul(query_embeddings_tensor, self.doc_embeddings.T)
        indices = None
        if USE_ANN:
            print("Using ANN")
            from ann import ann
            indices, _ = ann(query_embeddings_tensor,
                             self.doc_embeddings,
                             k=actual_k)
        _, indices = torch.topk(sims, actual_k, dim=1)
        indices_np = indices.cpu().numpy()
        documents_np = np.array(self.documents)
        selected_docs_batch = [documents_np[indices_np[i]
                                            ].tolist() for i in range(indices_np.shape[0])]
        return selected_docs_batch

    def _blocking_generate_text_batch(self, prompts: List[str]) -> List[str]:
        """
        Internal blocking LLM generation function.

        :param prompts: List of strings to generate text for.
        :returns: List of generated texts.
        """
        start_time = time.monotonic()
        inputs = self.llm_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        input_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.llm_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, top_k=TOP_K,
                top_p=TOP_P, temperature=TEMPERATURE, repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id
            )
        generated_ids = output_ids[:, input_length:]
        generated_texts = self.llm_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        end_time = time.monotonic()
        print(
            f"LLM generation took {end_time - start_time:.4f} seconds for {len(prompts)} prompts.")
        return generated_texts

    async def generate_text_batch(self, prompts: List[str]) -> List[str]:
        """
        Runs blocking LLM generation in executor.

        :param prompts: List of strings to generate text for.
        :returns: List of generated texts.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._blocking_generate_text_batch, prompts)

    async def batched_rag_pipeline(self, queries: List[str], k: int) -> List[str]:
        """
        Async RAG pipeline using executor for blocking parts.

        :param queries: List of queries to process.
        :param k: Number of top documents to retrieve.
        :returns: List of generated texts.
        """
        print(f"RAG Pipeline: Processing batch of {len(queries)} queries...")
        pipeline_start_time = time.monotonic()
        query_emb = await self.batched_get_embedding(queries)
        embed_time = time.monotonic()
        print(
            f"  Embeddings computed in {embed_time - pipeline_start_time:.4f}s")
        retrieved_docs_batch = await self.batched_retrieve_top_k(query_emb, k)
        retrieve_time = time.monotonic()
        print(f"  Docs retrieved in {retrieve_time - embed_time:.4f}s")
        prompts = []
        for query, retrieved_docs in zip(queries, retrieved_docs_batch):
            context = "\n".join(retrieved_docs)
            prompts.append(f"Question: {query}\nContext:\n{context}\nAnswer:")
        format_time = time.monotonic()
        print(f"  Prompts formatted in {format_time - retrieve_time:.4f}s")
        generated_texts = await self.generate_text_batch(prompts)
        generate_end_time = time.monotonic()
        print(
            f"LLM Generation finished in {generate_end_time - format_time:.4f}s")
        print(
            f"RAG Pipeline: Batch total time {generate_end_time - pipeline_start_time:.4f}s")
        return generated_texts


# Define request model
class QueryRequest(BaseModel):
    """Model for incoming query requests."""
    query: str
    k: int = 2
    _id: str = ""  # Will be populated internally


# --- Background Task ---
async def process_requests_loop(app_state: AppState):
    """
    The main loop processing requests from the queue.
    It waits for the first item indefinitely, then tries to fill the batch
    within the specified waiting time.

    :param app_state: The application state containing the request queue and models.
    :returns: None
    """
    print("Background processor started.")
    while True:
        batch: List[QueryRequest] = []
        first_item_received_time: Optional[float] = None
        try:
            # 1. Wait for the first item indefinitely
            print(
                f"Waiting for first item... Queue size: {app_state.request_queue.qsize()}")
            first_item = await app_state.request_queue.get()
            app_state.request_queue.task_done()
            batch.append(first_item)
            first_item_received_time = time.monotonic()  # Use monotonic clock
            print(f"Got first item: {first_item._id}")

            # 2. Try to fill the rest of the batch
            while len(batch) < MAX_BATCH_SIZE:
                time_since_first = time.monotonic() - first_item_received_time
                remaining_wait_time = MAX_WAITING_TIME - time_since_first
                if remaining_wait_time <= 0:
                    print("Max waiting time reached after first item.")
                    break
                try:
                    next_item = await asyncio.wait_for(
                        app_state.request_queue.get(), timeout=remaining_wait_time
                    )
                    app_state.request_queue.task_done()
                    batch.append(next_item)
                except AsyncTimeoutError:
                    print("Timeout waiting for subsequent items.")
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"Error getting subsequent item: {e}")
                    await asyncio.sleep(0.01)
                    break
        except asyncio.CancelledError:
            print("Processor loop cancelled.")
            break
        except Exception as e:
            print(f"Error getting first item or during batch fill: {e}")
            await asyncio.sleep(0.1)
            continue

        # 3. Process the batch if it's not empty
        if batch:
            print(f"Processing batch of size {len(batch)}...")
            batch_process_start_time = time.monotonic()
            try:
                k_for_batch = batch[0].k
                results = await app_state.batched_rag_pipeline(
                    queries=[req.query for req in batch], k=k_for_batch
                )
                for i, req in enumerate(batch):
                    response_queue = app_state.responses.get(req._id)
                    if response_queue:
                        try:
                            await response_queue.put(results[i])
                        except Exception as put_e:
                            print(
                                f"Error putting result for {req._id}: {put_e}")
                    else:
                        print(
                            f"Warning: Response queue not found for request {req._id}")
                batch_process_end_time = time.monotonic()
                print(
                    f"Batch processed in {batch_process_end_time - batch_process_start_time:.4f}s")
            except Exception as batch_e:
                print(f"Error processing batch: {batch_e}")
                error_message = f"Error processing batch: {batch_e}"
                for req in batch:
                    response_queue = app_state.responses.get(req._id)
                    if response_queue:
                        try:
                            await response_queue.put(error_message)
                        except Exception as notify_e:
                            print(
                                f"Error notifying client {req._id} about batch failure: {notify_e}")
        else:
            print("Processor loop yielded an empty batch.")
            await asyncio.sleep(0.01)


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles application startup and shutdown.

    :param app: The FastAPI application instance.
    :returns: An async generator.
    """
    print("Application startup: Initializing resources...")
    # Initialize state and store it in app.state
    app_state = AppState()
    app.state.app_state = app_state
    # Start the background task
    app_state.background_task = asyncio.create_task(
        process_requests_loop(app_state))
    print("Background processor task created.")

    yield  # Application runs here

    # --- Shutdown Logic ---
    print("Application shutdown: Cleaning up resources...")
    if app_state.background_task and not app_state.background_task.done():
        print("Cancelling background task...")
        app_state.background_task.cancel()
        try:
            await app_state.background_task
        except asyncio.CancelledError:
            print("Background processor task successfully cancelled.")
        except Exception as e:
            print(f"Exception during background task cancellation wait: {e}")

    print("Shutting down thread pool executor...")
    app_state.executor.shutdown(wait=True)
    print("Executor shut down.")

    if app_state.device == "cuda":
        print("Attempting to clear GPU memory...")
        try:
            del app_state.llm_model
            del app_state.embed_model
            del app_state.doc_embeddings
            torch.cuda.empty_cache()
            print("GPU memory cleared.")
        except Exception as e:
            print(f"Error clearing GPU memory: {e}")
    print("Application shutdown complete.")


# --- FastAPI App ---
# Pass the lifespan context manager to the FastAPI app
app = FastAPI(lifespan=lifespan)


# --- API Endpoint ---
@app.post("/rag")
async def predict(payload: QueryRequest, request: Request):
    """
    Handles incoming RAG requests asynchronously.

    :param payload: The request payload containing the query and other parameters.
    :param request: The FastAPI request object.
    """
    # Access state via request.app.state
    if not hasattr(request.app.state, 'app_state'):
        print("Error: App state not initialized!")
        return {"error": "Server not initialized correctly."}

    app_state: AppState = request.app.state.app_state
    payload._id = str(uuid.uuid4())
    response_q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    app_state.responses[payload._id] = response_q

    try:
        await app_state.request_queue.put(payload)
    except Exception:
        if payload._id in app_state.responses:
            del app_state.responses[payload._id]
        return {"error": "Failed to queue request"}

    try:
        response_timeout = MAX_WAITING_TIME + 60.0
        response = await asyncio.wait_for(response_q.get(), timeout=response_timeout)
        response_q.task_done()
        return {"response": response}
    except AsyncTimeoutError:
        return {"error": "Processing timeout"}
    except Exception:
        return {"error": "An internal error occurred"}
    finally:
        if payload._id in app_state.responses:
            del app_state.responses[payload._id]


# ----- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
