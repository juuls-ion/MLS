# import time
# import httpx # Use httpx for async requests
# import asyncio
# import argparse
# from tqdm.asyncio import tqdm # Use tqdm's async version
# import statistics
# from typing import List, Optional, Dict, Any
# import numpy as np
# BATCHED_URL = "http://localhost:8000/rag"

# async def send_request_async(client: httpx.AsyncClient, url: str, query: str = "What are cats?", k: int = 2) -> Optional[float]:
#     """Sends a single asynchronous request and returns latency."""
#     start = time.monotonic() # Use monotonic clock for latency
#     try:
#         response = await client.post(url, json={"query": query, "k": k}, timeout=65.0) # Add timeout
#         latency = time.monotonic() - start
#         response.raise_for_status() # Raise exception for bad status codes (4xx, 5xx)
#         # You might want to check the response content too if needed
#         # data = response.json()
#         # if "error" in data: return None
#         return latency
#     except httpx.RequestError as exc:
#         print(f"\nAn error occurred while requesting {exc.request.url!r}: {exc}")
#         return None
#     except Exception as e:
#         print(f"\nAn unexpected error occurred: {e}")
#         return None


# async def measure_rate_limited_performance_async(url: str, num_requests: int, rate: float, label: str) -> Dict[str, Any]:
#     """Measures performance by sending requests concurrently with rate limiting."""
#     latencies: List[float] = []
#     tasks = []
#     delay = 1.0 / rate if rate > 0 else 0
#     successful_requests = 0
#     failed_requests = 0

#     # Use a single client session for connection pooling
#     async with httpx.AsyncClient() as client:
#         progress_bar = tqdm(total=num_requests, desc=label, ncols=100)
#         start_time = time.monotonic()

#         for i in range(num_requests):
#             # Create a task to send the request
#             task = asyncio.create_task(send_request_async(client, url))
#             tasks.append(task)
#             progress_bar.update(1)

#             # If rate limiting, sleep *between launching tasks*
#             if delay > 0:
#                 launch_time = time.monotonic()
#                 # Sleep until it's time to launch the next request
#                 sleep_duration = (i + 1) * delay - (launch_time - start_time)
#                 if sleep_duration > 0:
#                     await asyncio.sleep(sleep_duration)
#                 # If we fall behind, launch immediately (sleep_duration <= 0)

#         # Wait for all launched tasks to complete and gather results
#         results = await asyncio.gather(*tasks)
#         end_time = time.monotonic()
#         progress_bar.close()


#     # Process results
#     for latency in results:
#         if latency is not None:
#             latencies.append(latency)
#             successful_requests += 1
#         else:
#             failed_requests += 1

#     total_time = end_time - start_time
#     # Calculate throughput based on *successful* requests and *total time*
#     throughput = successful_requests / total_time if total_time > 0 else 0
#     avg_latency = statistics.mean(latencies) if latencies else 0
#     p95_latency = np.percentile(latencies, 95) if latencies else 0


#     return {
#         "total_sent": num_requests,
#         "successful_requests": successful_requests,
#         "failed_requests": failed_requests,
#         "avg_latency_sec": avg_latency,
#         "p95_latency_sec": p95_latency,
#         "throughput_rps": throughput,
#         "total_time_sec": total_time
#     }

# async def benchmark_async(num_requests: int = 100, rate: float = 10): # Default to a rate > 0
#     """Runs the asynchronous benchmark."""
#     print(f"Benchmarking with {num_requests} requests at {rate} req/s (0 = no limit)...\n")

#     if rate == 0:
#         print("Warning: Rate=0 means launching all requests as fast as possible.")

#     batched_result = await measure_rate_limited_performance_async(BATCHED_URL, num_requests, rate, label="Batched")
   

#     print("\n--- Results ---")
#     print("Batched:")
#     print(f"  Total Sent: {batched_result['total_sent']}")
#     print(f"  Successful: {batched_result['successful_requests']}")
#     print(f"  Failed:     {batched_result['failed_requests']}")
#     print(f"  Avg Latency: {batched_result['avg_latency_sec']:.4f}s")
#     print(f"  P95 Latency: {batched_result['p95_latency_sec']:.4f}s")
#     print(f"  Throughput: {batched_result['throughput_rps']:.2f} requests/sec")
#     print(f"  Total Time: {batched_result['total_time_sec']:.2f}s")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Async RAG Load Testing")
#     parser.add_argument("--requests", type=int, default=100, help="Total number of requests to send")
#     parser.add_argument("--rate", type=float, default=32, help="Target requests per second (0 = no limit/maximum concurrency)")

#     args = parser.parse_args()
#     asyncio.run(benchmark_async(num_requests=args.requests, rate=args.rate))



##########THIS WORKS##########


import time
import httpx
import asyncio
import argparse
from tqdm.asyncio import tqdm
import statistics
from typing import List, Optional, Dict, Any
import numpy as np

BATCHED_URL = "http://localhost:8000/rag"
NAIVE_URL = "http://localhost:8001/rag"

async def send_request_async(client: httpx.AsyncClient, url: str, query: str = "What are cats?", k: int = 2) -> Optional[float]:
    start = time.monotonic()
    try:
        response = await client.post(url, json={"query": query, "k": k}, timeout=65.0)
        latency = time.monotonic() - start
        response.raise_for_status()
        return latency
    except httpx.RequestError as exc:
        print(f"\nRequest error at {exc.request.url!r}: {exc}")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None

async def measure_rate_limited_performance_async(url: str, num_requests: int, rate: float, label: str) -> Dict[str, Any]:
    latencies: List[float] = []
    tasks = []
    delay = 1.0 / rate if rate > 0 else 0
    successful_requests = 0
    failed_requests = 0

    async with httpx.AsyncClient() as client:
        progress_bar = tqdm(total=num_requests, desc=label, ncols=100)
        start_time = time.monotonic()

        for i in range(num_requests):
            task = asyncio.create_task(send_request_async(client, url))
            tasks.append(task)
            progress_bar.update(1)

            if delay > 0:
                launch_time = time.monotonic()
                sleep_duration = (i + 1) * delay - (launch_time - start_time)
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

        results = await asyncio.gather(*tasks)
        end_time = time.monotonic()
        progress_bar.close()

    for latency in results:
        if latency is not None:
            latencies.append(latency)
            successful_requests += 1
        else:
            failed_requests += 1

    total_time = end_time - start_time
    throughput = successful_requests / total_time if total_time > 0 else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0

    return {
        "total_sent": num_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "avg_latency_sec": avg_latency,
        "p95_latency_sec": p95_latency,
        "throughput_rps": throughput,
        "total_time_sec": total_time
    }

async def benchmark_async(num_requests: int, rate: float, target: str):
    print(f"Benchmarking '{target}' with {num_requests} requests at {rate} req/s...\n")

    targets = {
        "batched": (BATCHED_URL, "Batched"),
        "naive": (NAIVE_URL, "Naive"),
        "both": [("Batched", BATCHED_URL), ("Naive", NAIVE_URL)]
    }

    def print_summary(name: str, result: Dict[str, Any]):
        print(f"{name}:")
        print(f"  Total Sent:    {result['total_sent']}")
        print(f"  Successful:    {result['successful_requests']}")
        print(f"  Failed:        {result['failed_requests']}")
        print(f"  Avg Latency:   {result['avg_latency_sec']:.4f}s")
        print(f"  P95 Latency:   {result['p95_latency_sec']:.4f}s")
        print(f"  Throughput:    {result['throughput_rps']:.2f} requests/sec")
        print(f"  Total Time:    {result['total_time_sec']:.2f}s\n")

    if target == "both":
        for label, url in targets["both"]:
            result = await measure_rate_limited_performance_async(url, num_requests, rate, label=label)
            print_summary(label, result)
    else:
        url, label = targets[target]
        result = await measure_rate_limited_performance_async(url, num_requests, rate, label=label)
        print_summary(label, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async RAG Load Testing")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests to send")
    parser.add_argument("--rate", type=float, default=1, help="Target requests per second (0 = no limit)")
    parser.add_argument("--target", type=str, choices=["batched", "naive", "both"], default="both", help="Which server to test: 'batched', 'naive', or 'both'")

    args = parser.parse_args()
    asyncio.run(benchmark_async(num_requests=args.requests, rate=args.rate, target=args.target))

