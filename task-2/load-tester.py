import asyncio
import httpx
import time
import json
import argparse


async def send_request(client, url, payload):
    """Sends a single request and returns True on success (200 OK), False otherwise."""
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        # You could optionally check the content of the response here
        # print(f"Received: {response.json()}")
        return True
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        return False
    except httpx.HTTPStatusError as exc:
        print(
            f"Error response {exc.response.status_code} while requesting {exc.request.url!r}"
        )
        # print(f"Response content: {exc.response.text}") # Uncomment to see error details
        return False
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return False


async def run_load_test(url: str, payload: dict, num_requests: int, concurrency: int):
    """Runs the load test with specified parameters."""
    print(f"Starting load test:")
    print(f"  URL: {url}")
    print(f"  Payload: {json.dumps(payload)}")
    print(f"  Total Requests: {num_requests}")
    print(f"  Concurrency: {concurrency}")
    print("-" * 30)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    results = []
    start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:  # Adjust timeout as needed
        for i in range(num_requests):
            # Acquire semaphore before creating task
            await semaphore.acquire()
            task = asyncio.create_task(send_request(client, url, payload))
            # Add a callback to release semaphore when task is done
            task.add_done_callback(lambda t: semaphore.release())
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    successful_requests = sum(results)  # Counts True values
    failed_requests = num_requests - successful_requests

    print("-" * 30)
    print(f"Load test finished in {total_time:.2f} seconds.")
    print(f"  Successful requests: {successful_requests}")
    print(f"  Failed requests: {failed_requests}")

    if total_time > 0:
        throughput = successful_requests / total_time
        print(f"  Throughput: {throughput:.2f} requests per second (RPS)")
    else:
        print("  Throughput: N/A (test duration was zero)")

    if successful_requests > 0:
        avg_latency = total_time / successful_requests * concurrency  # Approximation
        print(
            f"  Approx Avg Latency per request (under load): {avg_latency * 1000:.2f} ms"
        )
    else:
        print("  Avg Latency: N/A (no successful requests)")

    return throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI RAG Throughput Test")
    parser.add_argument(
        "-u", "--url", default="http://127.0.0.1:8000/rag", help="Target URL"
    )
    parser.add_argument(
        "-n", "--num-requests", type=int, default=50, help="Total number of requests"
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "-q", "--query", default="What is a cat?", help="Query string for the payload"
    )
    parser.add_argument(
        "-k",
        "--k-docs",
        type=int,
        default=2,
        help="Number of documents to retrieve (k)",
    )

    args = parser.parse_args()

    test_payload = {"query": args.query, "k": args.k_docs}

    asyncio.run(
        run_load_test(args.url, test_payload, args.num_requests, args.concurrency)
    )
