# How to replicate results

Install all appropriate libraries (i.e. `CuPy`, `NumPy`, `time`, `matplotlib`, etc.)

## Task 1
- The functions in `task-1/task.py` call the functions defined in `task-1/distances.py` and `task-1/ann.py`.
- Run `python task-1/distances.py` to compare performance of CuPy and NumPy implementations of the four distance functions.
- Run `python task-1/ann.py` to benchmark ANN and K-Means functions.

## Task 2
- `task-2/ann.py` is identical to `task-1/ann.py`.
- Run `python task-2/serving_rag.py` to start the RAG Server. You can then run `python task-2/load_tester.py` to run the load tester. You can pass three arguments when running this file:
	- `--requests` - to specify the total number of requests to send. Defaults to `100`.
	- `--rate` - to specify how many requests to send per second. Defaults to `1`. `0` is no limit.
	- `--target=[batched|naive]` - to specify which server to run. Defaults to `batched`.