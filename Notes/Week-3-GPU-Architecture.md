# Week 3: GPU Architecture (Jan. 30, 31)

### Use of GPUs

All modern AI applications use **GEMM** (General Matrix Multiplication) and **GEMV** (General Matrix-Vector Multiplication).

<aside>

**GEMM:** C = A * B, where A, B, C are matrices
**GEMV:** y = A * x, where A is a matrix, x, y are vectors

</aside>

GEMM/V are vital to deep learning, as matrix operations dominate inference + training for convolutional layers (very common layer in Deep/Convolutional Neural Networks).

- Also very useful for LLM training.
- Optimising these operations therefore is key to improving training.

### Common Features of AI Accelerators

Increases in model size/complexity lead to hardware innovations:

- Transition from general-purpose CPUs to specialised AI chips (ASICs, GPUs).
- Focus on parallelism and scalability for complex AI tasks.
- Rise of energy-efficient AI accelerators.
- Increasing role of edge AI with chips designed for real-time, low-latency tasks.
    - All these features are designed with the intention of high parallelism for specific tasks. In deep learning, this refers to GEMM/V.

### NVIDIA GPUs

We use NVIDIA GPUs on the course:

- GPUs in general are suitable for AI tasks.
- NVIDIA GPUs are very accessible and well-documented.
- Many user-friendly libraries in the NVIDIA ecosystem.

### GPU Substructure

There’s a lot of waffle about the actual GPU structure. The main point is that GPUs are made of a bunch of building-blocks called **Streaming Multiprocessors** (SMs).

The SM is where the GPU does its core parallel computation. Each SM contains a bunch of CUDA cores, which do the computation, and some L1 caches and registers. 

The GPU is essentially a bunch of SMs, all connected to the main GPU RAM and an L2 cache.

### CUDA

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. It is a C++-like programming language that lets developers use GPU computing libraries for general-purpose processing.

**CUDA Kernel Code:** A function that runs on the GPU, but is initially called by the CPU

**CUDA Thread:** Lowest unit of GPU programming (i.e. a basic task to run, like GEMM).

**CUDA Warps**: Less important. A group of 32 threads that are scheduled together and executed in parallel. Typical GPU unit of execution. Normally all threads in a warp do the same instruction.

**CUDA Blocks:** Group of warps (essentially group of threads). Smallest unit of thread coordination exposed to programmers.

- **Blocks execute independently!**

**CUDA Grid**: A collection of blocks, created when the kernel is launched. Can be 1D, 2D, 3D.

- Grids and blocks are **logical concepts** used in CUDA, they don’t literally exist on the GPU. They let programmers design abstract code, which can be used on any NVIDIA GPU.
- Blocks and threads are the parameters that define a grid’s dimensionality.

### Combining GPU Architecture and CUDA

- Roughly speaking, CUDA cores handle threads, SMs handle blocks, the kernel runs in the grid.
- Shared memory (i.e. per block) is stored in the L1 cache.
- Global memory (i.e. for the grid) is stored in the GPU RAM.
- L2 cache helps copy memory from RAM to L1 cache.

The purpose of structuring into blocks/grids is because CUDA code is complied into blocks. This means that CUDA code can work on any NVIDIA GPU set-up, and can be abstractly split across GPUs, depending on the allocation of blocks in each GPU.

### GPU Programming

This course will use CuPy, Triton, PyTorch. These are all Python APIs for CUDA. We should definitely use CuPy.

When code is executed, we allocate out kernel function to a grid. 

For each grid, we have gridDim blocks and use blockIdx to locate the block inside.

For each block, we have blockDim threads and use threadIdx to locate the thread inside.

Per grid: 

- gridDim → # of blocks
- blockIdx → ID # per block

Per block:

- blockDim → # of threads per block
- threadIdx → ID # of threads inside, per block

Dim and Idx can be 1, 2, or 3 dimensions, but we typically use 1 dimension for 1D array and 2 dimensions for 2D-array (matrix).

In 1D array:

<aside>

idx = (blockIdx * blockDim) + threadIdx 

</aside>

gives a unique ID for each thread being used in the grid!

In practice, we set a static # of threads per block for a given task, and scale the # of blocks based of the size of the task.

- E.g. in GEMM, thread 0 of each block will handle the first-indexed value of each matrix and multiply them together to produced the first-indexed value of the output matrix.

### Setting <gridDim, blockDim>

Total thread number = gridDim * blockDim. If either is too low, GPU cores are not fully utilised. 

If blockDim is too high, excess thread # per block will overwhelm each block.

- There are only so many cores per SM, so if our blockDim exceeds the cores used per SM, some cores will have to be split across multiple threads, which is very inefficient (essentially double the processing time for that block).

In practice, the blockDim is usually set to 256. It divides evenly by 32, so it is compatible with the warp-based architecture. 256 threads per block often strikes a good balance between:

- **Occupancy:** Enough active threads so that if one warp stalls (e.g. waiting for data), other warps can do work.
- **Register/Shared Memory Use:** Not so big that you run out of SM resources.

### Host and Device Memory

When programming in CUDA, there’s a bunch of stuff you have to do to make the program actually work. You need to do a bunch of low-level nonsense to copy memory addresses to the GPU, and copy them back from GPU memory when calculations are complete. Thankfully, CuPy abstracts away all this for us. 

- One important note, I think you still have to use cudaDeviceSynchronise() to synchronise between the CPU and GPU at points. GPU execution is asynchronous, meaning that once the CPU sets it off, it will immediately begin its other tasks, unless you make it wait and sync up.

### Programming GPUs

Python is kinda goated, and the APIs (CuPy, PyTorch etc.) abstract away a lot of the set up. 

- They have kernel auto-generation, so you don’t need to manually fiddle with GPU memory per operation.
- They also abstract away block/thread number if you don’t want to deal with it.

CuPy has some disadvantages:

- **Performance Overhead:** Slower for small matrices due to kernel launch overhead.
- **Structural Limitations:** Less flexible for complex data structures compared to NumPy.
- **Python only:** CuPy is a Python only programming API

### CUDA Streams

A **CUDA stream** is a queue of GPU operations (like kernel launches, memory copies) that run in the order they are placed into the queue. Multiple streams on a single GPU can potentially run concurrently, as long as there are no explicit dependencies among those operations.

Streams improve performance:

- **Concurrent Kernel Execution:** If you have two unrelated tasks to run, you can put them in separate streams and the GPU can execute them simultaneously.
- **Pipelining:** By splitting a job into smaller batches and using multiple streams, stream A can copy the first batch to the GPU, while stream B can compute on previously transferred batches.
    - Stream C can also copy results back to the CPU.

Streams can also improve speed by uploading small bits of data to the GPU at a time, and processing them ASAP, instead of waiting for a complete data transfer to begun processing.

### Optimisation

This is all “further reading” to really optimise with CuPy.

CuPy can set block/grid numbers manually for optimisation, but this process is GPU dependent.

Use the Python package `nvitop` to check GPU utilisation, temperature, etc.

Other things to look up:

- cuBLAS
- Bank Conflict
- Register Spill
- Memory Coalescing
