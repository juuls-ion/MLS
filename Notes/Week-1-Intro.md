# Week 1: Intro (Jan. 17)

### 3 factors for big AI boom:

- Bigger data, due to internet booming + mobiles spreading, taking videos, photos, making social media posts etc.
- Improved algorithms, like Deep Neural Networks (DNNs), transformers, Adam Optimiser etc.
- Better hardware, GPUs now up to 100x faster than CPUs for tensor operations.

Data in ML systems starts as text, video, images etc., but are turned into tensors for processing.

### GPUs

GPUs are the standard AI-accelerating hardware. Recently, there’s been development in Tensor Processing Units (TPUs) (good specifically for tensor ops.) and Neural Processing Units (NPUs) (designed for the most specific calculations).

- GPUs are designed with many small on-chip cores, meant to perform matrix multiplications in parallel.
- In AI data centres, 100s of GPUs per rack are combined for extreme parallelism.
- Even though GPUs are very expensive, most of the cost is in energy spent running them.

### Requirements for ML Systems

**DNN programming:** Simplify the definition, modification, and testing of DNN using high-level APIs.

**Automatic differentiations:** Automatically compute gradients for backpropagation in DNNs.

**Tensor processing:** Transform diverse data into a unified tensor format for efficient processing.

**Training & deployment:** Implement common training methods and enable easy model deployment.

**AI accelerators:** Offload computations to various AI accelerators for improved performance.

**Distributed execution:** Distribute DNN workloads when a single accelerator’s memory is insufficient.

### Deep Dive of ML Frameworks (PyTorch)

Motivation behind PyTorch was to have a single Python software library to cover the entire lifecycle of developing an ML program:

- Provides auto-gradient function, which performs backpropagation calculations to derive gradients automatically.
- It uses parallelism a lot.
- PyTorch’s frontend is written in Python to be accessible and useful alongside data science libraries; backend is written in C for efficient performance dealing with hardware (e.g. GPUs)
    - It has C implementations of matrix operations under the hood, as Python is relatively slow. This makes PyTorch faster than plain Python for such calculations.
    - Making repeated system calls to a C kernel for subroutines is expensive. PyTorch creates computational sub-graphs to perform entire sets of calculations on GPUs at once.
        - I have no fucking idea what this means.

### Asynchronous Kernel Dispatching

When you write PyTorch code that uses GPUs (or other accelerators), each tensor operation (e.g. matrix multiplication) is submitted as a “kernel” to the GPU. In **synchronous** dispatch, your CPU thread waits for each GPU operation to finish before moving on. In **asynchronous** dispatch, the CPU simply *queues* the operation on the GPU and then continues running other instructions immediately, without waiting for the GPU to finish.

**Asynchronous kernel dispatching** gives 4 benefits:

- Non-blocking execution
    - The CPU (main thread) does *not* block on every GPU operation; it launches the kernel and moves on. While the GPU is crunching numbers, the CPU can continue loading data etc.
- Parallelism / Pipelining
    - Because the CPU isn’t waiting for each operation to complete, multiple tasks can overlap in time.
- Latency Hiding
    - “Latency” is the waiting time before a result is ready. “Hiding” that latency means overlapping it with other useful work. Asynchronous dispatch can hide the time it takes for one kernel to finish by doing other work in parallel.
- Scalability
    - Scalability here refers to how easily a program can handle larger problems, more data, or additional hardware resources. The CPU can queue work on *multiple* GPUs without forcing a global wait. Each GPU then runs its assigned kernels in parallel with the others.

![ScreenShot](/images/asynchronous.jpg)

### Main overall benefits of PyTorch:

- Python frontend is simple and flexible, with full lifecycle support.
- Underlying computational graph gives a standardised, unified expression for the computational steps (i.e. all ML-model generation processes follow these overall steps).
    - Also provides automatic differentiation, ability to parallelise, use different backends.
- Asynchronous kernel dispatching gives its benefits on different hardware accelerators (GPUs, TPUs, FPGAs etc.)
