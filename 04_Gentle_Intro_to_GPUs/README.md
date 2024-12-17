# A gentle intro to GPUs
> This aims to give you a bit of history about the GPU itself, why we use it for deep learning tasks, why its way faster than the CPU at certain tasks.

## Hardware
![](assets/cpu.png)
- CPU: Central Processing Unit 
    - General purpose
    - High clock speed
    - Few cores
    - High cache  (The caches on the chip are large compared to the GPU, the memory bandwidth from the CPU to the RAM slots slow, takes time. So we have big caches on chip to pre-load stuff)
    - Low Latency (complete as fast as possible)
    - Low throughput (can't do as much ops per second as GPU)

![](assets/gpu.png)
- GPU: Graphics Processing Unit 
    - Specialized    (can accomplice simpler instructions, but faster)
    - Low clock speed
    - Many cores
    - Low cache       (Higher memory bandwidth to access VRAM)
    - High Latency
    - High throughput

![](assets/tpu.png)
- TPU: Tensor Processing Unit 
    - Specialized GPUs for deep learning algorithms (matrix multiplication, etc)

![](assets/fpga.png)
- FPGA: Field Programmable Gate Array 
    - Specialized hardware that can be reconfigured to perform specific tasks
    - Very low latency
    - Very high throughput
    - Very high power consumption
    - Very high cost

## NVIDIA GPU History
> This is a brief history of NVIDIA GPUs -> https://www.youtube.com/watch?v=kUqkOAU84bA

![](assets/history01.png)
![](assets/history02.png)
![](assets/history03.png)

## What makes GPUs so fast for deep learning?
![](assets/cpu-vs-gpu.png)

On the CPU, you have a ton of control units, little cores, and a ton of caches everywhere. The cores can do advanced complex tasks, but there aren't a lot of them. So you can only do so much.
The GPU - simpler instructions=simpler controller=less space, A ton of cores, and a cache for DRAM.
The idea is: you want to put together a puzzle, and it doesn't matter which order you do it in! So you can do whichever row first or last, or in chunks, as long as you assemble it together in the end properly. This is what the GPU does!


- CPU (host)
    - minimize time of one task
    - metric: latency in seconds

- GPU (device)
    - maximize throughput
    - metric: throughput in tasks per second (ex: pixels per ms)

## Typical CUDA program
1. CPU allocates CPU memory
2. CPU copies data to GPU
3. CPU launches kernel on GPU (processing is done here)
4. CPU copies results from GPU back to CPU to do something useful with it

Kernel looks like a serial program; says nothing about parallelism. Imagine you are trying to solve a jigsaw puzzle and all you are given is the location of each puzzle piece. The high level algorithm would be designed to take these individual pieces, and solve a single problem for each of them; “put the piece in the correct spot”. As long as all the pieces are assembled in the right place at the end, it works! You don't need to start at one corner and work your way across the puzzle. You can solve multiple pieces at the same time, as long as they don't interfere with each other.

## Some terms to remember
- kernels (not popcorn, not convolutional kernels, not linux kernels, but GPU kernels)
- threads, blocks, and grid (next chapter)
- GEMM = **GE**neral **M**atrix **M**ultiplication
- SGEMM = **S**ingle precision (fp32) **GE**neral **M**atrix **M**ultiplication
- cpu/host/functions vs gpu/device/kernels
- CPU is referred to as the host. It executes functions. 
- GPU is referred to as the device. It executes GPU functions called kernels.



