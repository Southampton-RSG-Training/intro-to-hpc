---
title: "Introduction to HPC Technologies"
teaching: 0 # teaching time in minutes
exercises: 0 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- What are the main ways we can parallelise a program on modern HPC systems?
- How do OpenMP and MPI differ in how they achieve parallelism?
- What role do GPUs play in HPC, and how do approaches like CUDA and OpenACC use them?
- Why might we combine OpenMP and MPI in the same program?
- Why is it important for code to scale well on large HPC systems?
- What can go wrong if we try to optimise too early in the development process?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Differentiate at a high level between the features of OpenMP, MPI, CUDA and AI/ML approaches and what they are used for
- Briefly summarise the main OpenMP compiler directives and what they do
- Describe how to compile and run an OpenMP program
- Briefly summarise the main MPI message-passing features and how they are used
- Describe how to compile and run an MPI program
- Describe the advantages and drawbacks for using a hybrid OpenMP/MPI approach
- Briefly summarise how a CUDA program is written
- Describe why code scalability is important when using HPC resources
- Describe the differences between strong and weak scaling
- Summarise the dangers of premature optimisation

::::::::::::::::::::::::::::::::::::::::::::::::

High-performance computing relies on parallelism. Modern systems combine thousands of processors, each capable of
working on part of a problem. To use this power effectively, we must understand the main models of parallel
programming—shared memory, distributed memory, and accelerator-based computing—and how each fits into real workloads.

In this lesson, we explore the core HPC technologies: OpenMP for threading, MPI for message passing, and GPU frameworks
such as CUDA and OpenACC. We will see how they differ, where they overlap, and why combining them often produces the
best results. We will also touch on performance measurement and scalability—vital concepts for making efficient use of
large systems.

## Common languages

You can use any programming language you want to run code on a HPC cluster. However, there are good and bad choices. For
example, whilst Python is easy to write and develop, the slow performance makes it a less desirable language to write a
program in which demands high performance. Because of this, it's usually best to use a compiled language. The most
common compiled languages in research are C, C++ and Fortran.

This doesn't mean you can't run Python on HPC. In fact, it is very common. However, it is often used in a way where the
computationally tough bits of the code are written in other languages. An example of this is PyTorch, where the
computational bits are written in C++ and accelerated using CUDA. Python libraries such a Numpy and Numba also take
advantage of compiled code to speed up computation. Furthermore, MPI is also available for Python.

## Landscape of technologies

The intention of this section is to give an idea of how the popular libraries and frameworks are used to paralleise code
and how they are run on HPC clusters.

We will be taking this short program to add together two vectors and showing how they are parallelised. We won't go over
the details of the implementation, giving only a high level overview of what's happening.

The function which we will parallelise is `vector_add`

[vector.c](files/vector_serial.c)

```c
void vector_add(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}
```

```bash
#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00

module load gcc
gcc vector_serial.exe -o vector_serial.exe
./vector_serial.exe
```

### OpenMP

- Concept: shared-memory model.
- Common directives (`parallel`, `for`, `reduction`).
- Compilation (`gcc -fopenmp`).
- Simple Slurm job example with `--cpus-per-task`.
- Strengths (simple to add) and limits (single-node memory).

```c
void vector_add(int *a, int *b, int *c, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}
```

```bash
#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:01:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gcc
gcc -fopenmp vector_openmp.exe -o vector_openmp.exe
./vector_openmp.exe

```

### MPI

- Concept: distributed-memory model using message passing.
- Key functions (`MPI_Init`, `MPI_Send`, `MPI_Recv`, `MPI_Finalize`).
- Compilation (`mpicc`).
- Example Slurm job with `--ntasks`.
- Benefits (scales across nodes) and challenges (communication overhead).

```c
void vector_add(int *a, int *b, int *c, int n)
{
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_local = n / size;
    int *a_local = malloc(n_local * sizeof(int));
    int *b_local = malloc(n_local * sizeof(int));
    int *c_local = malloc(n_local * sizeof(int));

    MPI_Scatter(a, n_local, MPI_INT, a_local, n_local, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, n_local, MPI_INT, b_local, n_local, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n_local; i++)
    {
        c_local[i] = a_local[i] + b_local[i];
    }

    MPI_Gather(c_local, n_local, MPI_INT, c, n_local, MPI_INT, 0, MPI_COMM_WORLD);

    free(a_local);
    free(b_local);
    free(c_local);

    MPI_Finalize();
}
```

```bash
#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:01:00

# Load a MPI library, which give access to the library files and tools part of
# MPI
module load openmpi
mpicc vector_mpi.exe -o vector_mpi.exe

# Using the srun command is the easiest way to launch an MPI program on Iridis.
# It is part of Slurm and gives lots of flexibility on how to launch programs
# on multiple nodes. It uses the values in the SBATCH directives above to configure
# how to launch our program
srun ./vector_mpi.exe

# Alternatively, you can use mpirun/mpiexec which is part of the MPI library. Functionally
# it does the same thing as srun, but requires more manual setup as it is has no
# knowledge of Slurm and the resources Slurm allocated
mpirun -np $SLURM_NTASKS ./vector_mpi.exe
```

::::::::::::::::::::::::::::::::::::: callout

### Hybrid MPI+OpenMP

MPI and OpenMP don’t have to be competing choices. They can be used together in a hybrid parallel model, where MPI
distributes work across nodes and OpenMP manages threads within each node. This combination allows applications to scale
beyond a single machine while using memory and cores more efficiently.

Hybrid parallelism reduces data duplication between processes and improves load balancing through OpenMP’s flexible
scheduling. It also helps lower communication costs by keeping shared-memory operations local to a node.

The main drawbacks are added complexity and potential overheads from managing both models. Code becomes harder to write,
debug, and port between systems. Still, hybrid MPI+OpenMP programs are often the best solution for large-scale workloads
where pure MPI or OpenMP alone falls short.

:::::::::::::::::::::::::::::::::::::::::::::

### GPU Parallelisation

```bash
#!/bin/bash

#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

# The NVHPC module includes the libraries required for GPU parallelisation
module load nvhpc

nvcc vector_cuda.cu -o vector_cuda.exe
nvidia-smi
./vector_cuda.exe
xe
```

CUDA:

- Explicit GPU programming model (kernels, threads, memory).
- Example kernel declaration (`__global__ void kernel(...)`).
- Compilation (`nvcc`).
- Slurm job example with `--gres=gpu:1`.
- Benefits (fine-grained control) and drawbacks (complexity, vendor lock-in).

```cpp
__global__ void vector_add_kernel(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void vector_add(int *a, int *b, int *c, int n)
{
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

OpenACC:

- Directive-based GPU offload, similar to OpenMP: `#pragma acc parallel loop`.
- Compilation also similar to OpenMP `nvc -acc`.
- Advantages: incremental acceleration; limits: less control.

```c
void vector_add(int *a, int *b, int *c, int n)
{
#pragma acc parallel loop
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}
```

## Putting It Together

- Comparing OpenMP, MPI, OpenACC, and CUDA at a high level.
- Typical use cases (e.g. CFD, ML, simulations).
- Choosing the right model for the problem.

## Measuring and improving parallel performance

When we submit a job to run on a cluster, we have the option of specifying the amount of memory and the number of CPUs
(and GPUs) that will be allocated. We need to consider to what extent that code is *scalable* with regards to how it
uses the requested resources, to avoid asking for, and wasting, more resources than can be used efficiently. So before
we start asking for lots of resources, we need to know how the performance of our scales with the number of CPUs (or
GPUs) made available to it. There are two primary measures of execution time we need need to measure:

- **Wall clock time (or actual time)** - this is the time it takes to run from start of execution to the end, as
  measured on a clock. In terms of scaling measurements, this does not include any time waiting for the job to start.
- **CPU time** - this is the time actually spent running your code on a CPU, when it is processing instructions. This
  does not include time waiting for input or output operations, such as reading in an input file, or any other waiting
  caused by the program or operating system.

In most cases, measuring just the wall clock time is usually sufficient for working out how your code scales. But what
is code scalability?

### What is scalability?

Scalability describes how efficiently a program can use additional resources to solve a problem faster, or to handle
larger problems. A scalable code continues to achieve performance improvements as more resources are allocated to it.
Programs which don't scale well show diminishing returns as more resources are allocated, often due to bottlenecks such
as serial code sections or other overheads. It's important to note that not all programs need to scale perfectly or to
hundreds of thousands of processors. Every program has a practical scaling limit beyond which performance gains level
off or even decline. What matters is understanding where the limit lies for your application and what the bottleneck is.

Bottlenecks are the parts of a program that limit its scalability. Even small sections of serial code, or operations
that require coordination between processors, can dominate the total runtime. According to [Amdahl’s
Law](https://en.wikipedia.org/wiki/Amdahl%27s_law), the speedup of a parallel program is constrained by its serial
fraction, so perfect scaling is impossible when any part of the program must execute sequentially. Typical bottlenecks
include communication overhead, synchronisation delays, I/O operations, and load imbalance. As processor count
increases, these costs can outweigh the benefits of parallel execution, leaving some resources idle.

Scalability is measured by observing how the program's execution time changes as the number of processors increases.
This can be quantified through speedup (the ratio of single-processor runtime to multi-processor runtime) and efficiency
(the ratio of achieved speedup to the number of processors used). These are calculated through measurements of wall
clock time and plotted against the processor count to show how performance scales.

Measuring scalability helps identify whether performance limitations stem from the code itself, the problem size, or the
system architecture. Because computing resources are finite, measuring scalability is essential to ensure they are used
efficiently. It allows you to determine when adding more cores no longer provides meaningful benefits, preventing wasted
resources.

### Strong Scaling

Strong scaling measures how execution time changes when the problem size *stays constant* but the number of processors
increases. Ideally, when doubling the processor count we should see expect for the runtime to be halved. In practise,
performance gains are limited by serial code and overheads such as communication, synchronisation or I/O operations
limited by the file or operating system.

### Weak Scaling

Weak scaling measures show runtime changes when both the problem size and number of processors increase proportionally,
keeping the workload per processor constant; in contrast, when measuring strong scaling, the workload per processors
decreases. Ideally, the runtime should remain constant as more processors are added. Weak scaling is important for large
simulations which would be functionally impossible without a large number of resources.

### The dangers of premature optimisation

If your code is still taking too long to run after parallelising it, or if it scales poorly, it's tempting to dive head
first in and try to optimise everything you think is slow! But before you do that, you need to think about the [rules of
optimisation](https://hackernoon.com/the-rules-of-optimization-why-so-many-performance-efforts-fail-cf06aad89099):

1. Don't,
2. Don't... *yet*, and,
3. If you need to optimise your code, *profile* it first.

For most code we write, premature optimisation is often bad practice which leads to long nights of debugging.
Optimisation often leads to more complex code resulting in code which is more difficult to read, making it harder to
understand and maintain; even with all the code comments in the world! Another issue is that your premature optimisation
may change the result without you realising until much further down the line.

It is often effort-intensive, and difficult at a low level, particularly with modern compilers and interpreters, to
improve on or anticipate the optimisations that they already automatically implement for us. It is often better to focus
on writing understandable code which does what you want and then *only* optimise if it too slow. You will often find
that code you think is going to be slow, is often fast enough to not be a problem!

Once you have measured the strong and weak scaling profiles of your code, you can also *profile* your code to find where
the majority of time is being spent to best optimise it. Only then should you start thinking about optimising. If you
want to take this philosophy further, consider the [Rules of Optimisation
Club](https://perlbuzz.com/2008/02/19/the_rules_of_optimization_club/).

::::::::::::::::::::::::::::::::::::: callout

## What is profiling?

Profiling your code is all about understanding its complexity and performance characteristics. The usual intent of
profiling is to work out how best to *optimise* your code to improve its performance in some way, typically in terms of
speedup or memory and disk usage. In particular, profiling helps identify *where* bottlenecks exist in your code, and
helps avoid summary judgments and guesses which will often lead to unnecessary optimisations.

Each programming language will typically offer some open-source and/or free tools on the web, with you can use to
profile your code. Here are some examples of tools. Note though, depending on the nature of the language of choice,
the results can be hard or easy to interpret. In the following we will only list open and free tools:

- Python: [line_profiler](https://github.com/pyutils/line_profiler),
  [prof](https://docs.python.org/3.9/library/profile.html)
- C/C++: [xray](https://llvm.org/docs/XRay.html), [perf](https://perf.wiki.kernel.org/index.php/Main_Page),
  [gprof](https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_mono/gprof.html)
- R: [profvis](https://github.com/rstudio/profvis)
- MATLAB: [profile](https://www.mathworks.com/help/matlab/ref/profile.html)
- Julia: [Profile](https://docs.julialang.org/en/v1/manual/profile/)

[Donald Knuth](https://en.wikipedia.org/wiki/Donald_Knuth) said *"we should forget about small efficiencies, say about
97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that
critical 3%."* Optimise the obvious trivial things, but avoid non-trivial optimisations until you've understood what
needs to change. Optimisation is often difficult and time consuming. Pre-mature optimization may be a waste of your
time!

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Premature optimisation is often dangerous. You should first profile and assess the scaling of your code before you
  decide to optimise it.

::::::::::::::::::::::::::::::::::::::::::::::::
