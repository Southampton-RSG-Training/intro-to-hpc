---
title: "Landscape of HPC Technologies"
teaching: 0 # teaching time in minutes
exercises: 0 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- Which programming language should I use?
- What are the main ways we can parallelise a program on modern HPC systems?
- How do OpenMP and MPI differ in how they achieve parallelism?
- What role do GPUs play in HPC, and how do approaches like CUDA and OpenACC use them?
- Why is it important for code to scale well on large HPC systems?
- What can go wrong if we try to optimise too early in the development process?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Differentiate at a high level between the features of OpenMP, MPI, CUDA and OpenACC
- Briefly summarise the main OpenMP compiler directives and what they do
- Describe how to compile and run an OpenMP program
- Briefly summarise the main MPI message-passing features and how they are used
- Describe how to compile and run an MPI program
- Briefly summarise how a CUDA program is written
- Briefly summarise how an OpenACC program is written and compiled
- Describe why code scalability is important when using HPC resources
- Describe the differences between strong and weak scaling
- Summarise the dangers of premature optimisation

::::::::::::::::::::::::::::::::::::::::::::::::

To get the most performance possible, high-performance computing relies on parallelism. Modern systems combines tens of
thousands of processors, each working on their own part of the problem. To be able to wield this power, we need to
understand how to write parallel code. This is far more complicated than the scope of this episode. Therefore we will,
instead, look at the landscape of the core HPC technologies used today. In particular, we will look at the following
parallel frameworks: OpenMP, MPI, CUDA and OpenACC. We'll additionally also look at the programming languages used in
HPC and touch briefly on how we measure the performance and scalability of our code.

## Common programming languages

In principle, any programming language can be used to write code which runs on an HPC cluster. In practise, however,
there are some languages which are better suited to writing highly performant code. Interpreted languages, such as
Python, are easy to develop in, but are much slower to execute than compile languages, making them less suitable for
computation-heavy applications. For this reason, compiled languages such as C, C++ and Fortran are some of the most
common choices of programming language when writing high performance applications as they produce fast, optimised
executables.

Python does, however, remain widely used in HPC, typically by performance-critical sections of code being written in
compiled languages and accessed through a low-level Python interface. Frameworks such as PyTorch, NumPy, SciPy and Numa
all rely on underlying C, Fortran and/or CUDA code to accelerate computation, while libraries such as mpi4py or
multiprocessing make it possible for distributed parallelism directly from Python. Hybrid approaches like this have
become more common because they combine Python's ease of use with the speed of compiled code.

Other specialised languages and frameworks are also used in HPC, depending on the application area. CUDA and ROCm are
employed for GPU programming, while newer languages such as Julia aim to combine ease of development with high
performance. Domain-specific tools like MATLAB, and R also appear in HPC environments, though they often rely on compiled
extensions or external libraries for parallel execution.

So which language should you use? There is no simple answer. The best choice depends heavily on your specific
application, the target hardware (like CPUs or GPUs), and the trade-offs you are willing to make between raw
computational performance and ease of development. Ultimately, the right language is the one that allows you to solve
your problem efficiently, leveraging the available libraries and expertise within your team.

::::::::::::::::::::::::::::::::::::: callout

### Which compiler should you use?

Iridis and most HPC systems offer a choice of compilers to use. But which one should you use? In most cases, you would
want to use the compiler specific to the type of CPU you are using on the system, because these implement architecture
specific optimisations e.g. use AMD's `aocc` on Iridis 6. If you are using GPUs, you have little choice in using
anything other than NVIDIA's `nvc`, `nvcc` or `nvfortran` compilers.

However, the code you are using may only have been tested on a specific compiler such as GCC. In those cases, it's often
best to stick with what is known to work. However, there is nothing stopping you using Intel's compilers on an AMD based
system, if the code is only tested or depends on the Intel compilers.

:::::::::::::::::::::::::::::::::::::::::::::

## Landscape of technologies

We'll now take a high level look at a selection of some of the most used libraries and frameworks used to parallelise
code in research. In particular, we will see how to use them, the code changes required and how to run them on Iridis.
The intention is not to make you proficient with these frameworks--or even dangerous--but to give a high level
appreciation on what is being used and how.

To illustrate this, we will use a simple program, written in C, which adds together two vectors to explore the code
changes required. More specifically, we'll care be modifying the following function `vector_add`. We have chosen to use
C here, but the language does not really matter.

```c
// *a, *b, and *c are arrays and n is the length of them.
// The result of the addition is returned back in *c
void vector_add(int *a, int *b, int *c, int n) {
    // This is the loop which we'll parallelise
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

You can find the entire program in [vector_serial.c](files/vector/vector_serial.c). To run this code on Iridis X, we'll
use the following [submission script](files/vector/submit_vector_serial.sh).

```bash
#!/bin/bash
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00

# Load the gcc module, which is the compiler we'll use
module load gcc

# Compile the program using gcc
gcc vector_serial.c -o vector_serial.exe

# Run the compiled executable
./vector_serial.exe
```

After the program has run, we should see the following output.

```output
Verification (first 5 elements):
c[0] =   0 (expected:   0)
c[1] =   3 (expected:   3)
c[2] =   6 (expected:   6)
c[3] =   9 (expected:   9)
c[4] =  12 (expected:  12)
```

### OpenMP

The first framework we'll look at is OpenMP. As mentioned in the previous episode, OpenMP is an industry-standard
framework designed for parallel programming in a shared-memory environment. OpenMP spawns threads with each one,
ideally, running on its own CPU core. OpenMP works by using *compiler directives* to tell the compiler which code needs
to be parallelised, letting OpenMP and the compiler takes care of all the parallelisation details. In general, you just
need to say which parts of your code you want to run in parallel.

::::::::::::::::::::::::::::::::::::: callout

### Compiler directives

If you're unfamiliar with compiler directives, you can think of them as being a *special command* for the compiler, not
for the program itself. In C and C++, these almost *always* start with `#pragma`. Think of it as a special note to the
compiler which says, "when you compile this specific piece of code, do something extra." Since these are compiler
options, they do not modify the run time behaviour of the program, only how the final executable is compiled.

:::::::::::::::::::::::::::::::::::::::::::::

Most of the parallelisation with OpenMP is done with these compiler directives. However, OpenMP does also offer a
library of runtime functions which gives finer grained control, such as if you need to ensure thread synchronisation or
need to go off the beaten track. To parallelise our `vector_add` function, we only need to add a single line of code,
using a compiler directive, just before the loop.

```c
void vector_add(int *a, int *b, int *c, int n) {
// This directive tells OpenMP to spawn threads and to
// divide the loop iterations between them. Each thread will
// handle a fraction of the loop iterations/vector addition
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

The full program is in [vector_openmp.c](files/vector/vector_openmp.c). Let's break this down more. The directive we
used, `#pragma omp parallel for`, tells the compiler that the next for loop should be parallelised. The compiler then
automatically parallelises it for us, creating a team of threads and dividing the loop's work among them. In this case,
each thread will perform a portion of the vector addition. All OpenMP directives begin with `#pragma omp`, followed by a
specific command.

There are lots of other directives available, with `#pragma omp parallel for` being the most commonly used. Another
useful directive is `#pragma omp atomic` which prevents multiple threads from modifying a variable at once. This is one
way to prevent a race condition mentioned in the previous episode. OpenMP also provides a library of runtime functions
which offers even more control. For example, we can use the function `omp_set_num_threads` to control the number of
threads that OpenMP will use. In C, we need to include the appropriate header file, `omp.h`, to access this function.

```c
#include <omp.h>

void vector_add(int *a, int *b, int *c, int n) {
    // Manually set the number of threads to 8
    omp_set_num_threads(8);

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

::::::::::::::::::::::::::::::::::::: callout

### Directives and functions for synchronisation

Effective control of thread synchronisation is essential when parallelising code with OpenMP, as improper handling of
shared data can lead to race conditions and unpredictable results. To support this, OpenMP provides a range of
directives and library functions that coordinate access to shared data and manage thread behaviour. The tables below
summarise several commonly used examples.

| Compiler Directive     | Description                                                                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `#pragma omp atomic`   | Ensures a specific operation, such as modifying a variable, is executed atomically, e.g. by one thread at a time, to prevent race conditions. |
| `#pragma omp critical` | Defines a region of code that only one thread can execute at a time.                                                                          |

This list is not exhaustive. A complete reference for all OpenMP directives and functions is available in the [OpenMP
6.0 Reference Guide](https://www.openmp.org/wp-content/uploads/OpenMP-RefGuide-6.0-OMP60SC24-web.pdf).

:::::::::::::::::::::::::::::::::::::::::::::

To compile an OpenMP program, we need to use the `-fopenmp` flag, e.g. `gcc -fopenmp vector_openmp.c`. If we don't use
the `-fopenmp` flag, the compiler directives are ignored and any library functions from `omp.h` will not be found
causing a compilation error.

::::::::::::::::::::::::::::::::::::: callout

### Do I need to keep the serial version?

Setting the number of threads or processes to one will produce the same behaviour as the serial program, assuming no
programming errors. However, to be extra safe, you can use conditional compilation to maintain both parallel and serial
versions within the same code. Compiled languages support conditional compilation, which allows certain sections of code
to be compiled only if a specific condition is met. When using OpenMP, the compiler variable `_OPENMP` is defined when
the `-fopenmp` flag is passed, so you can use this variable to prevent OpenMP directives and functions from being
compiled when `-fopenmp` is not used.

:::::::::::::::::::::::::::::::::::::::::::::

To launch an OpenMP program, run it like any other program. The number of threads for OpenMP to use can be controlled
using the environment variable `OMP_NUM_THREADS`. If this is left unset, OpenMP will spawn one thread per logical CPU
core. Normally, on a HPC cluster this is probably what is wanted. However, when running on your own computer during
development and testing, you will probably want to set `OMP_NUM_THREADS` to be less than then number of CPU cores to
avoid overloading the computer.

The following is an example of how you would compile and launch an OpenMP program on Iridis X.

```bash
#!/bin/bash

#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:01:00

# Use OMP_NUM_THREADS to say we can to use --cpus-per-task number of
# threads. The --cpu-per-task SBATCH directive is populated into the
# SLURM_CPUS_PER_TASK environment variable
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile the program using gcc
module load gcc
gcc -fopenmp vector_openmp.exe -o vector_openmp.exe

# We run the compiled executable just like the serial version
./vector_openmp.exe
```

The main advantage of OpenMP is that it requires only minimal code modification to parallelise existing programs,
particularly when the main computational workload lies within loops. Beyond simple loop-level parallelism, parallelising
more complex program structures becomes more challenging—though this is true of most parallel frameworks. OpenMP still
provides a wide range of straightforward directives, making it easy to adapt serial code without needing to design the
program for parallel execution from the start. Its main limitation is that it uses a shared-memory model, which requires
careful management of thread synchronisation and restricts scalability to a single compute node.

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

### OpenACC

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

### CUDA

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


<!-- ## Putting It Together

- Comparing OpenMP, MPI, OpenACC, and CUDA at a high level.
- Typical use cases (e.g. CFD, ML, simulations).
- Choosing the right model for the problem. -->

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
