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

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Differentiate at a high level between the features of OpenMP, MPI, CUDA and OpenACC
- Briefly summarise the main OpenMP compiler directives and what they do
- Describe how to compile and run an OpenMP program
- Briefly summarise the main MPI message-passing features and how they are used
- Describe how to compile and run an MPI program
- Briefly summarise how an OpenACC program is written and compiled
- Briefly summarise how a CUDA program is written

::::::::::::::::::::::::::::::::::::::::::::::::

To get the most performance possible, high-performance computing relies on parallelism. Modern systems combines tens of
thousands of processors, each working on their own part of the problem. To be able to wield this power, we need to
understand how to write parallel code. This is far more complicated than the scope of this episode. Therefore we will,
instead, look at the landscape of the core HPC technologies used today. In particular, we will look at the following
parallel frameworks: OpenMP, MPI, OpenACC and CUDA.

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

## OpenMP

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

## MPI

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

## Using GPUs instead of CPUs

Besides using multiple CPUs, we can also use Graphical Processing Units (GPUs) to do calculations in parallel. GPUs were
originally designed to speed up rendering to display images to a screen, a task that involves performing millions of
simple, repetitive calculations at once. Researchers soon realised this design was also perfect for many scientific
problems.

GPUs are highly parallel, built to perform thousands of operations at the same time. This makes them ideal for work that
can be split into many identical, independent tasks. While CPUs are designed to tackle complex tasks one after another,
GPUs are optimised for doing the exact same operation on large amounts of data simultaneously. This is perfect for
problems like matrix operations, where every element can be processed in the same way.

However, *offloading* work to a GPU is more complicated than parallelising using CPUs. The main reason is that the CPU and
the GPU have their own separate memory spaces. Data stored in the CPU's memory is not visible to the GPU, and
vice-versa. This setup is similar to the separate spaces for each process in MPI application. Data must be copied from
the CPU's memory to the GPU's memory. This transfer step is slow
compared to the speed of the calculations themselves. Therefore, efficient GPU programs must minimise data
transfers, often by keeping data on the GPU as long as possible. While optimising CPU code often focuses on reducing the
total number of calculations, optimising GPU code is usually more about reducing data transfers and organising data
efficiently in the GPU's memory.

This level of detail is beyond the scope of this introduction. As before, we will only give a broad overview of two
popular frameworks: OpenACC and CUDA. You can think of these as being similar to OpenMP and MPI:

- OpenACC is like OpenMP: you can often add it to existing code, using compiler directives to automatically handle the
  parallelisation.

- CUDA is similar to MPI: it is a more complex framework that requires you to design your program around it from the
  start to achieve the best performance.

::::::::::::::::::::::::::::::::::::: callout

### CPU and GPUs, what's the difference?

![](fig/cpu_gpu_arch.png)

This diagram illustrates the key difference between a CPU and a GPU. On the left, the CPU is shown with a few, large
"Cores" (green). Each core is complex, paired with significant "Control" logic (yellow) and large, fast memory caches
(purple and blue). This design makes each CPU core very "smart" and powerful, ideal for handling complex instructions
and varied tasks one after another, as mentioned earlier.

On the right, the GPU has a completely different structure. It is composed of hundreds or even thousands of tiny, simple
cores (the large green grid). Notice how much less space is dedicated to "Control" logic and complex caches. This
architecture is not designed for complex, sequential tasks. Instead, it is a massive parallel workforce, built to
execute the same simple operation (like `c[i] = a[i] + b[i]`) at the same time across thousands of different pieces of
data. This visual "many-core" design is what allows it to perform thousands of operations concurrently.

GPUs also use a different memory model. Each GPU core can access the large global memory, which is shared across the
entire GPU. It is slower to read and write to this. To improve performance, groups of cores are organised into blocks
which shared a small, but very fast, memory area. This setup is similar to shared-memory parallelism on CPUs, where
threads cooperate through a common memory space. However, each GPU block’s shared memory is private to that block, much
like how separate processes in a distributed-memory model (such as MPI) each have their own memory and must explicitly
exchange data. Efficient GPU programs manage this hierarchy carefully, reusing shared memory to reduce costly access to
global memory.

:::::::::::::::::::::::::::::::::::::::::::::

## OpenACC

OpenACC is a framework for parallel programming on GPUs, both for NVIDIA and AMD GPUs; i.e. it is platform agnostic.
Just like OpenMP, it uses compiler directives to tell the compiler which parts of code should be executed, in parallel,
on a GPU. Also like OpenMP, the OpenACC runtime and compiler handles all of the parallelisation details such as
generating the parallel code, transferring data between the CPU and GPU and synchronising/managing GPU threads.

Whilst most of the heavy lifting for OpenACC is done with compiler directives, a runtime library is also available for
finer control over things such as selecting with GPU to use, finer grained data movement and synchronisation. But to
parallelise our `vector_add` function to a GPU, we only need to add a single line of code.

```c
void vector_add(int *a, int *b, int *c, int n) {
// This OpenACC directive tells the compiler to parallelise the
// following 'for' loop, running its iterations concurrently
// on the GPU
#pragma acc parallel loop
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

The full program is in [vector_openacc.c](files/vector/vector_openacc.c). The directive `#pragma acc parallel loop`
tells the compiler to parallelise the loop and execute it on the GPU. Each GPU thread performs part of the vector
addition. OpenACC also takes care of allocating memory on the GPU and transferring data from the CPU to GPU's memory. It
also copies the results back to the CPU when the loop is completed. In the next section about CUDA, we will see how much
additional code we need to write to do just this.

::::::::::::::::::::::::::::::::::::: callout

## Some other useful directives

As we've seen,`#pragma acc parallel loop` targets a single loop. OpenACC also provides the `#pragma acc kernels`
directive. This directive is meant to be put before a larger region of code, such as a complicated loop. The compiler
then analyses this entire region and automatically determines the best way to convert the code, including any loops it
finds, into parallel "kernels" of code to run on the GPU.

The main difference is that `#pragma acc parallel loop` is prescriptive: you are explicitly telling the compiler to
parallelise that one loop in one specific way. In contrast, `#pragma acc kernels` is descriptive: you are telling the
compiler "here is a block of code to accelerate," giving it freedom to analyse the code and choose the most efficient
way to run it.

Additionally, whilst OpenACC automatically manages data transfers between the GPU and CPU, more explicit data management
is possible using directives such as `#pragma acc data`, `copy`, `copyin`, and `copyout` for more control.

:::::::::::::::::::::::::::::::::::::::::::::

To compile an OpenACC program, we need a compiler that supports it. On Iridis X this is either GCC or one from NVIDIA's
compiler collection. With GCC we need to use the `-fopenacc` flag. However, when using NVIDIA's C compiler the flag is
instead `-acc`, e.g. `nvc -acc vector_openacc.c`. Without the flag, the OpenACC directives are ignored and GPU code is
not generated.

The following is an example of how you would compile and launch an OpenACC program on Iridis X. It is especially
important that we remember to request a GPU using the `--gres=gpu:1` directive and select a partition where nodes have
GPUs.

```bash
#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

# Load the NVIDIA HPC SDK module, which provides OpenACC support
module load nvhpc

# Compile the program with GPU offloading enabled
nvc -acc vector_acc.c -o vector_acc.exe

# Check GPU availability and run the program
nvidia-smi
./vector_acc.exe
```

The main advantage of OpenACC, like OpenMP, is that it allows rapid GPU parallelisation with minimal code changes. It is
particularly useful for incrementally porting existing CPU applications to GPUs. However, compared to lower-level
frameworks like CUDA, it offers less fine-grained control over GPU execution and memory management. OpenACC is therefore
well suited for scientific and engineering applications where productivity and portability are more important than
maximal performance.

## CUDA

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

::::::::::::::::::::::::::::::::::::: keypoints

- Nothing yet.

::::::::::::::::::::::::::::::::::::::::::::::::
