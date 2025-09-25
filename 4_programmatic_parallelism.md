---
title: "Introduction to Programmatic Parallelism"
teaching: 0 # teaching time in minutes
exercises: 0 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- Did you know you have to have this question section?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe the concept of parallelisation and its significance in improving performance
- Compare and contrast the operations and benefits of shared and distributed memory systems
- Differentiate between parallelising programs via processes and threads
- Define a race condition and how to avoid them

::::::::::::::::::::::::::::::::::::::::::::::::

Parallel programming has been important to (scientific) computing for decades as a way to decrease program run times,
making more complex computations possible, such as climate modelling, pharmaceutical development, aircraft design, AI
and machine learning, etc. Without parallelisation, these computations which would take years to finish can be completed
in hours or days. In this episode, we will cover the foundational concepts of parallel programming. But, before we get
into the nitty-gritty details of parallelisation frameworks and techniques, let's first familiarise ourselves with the
key ideas behind parallel programming.

## What is parallelisation?

At some point you, or someone you know, has probably asked "How can I make my code run faster?" The answer to this
question will depend on the code, but there are a few approaches you might try:

- Optimise the code.
- Move the computationally demanding parts of the code from a slower, interpreted language, such as Python, to a faster,
  compiled language such as C or Fortran.
- Use a different theoretical/computational or approximate method which requires less computational power.

Each of the above approaches is intended to reduce the total amount of work the processor does. A different strategy for
speeding up code is parallelisation, which is where you split the computational workload across multiple processing
units. The "processing units" are typically either central processing units (**CPU**s) or graphics processing units
(**GPU**s).

## Sequential vs. parallel computing

Classically, computers execute one operation/instruction at a time in the sequence specified by the code you have
written. In other words, the software you have developed is converted--or compiled--into a series of instructions which
are executed one after another. We call this serial execution.

In contrast, with parallel computing, multiple instructions from the same program are executed simultaneously on
different processing units which are working independently on their own set of instructions. This means more work is
done at once, so we get the results quicker than if we were running an equivalent sequential program executing only one
instruction at a time. The process of changing sequential code to parallel code is called parallelisation.

::::::::::::::::::::::::::::::::::::: callout

The basic concept of parallel computing is simple: we divide our job in tasks that can be executed at the same time so
that we finish the job in a fraction of the time that it would have taken if the tasks are executed one by one.

Suppose that we want to paint the four walls in a room. This is our **problem**. We can divide our **problem** into 4
different **tasks**: paint each of the walls. In principle, our 4 tasks are independent of each other in the sense that
we don't need to finish one to start another. However, this does not mean that the tasks can be executed simultaneously
or in parallel. It all depends on the amount of resources that we have for the tasks.

If there is only one painter, they could work for a while in one wall, then start painting another one, then work a
little bit on the third one, and so on. The tasks are being executed concurrently **but not in parallel** and only one
task is being performed at a time. If we have 2 or more painters for the job, then the tasks can be performed in
**parallel**.

In our analogy, the painters represent a computer's CPU cores. The number of CPU cores available determines the maximum
number of tasks that can be performed in parallel. While we might have dozens of tasks (e.g. painting sections of
each wall), we can only make progress on a number of them equal to the number of CPU cores (painters) we have. The
process of managing many tasks with fewer resources is concurrency.

::::::::::::::::::::::::::::::::::::::::::::::::

## Key parallelisation concepts

There is, unfortunately, more to parallelisation than simply diving work across multiple processors. Whilst the idea of
splitting tasks to achieve faster results is *conceptually* simple, the implementation requires an understand of how
work is executed across multiple processors and how the memory that this work uses is organised.

Consider a simple computer where you have a single CPU core, some RAM (fast primary memory), a storage device such as a
hard disk (slower secondary memory), input devices (e.g. keyboard, mouse) and an output device (screen). Now, imagine
you add more CPU cores. Suddenly, there are several additional things you need to think about:

- If there are two cores, they might share the same RAM (shared memory) or each have their own dedicated RAM (private
  memory). This distinction affects how data can be accessed and shared.
- In a shared memory setup, what happens if two cores try to read or write the same memory location at the same time?
  This can cause a race condition, where the outcome depends on the timing of operations. Preventing these conflicts
  requires careful programming.
- How do we divide and distribute the computational tasks among the CPU cores? Dividing the workload evenly between the
  cores is essential for maximum speed-up.
- How will the cores exchange data and coordinate their actions? Mechanisms are now required to share data and
  coordinate actions to ensure results are consistent and correct.
- After the tasks are complete, where should the final results be stored? Should they remain in the memory of one core,
  be copied to a shared memory area, or written to disk? Additionally, which core handles producing the output on the
  screen?

To make effective use of multiple CPU cores, we must understand what **processes** and **threads** are, and how they
interact with the computer's memory.

## Processes and threads

### Processes

A process is an individual running instance of a program. Each process operates independently and possesses its own set
of resources, such as memory space and open files, managed by the operating system. Because of this separation, data in
one processes is isolated and cannot be directly accessed by another.

In one approach to parallel programming, the aim is to achieve parallel execution by running many coordinated processes
at the same time. However, what if one processes needs information from another processes? Since processes are isolated
and do not share memory, the information has to be explicitly communicated by the programmer. This is the role of
parallel programming frameworks such as MPI (Message Passing Interface). MPI provides a standardised library of
functions that allow processes to exchange messages, coordinate tasks and collectively work on a problem together.

This style of parallelisation--spawning large numbers of independent processes--is the dominant form of parallelisation
on HPC systems. By combining MPI with a cluster's job or resource scheduler, it is possible to launch and coordinate
processes are many compute nodes. Instead of having access to just a single CPU, our code can now use thousands or even
tens of thousands of CPUs.

![Processes](fig/multiprocess.svg)

### Threads

A thread is a unit of execution which exists within a process. Unlike processes, threads share their parent process'
resources, including memory and open files, so they can directly read and write the same data. This shared access makes
threads more efficiently than processes, since they do not have to communicate information between them.

By running several threads across multiple CPU cores, program can dictate for each thread to work on their own tasks.
For example, one thread may handle input/output whilst other threads perform some number crunching, or multiple threads
might process different parts of a dataset simultaneously.

A major advantage of using threads is their relative ease of use compared to processes. With frameworks such as OpenMP,
existing code can often be adapted for parallel execution with relatively minor changes. Because threads share a memory
space, there is no need for explicit message-passing mechanisms (as required with MPI). However, this shared memory
model introduces the possibility of race conditions, where two or more threads attempt to update the same data at once.
Careful synchronisation is therefore required.

It is important to note, however, that threads a confined to a single process and therefore to a single computer.
Programs which are parallelised using threads cannot span across compute nodes in a cluster.

![Threads](fig/multithreading.svg)

## Shared vs. distributed memory parallelisation

## Synchronisation and Race conditions

::::::::::::::::::::::::::::::::::::: keypoints

- You need a list of key points

::::::::::::::::::::::::::::::::::::::::::::::::
