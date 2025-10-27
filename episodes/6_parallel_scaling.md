---
title: "Measuring and improving parallel performance"
teaching: 0 # teaching time in minutes
exercises: 0 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- Why is it important for code to scale well on large HPC systems?
- What can go wrong if we try to optimise too early in the development process?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe why code scalability is important when using HPC resources
- Describe the differences between strong and weak scaling
- Summarise the dangers of premature optimisation

::::::::::::::::::::::::::::::::::::::::::::::::

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

## What is scalability?

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

## The dangers of premature optimisation

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
