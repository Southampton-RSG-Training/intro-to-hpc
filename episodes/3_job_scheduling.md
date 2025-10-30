---
title: "Introduction to Job Scheduling"
teaching: 0 # teaching time in minutes
exercises: 0 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- What is a job scheduler and why is it needed?
- What is the difference between a login node and a compute node?
- How can I see the available resources and queues?
- What is a job submission script?
- How do I submit, monitor, and cancel a job?
- How (and when) should I use an interactive job?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Describe briefly what a job scheduler does
- Summarise how to query the available resources on an HPC system
- Describe a minimal job submission script and parameters that need to be specified
- Summarise how to submit a batch job and monitor it until completion
- Contrast when to run programs on an HPC login node vs running them on a compute node
- Summarise the process for requesting and using an interactive job

::::::::::::::::::::::::::::::::::::::::::::::::

An HPC cluster has thousands of nodes shared by many users. A job scheduler is the software that manages this, deciding
who gets what resources and when. It ensures that tasks run efficiently and fairly, matching a job's resource request to
available hardware. On Iridis, the scheduler is Slurm, but the concepts are transferable to other schedulers such as
PBS.

![Queueing up to eat at a popular restaurant is like queueing up to run something on an HPC
cluster.](fig/restaurant_queue_manager.svg)

The scheduler acts like a manager at a popular restaurant. You must queue to get in, and you must wait for a table
(compute resources) to become free. This is why your jobs may sit in a queue before they start, unlike on your computer.
In this episode, we'll look at what a job scheduler is and how you interact with it to get your jobs running on Iridis.

## What can I do on the login nodes?

The login nodes are the gateway to any cluster. On Iridis, they are intended only for light and short tasks such as
editing files, managing data, compiling code and submitting/monitoring running jobs.

You must not run computationally intensive or long-running tasks on them. Login nodes are a shared resource for all
users to access the system. Running intensive jobs on them slows the system down for everyone. Any such process will be
ended automatically, and repeated misuse may lead to your access to Iridis being restricted. To enforce this, login
nodes have strict resource limits. You are limited to 64 GB of RAM and 2 CPUs (Iridis X also provides an NVIDIA L4 GPU).

All computationally intensive work must be submitted to the job scheduler. This places your job in a queue, and Slurm
will allocate dedicated compute node resources to it when they become available. If your task requires more power than
the login node provides (e.g., compiling large, complex code or transferring huge data files), you must perform it on a
compute node. You can do this by submitting a job or by starting an interactive session, both of which we will cover
later.

## Querying available resources

Compute nodes in Slurm are grouped together and organised into different **partitions**. You can think of a partition as
the actual job queue for a certain set of hardware. Clusters are often made up of different types of compute nodes, e.g.
some with lots of memory, some with GPUs, some with restricted access, and some that are just standard compute nodes.
The partitions are how Slurm organises this hardware. Each partition can have its own rules, such as a maximum run time
or a limit on the number of nodes you can use. To find what the partitions are on Iridis 6 and their current state, we
can use the `sinfo` command:

```bash
[iridis6]$ sinfo -s
PARTITION             AVAIL  TIMELIMIT   NODES(A/I/O/T) NODELIST
batch*                   up 2-12:00:00      99/35/0/134 red[6001-6134]
highmem                  up 2-12:00:00          3/1/0/4 gold[6001-6004]
worldpop                 up 2-12:00:00          0/6/0/6 red[6135-6140]
scavenger                up   12:00:00          0/6/0/6 red[6135-6140]
interactive_practical    up   12:00:00          1/0/0/1 red6128
```

The `-s` flag outputs a summarised version of this list. Omitting this flag provides a full listing of nodes in each
queue and their current state, which gets quite messy.

We can see the general availability of each partition/queue, as well as the maximum time limit for jobs (in
days-hours:minutes:seconds format). For example, on the batch queue there is a 2 and a half day limit, whilst the
scavenger queue has a 12 hour limit. The * appended to the batch partition name indicates it is the preferred default
queue. The NODES column indicates the number of nodes in a given state,

| Label | State  | Description                                    |
|-------|--------|------------------------------------------------|
| A     | Active | These nodes are busy running jobs              |
| I     | Idle   | These nodes are not running jobs               |
| O     | Other  | These nodes are down, or otherwise unavailable |
| T     | Total  | The total number of nodes in the partition     |

Finally, the NODELIST column is a summary of the nodes belonging to a particular queue; if we didn't use the `-s`
option, we could get a complete list of each node in each state. In this particular instance, we can see that 35 nodes
are idle in the batch partition, so if that queue fits our needs (and we have access to it) we may decide to submit to
that as there are available resources.

We can find out more details about specific partitions by using the `scontrol show` command, which lets us view more
configuration details of a particular partition. To see the breakdown of the batch partition, we use:

```bash
$ scontrol show partition=batch
PartitionName=batch
   AllowGroups=ALL DenyAccounts=worldpop AllowQos=ALL
...
   MaxTime=2-12:00:00 MinNodes=0
...
   State=UP TotalCPUs=25728 TotalNodes=134
...
   DefMemPerCPU=3350 MaxMemPerNode=650000
```

This purposefully truncated output shows who has does and doesn't have access (AllowGroups and DenyAccounts) as well as
details about the configuration and details of the nodes in the partition (e.g. MaxTime, MinNodes, TotalNodes,
TotalCPUs). To get more detail about a particular node in a partition, we use,

```bash
$ scontrol show node=red6001
NodeName=red6001 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=192 CPUEfctv=192 CPUTot=192
...
   RealMemory=770000 AllocMem=643200 FreeMem=531885 Sockets=2
   State=ALLOCATED
...
   Partitions=batch
```

This provides a detailed summary of the node, including the number of CPUs on it (CPUTot), if there are GPUs (Gres) as
well as the state of the node (State), the current resources allocated to a user (CPUAlloc, AllocMem) and other
interesting information.

## Job submission scripts

When we submit a job to run, we have to write a *Slurm script* which contains the commands/programs  we want to run on a
compute node. This is usually a script written in the Bash shell language. A very minimal example looks something like
this.

```bash
#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00

date
```

The `#SBATCH` lines are special comments which provide information about the resources we are requesting for our job to
Slurm. For example the partition we want to run on job on, the maximum time we expect the job to take when running, and
the number of nodes and CPUs we’d like to request (in this case, just one CPU on one node).

| Parameter           | Description                                                              | Example Value        |
|---------------------|--------------------------------------------------------------------------|----------------------|
| `--job-name`        | Sets a name for your job.                                                | `my_analysis`        |
| `--nodes`           | Requests a specific number of compute nodes.                             | `1`                  |
| `--ntasks`          | Requests a total number of tasks (e.g., MPI processes).                  | `16`                 |
| `--ntasks-per-node` | Specifies the number of tasks to run on each node.                       | `8`                  |
| `--cpus-per-task`   | Requests a number of CPU cores for each task (e.g., for OpenMP threads). | `4`                  |
| `--time`            | Sets the maximum wall-clock time for the job (HH:MM:SS).                 | `01:30:00`           |
| `--partition`       | Specifies the queue (partition) to submit the job to.                    | `amd` or `a100`      |
| `--gres`            | Requests generic consumable resources, most commonly GPUs.               | `gpu:1`              |
| `--output`          | Specifies the file to write the standard output (STDOUT) to.             | `job_output.log`     |
| `--error`           | Specifies the file to write the standard error (STDERR) to.              | `job_error.log`      |
| `--mail-user`       | Your email address for job status notifications.                         | `a.user@soton.ac.uk` |
| `--mail-type`       | Specifies which events trigger an email (e.g., BEGIN, END, FAIL, ALL).   | `END,FAIL`           |

### What goes into a job script?

- Pretty much whatever you need to do to run the job

```bash
#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=00:01:00

# Optional: print useful job info
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Load required modules
module purge
module load python/3.11

# Activate Python virtual environment
source ~/myenv/bin/activate

# Optional: set any environment variables or configuration
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Move to job directory (optional)
cd $SLURM_SUBMIT_DIR

# Run the Python script
python my_script.py --input data/input.txt --output results/output.txt
```

## Submitting, monitoring and cancelling jobs

- Use `sbatch` to submit a job
- Returns an job id, which you can use to track the progress and other things
- Use `squeue` to look at jobs, e.g. `squeue -u $USER`
- Jobs can be cancelled using `scancel <jobid>` or `scancel -u $USER` to cancel all jobs

### Submitting jobs

Next, launch our new job:

```bash
$ sbatch example-job.sh
Submitted batch job 715860
```

We can use this job ID to ask Slurm for more information about it:

```bash
scontrol show jobid=715860
JobId=715860 JobName=example.sh
   UserId=ejp1v21(32917) GroupId=fp(245) MCS_label=N/A
   Priority=348672 Nice=0 Account=default QOS=default
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=00:00:09 TimeLimit=00:01:00 TimeMin=N/A
   SubmitTime=2025-10-29T14:58:31 EligibleTime=2025-10-29T14:58:31
   AccrueTime=2025-10-29T14:58:31
   StartTime=2025-10-29T14:58:32 EndTime=2025-10-29T14:59:32 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2025-10-29T14:58:32 Scheduler=Main
   Partition=batch AllocNode:Sid=login6002:1385285
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=red6086
   BatchHost=red6086
   NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   ReqTRES=cpu=1,mem=3350M,node=1,billing=1
   AllocTRES=cpu=1,mem=3350M,node=1,billing=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:1 CoreSpec=*
   MinCPUsNode=1 MinMemoryCPU=3350M MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/iridisfs/home/ejp1v21/example.sh
   WorkDir=/iridisfs/home/ejp1v21
   StdErr=/iridisfs/home/ejp1v21/slurm-715860.out
   StdIn=/dev/null
   StdOut=/iridisfs/home/ejp1v21/slurm-715860.out
```

In particular, we can see:

<!-- The following section below will need updating given the output above -->
As we might expect the JobState is RUNNING, although it may be PENDING if waiting to be assigned to a node, or if we weren’t fast enough running the scontrol command it might be COMPLETED
How long the job has run for (RunTime), and the job’s maximum specified duration (TimeLimit)
The job’s SubmitTime, as well as the job’s StartTime for execution: this may be the actual start time, or the expected start time if set in the future. The expected EndTime is also specified, although if it isn’t specified directly in the job script this isn’t always exactly StartTime + specified duration; it’s often rounded up, perhaps to the nearest minute.
The queue assigned for the job is the devel queue, and that the job is running on the dnode036 node
The resources assigned to the job are a single node (NumNodes=1) with 128 CPU cores, for a single task with 1 CPU core per task. Note that in this case we got more resources in terms of CPUs than what we asked for. For example in this instance, we can see that we actually obtained a node with 128 CPUs (although we won’t use them)
We didn’t specify a working directory within which to execute the job, so the default WorkDir is our home directory
The error and output file locations, as specified by StdErr and StdOut

### Monitoring jobs

As we’ve seen, we can check on our job’s status by using the command squeue. Let’s take a look in more detail.

```bash
squeue -u $USER
```

You may find it looks like this:

  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
5791510 cosma7-pa example- yourUser PD       0:00      1 (Priority)

So -u yourUsername shows us all jobs associated with our machine account. We can also use -j to query specific job IDs, e.g.: squeue -j 5791510 which will, in this case, yield the same information since we only have that job in the queue (if it hasn’t already completed!).

In either case, we can see all the details of our job, including the partition, user, and also the state of the job (in the ST column). In this case, we can see it is in the PD or PENDING state. Typically, a successful job run will go through the following states:

PD - pending: sometimes our jobs might need to wait in a queue first before they can be allocated to a node to run.
R - running: job has an allocation and is currently running
CG - completing: job is in the process of completing
CD - completed: the job is completed
For pending jobs, helpfully, you may see a reason for this in the NODELIST(REASON) column; for example, that the nodes required for the job are down or reserved for other jobs, or that the job is queued behind a higher priority job that is making use of the requested resources. Once it’s able to run, the nodes that have been allocated will be displayed in this column instead.

However, in terms of job states, there are a number of reasons why jobs may end due to a failure or other condition, including:

OOM - ouf of memory: the job attempted to use more memory during execution than what was available
S - suspended: job has an allocation, but it has been suspended to allow for other jobs to use the resources
CA - cancelled: the job was explicitly cancelled, either by the user or system administrator, and may or may not have been started
F - failed: the job has terminated with a non-zero exit code or has failed for another reason
You can get a full list of job status codes via the SLURM documentation.

### Cancelling jobs

Sometimes we’ll make a mistake and need to cancel a job. This can be done with the scancel command. Let’s submit a job and then cancel it using its job number (remember to change the walltime so that it runs long enough for you to cancel it before it is killed!).

```bash
$ sbatch example-job.sh
Submitted batch job 5791551
```

Now cancel the job with its job number (printed in your terminal). A clean return of your command prompt indicates that the request to cancel the job was successful.

```bash
scancel 5791551
```

It might take a minute for the job to disappear from the queue.

## Interactive jobs

On occasions, for instance, when needing to debug on Iridis, it can be useful to start jobs directly from a compute
node. To do so use the `sinteractive` command. By default, this will give a single node for 2 hours, but this can be
changed with the normal flags to `sbatch`. If sufficient resources are available, the interactive job
will start immediately, otherwise, it will still need to queue to start and the terminal will be unavailable. Once the
job has started the user is logged in to a node of the job and they can run commands from there across all the allocated
nodes. By default, the user should also be able to use any GUI interfaces as long as X-forwarding is set up correctly
when the user connected to Iridis (by using the -X SSH flag).

As resources may not be available immediately to satisfy the requirements of an interactive job, it is normally only
practical to use interactive jobs for short jobs of a few hours or less, running on a handful of nodes. For example, a
user may wish to test their applications before submitting a long-running job. Some estimates of what resources are
available can be seen with the `sinfo` command. This will show any idle nodes, along with reserved and allocated nodes.

`sinteractive` is a command line tool that is available on the login nodes.

The most basic usage looks like this:

```bash
sinteractive
```

This will start an interactive session on a serial node with 1 CPU and roughly 20 GB of memory.

You can specify the partition to use with sinteractive by doing:

```bash
sinteractive --partition=<partitionname>
```

Please see our documentation regarding our partitions to select one or run the sinfo command on a login node.

To request more resources like the number of CPUs per task:

```bash
sinteractive --partition=<partitionname>  --cpus-per-task=10
```

All `#SBATCH` flags can be passed at the command line to `sinteractive` which will allow you to customize your
`sinteractive` sessions within the parameters of the settings we have allowed for Slurm.

Common settings that users can apply are to request more time or a custom memory request.

```bash
sinteractive --partition=<partitionname>  --time=<custom_time_value> --mem=<custom_memory_value_in_MB>
```

The default `sinteractive` command in Iridis 5 will assign your interactive job to a gold compute node in the serial
partition as a result. By default, `sinteractive` requests 1 task on 1 node, which by default assigns 1 CPU to that
task. Even if you specify a different partition as directed above, you will be given a node in serial unless you ask for
more than 20 CPUs.

::::::::::::::::::::::::::::::::::::: keypoints

- The job scheduler (like Slurm) manages all user jobs to ensure fair and efficient use of the cluster.
- Login nodes are for light tasks (editing, compiling); compute nodes are for running scheduled, intensive jobs.
- Use `sinfo` and `scontrol` to query the status of partitions (queues) and nodes.
- A job script is a Bash script containing `#SBATCH` directives (resource requests) and the commands to be run.
- Use `sbatch` to submit a job, `squeue` to monitor its status, and `scancel` to cancel it.
- Use `sinteractive` to request a live terminal session on a compute node for debugging or interactive work.

::::::::::::::::::::::::::::::::::::::::::::::::
