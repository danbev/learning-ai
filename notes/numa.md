## Non-Uniform Memory Access (NUMA)
In a NUMA system, memory and processors are organized into nodes. Each node
contains one or more processors and a portion of the system's total memory. 

On my Linux machine I have:
```console
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-15
```
So this is bascially a a Uniform Memory Access system (UMA).

In ggml-cpu we have the following:
```c++
    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };
```
`GGML_NUMA_STRATEGY_DISTRIBUTE` spreads the workload and memory across all
available NUMA nodes.

`GGML_NUMA_STRATEGY_ISOLATE` confines/limits the entire workload to a single
NUMA node.

`GGML_NUMA_STRATEGY_NUMACTL` uses the system's numactl policies for memory and
CPU placement.
```console
$ numactl --show
policy: default
preferred node: current
physcpubind: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
cpubind: 0
nodebind: 0
membind: 0
preferred:

$ numactl -H
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
node 0 size: 31669 MB
node 0 free: 3358 MB
node distances:
node   0
  0:  10
```

`GGML_NUMA_STRATEGY_MIRROR` replicates the same data across multiple NUMA nodes.
