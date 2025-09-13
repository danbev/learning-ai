## GPU Memory

## RTX-4070 Memory
My GPU has 6 GDDR6X which each can store 2 GB of data, giving a total of 12 GB
of memory:
```
[Mem2] [Mem5]    [Mem6]
   \      |     /
    \     |    /
     [  GPU   ]
    /     |    \
   /      |     \
[Mem1] [Mem3]    [Mem4]
```
Each memory chip as a controller associated with it that exists in/on the GPU
die itself:
```
GPU Die (AD104)

    ┌───────────────────────────────────────┐
    │  [MC0] [MC1] [MC2] [MC3] [MC4] [MC5]  │ ← Memory Controllers
    └────┬────┬────┬────┬────┬────┬─────┬───┘
         │    │    │    │    │    │     │   ~80 connections to each memory chip
         │    │    │    │    │    │     │
    [GDDR6X][GDDR6X][GDDR6X][GDDR6X][GDDR6X][GDDR6X]
     chip1   chip2   chip3   chip4   chip5   chip6
      2GB    2GB      2GB     2GB     2GB    2GB
```
Now, each SM contains:
```
SM_0
├── 128 CUDA Cores (ALU units)
├── 4 Warp Schedulers
├── 1 LSU (Load/Store Unit) that handles:
│   ├── Global memory loads/stores
│   ├── Shared memory access
│   ├── Texture cache access
│   ├── Constant cache access
│   └── Local memory (spilled registers)
└── Various caches (L1, texture, etc.)
...

```

The Memory Management Unit (MMU) is a component on the GPU die. This translated
virtual addresses used by programs into physical addresses used by the hardware.
```
MMU translates virtual → physical:
Virtual:  0x7F8A12345678
          ↓ (page table lookup)
Physical: 0x312345678

GPU Memory Layout:
├── 0x000000000 - 0x07FFFFFFF → Memory Controller 0 (2GB)
├── 0x080000000 - 0x0FFFFFFFF → Memory Controller 1 (2GB)
├── 0x100000000 - 0x17FFFFFFF → Memory Controller 2 (2GB)
├── 0x180000000 - 0x1FFFFFFFF → Memory Controller 3 (2GB)
├── 0x200000000 - 0x27FFFFFFF → Memory Controller 4 (2GB)
└── 0x280000000 - 0x2FFFFFFFF → Memory Controller 5 (2GB)
```

On an SM:
```
Warp Scheduler on SM15:
├── Fetches instruction from SM's I-Cache: "LD.E R1, [R2+offset]"
├── Decodes instruction type: Global memory load
├── Reads source register R2: contains virtual address 0x7F8A12345678
├── Calculates effective address: R2 + offset = 0x7F8A12345680
└── Identifies this needs LSU handling
```
Warp Scheduler passes to SM's LSU:
```
├── Virtual address: 0x7F8A12345680
├── Thread mask: which threads in warp need data
├── Instruction details: destination register R1
├── Request ID: for tracking completion
└── Warp ID: so LSU knows who to notify when done
```

The LSU (also on the SM) does the following, it receives the virtual address
0x7F8A12345680:
```
Option A - TLB Hit (Fast Path):
├── LSU checks its local TLB (Translation Lookaside Buffer)
├── TLB contains cached translation: 0x7F8A12345680 → 0x312345680  
└── LSU now has physical address immediately
```
```
Option B - TLB Miss (Slow Path):  
├── LSU checks TLB: MISS
├── LSU sends translation request to central MMU unit
├── MMU performs page table walk in GPU memory
├── MMU returns: 0x7F8A12345680 → 0x312345680
├── LSU caches result in its TLB for future use
└── LSU now has physical address
```

Back in the LSU with the physical addess the LSU routes to the correct
memory controller (which remember is on the GPU die itself):
```
LSU with physical address 0x312345680:
├── Examines address bits [31:29] = 001
├── Determines: Route to Memory Controller 1
├── Strips off controller-selection bits
├── Sends request to MC1 with address 0x12345680
└── Adds request to MC1's queue

Address Routing Logic:
if (addr[31:29] == 0b000) → Route to MC0
if (addr[31:29] == 0b001) → Route to MC1  ← Our example
if (addr[31:29] == 0b010) → Route to MC2
if (addr[31:29] == 0b011) → Route to MC3
if (addr[31:29] == 0b100) → Route to MC4
if (addr[31:29] == 0b101) → Route to MC5
```

Memory controller (on GPU):
```
Memory Controller 1 receives request:
├── Address within MC1's space: 0x12345680
├── Decodes GDDR6X addressing:
│   ├── Bank:   addr[26:24] = 4
│   ├── Row:    addr[23:10] = 0x1234  
│   └── Column: addr[9:0] = 0x680
├── Checks for bank conflicts
├── Schedules DRAM commands: ACTIVATE → READ → PRECHARGE
└── Drives signals to GDDR6X Chip 1
```

GDDR6X Chip 1:
```
├── Receives ACTIVATE command for Row 0x1234, Bank 4
├── Opens row into internal sense amplifiers
├── Receives READ command for Column 0x680
├── Returns 32 bits of data on DQ pins
└── MC1 receives data and sends back to requesting LSU
```

Data return path:
```
GDDR6X → MC1 → LSU → Warp Scheduler → Write to register R1
```


Each memory controller in the GPU connects to each GDDR6X chip through around
80 connections.
```
Data Signals (32 bits):
├── DQ0-DQ31        (32 pins) ← Bidirectional data

Address Signals (~14 bits):
├── A0-A13          (14 pins) ← Row/Column addresses

Bank/Control Signals:
├── BA0-BA2         (3 pins)  ← Bank address
├── BG0-BG1         (2 pins)  ← Bank group address
├── RAS#            (1 pin)   ← Row Address Strobe
├── CAS#            (1 pin)   ← Column Address Strobe
├── WE#             (1 pin)   ← Write Enable

Clock Signals:
├── CK              (1 pin)   ← Clock positive
├── CK#             (1 pin)   ← Clock negative (differential)

Chip Control:
├── CS#             (1 pin)   ← Chip Select
├── CKE             (1 pin)   ← Clock Enable
├── ODT             (1 pin)   ← On-Die Termination
├── RESET#          (1 pin)   ← Reset signal

Data Strobes (for timing):
├── DQS0, DQS0#     (2 pins)  ← Data strobe for DQ0-7
├── DQS1, DQS1#     (2 pins)  ← Data strobe for DQ8-15
├── DQS2, DQS2#     (2 pins)  ← Data strobe for DQ16-23
├── DQS3, DQS3#     (2 pins)  ← Data strobe for DQ24-31

Power/Ground:
├── VDD, VDDQ       (~20 pins) ← Multiple power supplies
├── VSS, VSSQ       (~20 pins) ← Multiple ground connections

```
```
GPU Die Memory Controller 0:
┌─────────────────────────┐
│ Scheduler & Logic       │
│                         │
│ PHY (Physical Layer):   │
│ ├── 32 Data Drivers     │──── DQ0-DQ31 ────┐
│ ├── 14 Address Drivers  │──── A0-A13   ────┤
│ ├── 8 Control Drivers   │──── RAS/CAS/etc ─┤
│ ├── 2 Clock Drivers     │──── CK/CK#   ────┤    GDDR6X
│ ├── 8 Strobe Drivers    │──── DQS0-3   ────┤    Chip 0
│ └── Power Management    │──── VDD/VSS  ────┤    (2GB)
└─────────────────────────┘                  │
                                             │
                            80+ PCB traces   │
                            connecting ──────┘
                            ```
```
So each of the 6 GDDR6X chips has 2 GB of memory, and the memory cells are
organized into banks. Each chip typically has a 32-bit interface, meaning it can
read or write 32 bits (4 bytes) of data in a single operation. This is done by
using a shared data bus, 6*32 = 192 bits (24 bytes).

There are 16 banks each organized into 4 bank groups.
```
 +-------------+ +-------------+ +-------------+ +-------------+
 | Bank Group 0| | Bank Group 1| | Bank Group 2| | Bank Group 3|
 |-------------+ +-------------+ +-------------+ +-------------+
 | Bank 0      | | Bank 0      | | Bank 0      | | Bank 0      |
 +-------------+ +-------------+ +-------------+ +-------------+
 | Bank 1      | | Bank 1      | | Bank 1      | | Bank 1      |
 +-------------+ +-------------+ +-------------+ +-------------+
 | Bank 2      | | Bank 2      | | Bank 2      | | Bank 2      |
 +-------------+ +-------------+ +-------------+ +-------------+
 | Bank 3      | | Bank 3      | | Bank 3      | | Bank 3      |
 +-------------+ +-------------+ +-------------+ +-------------+
```
Bank:
```
  Each Bank = Array of Memory Cells
  ┌─────────────────────────────────────┐
  │ Row 0:  [cell][cell][cell]...[cell] │ ← 1 row = multiple columns
  │ Row 1:  [cell][cell][cell]...[cell] │
  │ Row 2:  [cell][cell][cell]...[cell] │
  │   ...                               │
  │ Row N:  [cell][cell][cell]...[cell] │
  └─────────────────────────────────────┘
```
So each memory chip has a capacity of storing 2GB chip, and we have 16 banks
that gives us 2GB / 16 banks = 128MB per bank.
```
128MB * 8 = 1,073,741,824 bits per bank
Rows    = 16384
Columns = 16384
16384 rows * 16384 columns * 8 bits = 1,073,741,824 bits = 128MB
```

We a flow that looks something like this:
```console
GPU Core → LSU → MMU → Address Routing → Memory Controller
```
The warp schduler decodes a fetched instruction and if it is an load/store
instruction it send it to the LSU includuing the virtual address. The LSU then
tries to use its TLB to translate the virtual address to a physical and if it
does not exist in the TLB it sends a request to the MMU. The MMU then translates
the virtual address to a physical address, and maps this to the correct memory
controller for the physical address. The memory controller then decodes the
the address to figure out which bank group (2 bits for 4 groups), and which bank
(also 2 bits for 4 banks per group) to use, and then which row and column.
The memory controller send the commands (bank/row activation, column read/write,)
over the 32 bit interface using PAM4 signaling.

* Each of the 6 memory controllers connects to exactly 1 GDDR6X chip
* Each chip gets its own dedicated 32-bit data bus + control signals
* No shared buses or chip select lines needed


* Each GDDR6X chip: 32-bit interface = 4 bytes per operation
* Each chip can read OR write (not both simultaneously)
* 6 chips × 4 bytes = 24 bytes per operation

```
32-bit Address: [Bank][Row Address][Column Address][Byte Select]
                  3 bits   14 bits     10 bits      3 bits
                    ↓        ↓           ↓           ↓
                  Bank    Word Line   Bit Lines   Which byte
                Selector   Select     Select     in the word
```

```
Column 0  Column 1  Column 2  Column 3
Row 0:    [C]      [C]       [C]       [C]     ← Word Line 0
Row 1:    [C]      [C]       [C]       [C]     ← Word Line 1
Row 2:    [C]      [C]       [C]       [C]     ← Word Line 2
Row 3:    [C]      [C]       [C]       [C]     ← Word Line 3
          |        |         |         |
       Bit Line Bit Line Bit Line Bit Line
         0        1         2         3
```

GDDR6X uses PAM4 signaling which happens between the memory controller and
the memory chips. PAM4 uses 4 voltage levels to encode 2 bits per symbol, which
effectively doubles the data rate compared to traditional binary signaling
(2 levels, 1 bit per symbol). This allows for higher data throughput without
increasing the frequency of the signal.
