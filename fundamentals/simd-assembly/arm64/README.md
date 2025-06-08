## SIMD arm64 Assembly Language exploration
This directory contains assembly language code examples that show how SIMD
instructions work on arm64. 

### ARM 64 Addressing
Something that was new to me is how addressing works in ARM 64 assembly language
compared to arm IoT development that I've done:
```assembly
    adrp x0, vec1@PAGE                        // Get page address of vec1
```
So 64 bit addressing can use linker directives.

So arm64 used a page-based addressing model where memory is divided into
4KB pages.

```console
$ make fadd
clang -g -O0 -g -o bin/fadd src/fadd.s

$ lldb bin/fadd
(lldb) target create "bin/fadd"
Current executable set to '/Users/danbev/work/ai/learning-ai/fundamentals/simd-assembly/arm64/bin/fadd' (arm64).
(lldb) br set -n main                                                                        Breakpoint 1: where = fadd`main, address = 0x0000000100000370

```
Now, if we run this we will see the action instructions and the arguments
that are used:
```console
(lldb) r
Process 38973 launched: '/Users/danbev/work/ai/learning-ai/fundamentals/simd-assembly/arm64/bin/fadd' (arm64)
Process 38973 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x0000000100000370 fadd`main
fadd`main:
->  0x100000370 <+0>:  adrp   x0, 4
    0x100000374 <+4>:  add    x0, x0, #0x0 ; vec1
    0x100000378 <+8>:  ld1.4s { v0 }, [x0]
    0x10000037c <+12>: adrp   x1, 4
Target 0: (fadd) stopped.
```
So adrp x0, vec1@PAGE has bee re replaced with adrp x0, 4. So 4 is the page number
relative to the current instruction.
```console
(lldb) image lookup -n main
1 match found in /Users/danbev/work/ai/learning-ai/fundamentals/simd-assembly/arm64/bin/fadd:
        Address: fadd[0x0000000100000370] (fadd.__TEXT.__text + 0)
        Summary: fadd`main
```
Now, we mentioned that pages are 4KB, that is 4096 bytes, and 4096 = 2^12. The offset
of an address is the last 12 bits of the address, so if we stip off the last 12 bits
we can get the base address of the page:
```console
(lldb) p/x 0x100000370 & ~0xfff
(long) 0x0000000100000000
```
So this is the base address of the page, and we can see that the page is 4KB.

Now, in our example we have:
```console
.section __DATA,__data
    .align 4                         // 4 bytes
    vec1: .float 1.0, 2.0, 3.0, 4.0
```
And when we inspect the address of `vec1` we see:
```console
(lldb) p &vec1
(void **) 0x0000000100004000
```
So just to clarify this. We first have our text segment:
```
Page 0: 0x100000000 - 0x100000FFF (4KB)
Page 1: 0x100001000 - 0x100001FFF (4KB)
Page 2: 0x100002000 - 0x100002FFF (4KB)
Page 3: 0x100003000 - 0x100003FFF (4KB)
```
This is where our `main` function is located.
We can inspect this memory location using:
```console
(lldb) memory read -f i -c 10 0x100000370
->  0x100000370: adrp   x0, 4
    0x100000374: add    x0, x0, #0x0 ; vec1
    0x100000378: ld1.4s { v0 }, [x0]
    0x10000037c: adrp   x1, 4
    0x100000380: add    x1, x1, #0x10 ; vec2
    0x100000384: ld1.4s { v1 }, [x1]
    0x100000388: fadd.4s v0, v0, v1
    0x10000038c: adrp   x2, 4
    0x100000390: add    x2, x2, #0x20 ; result
    0x100000394: st1.4s { v0 }, [x2]
```

Then we have our data segment:
```console
Page 4: 0x100004000 - 0x100004FFF (4KB)
```
And this is where our vectors are stored:
```
vec1   at 0x100004000
vec2   at 0x100004010
result at 0x100004020
```
We can inspect `vec1` using:
```console
(lldb) memory read -f f -c 4 0x100004000
0x100004000: 1
0x100004004: 2
0x100004008: 3
0x10000400c: 4
```

So in summary when we see:
```assembly
    adrp x0, vec1@PAGE         // Get page address of vec1
    add x0, x0, vec1@PAGEOFF   // Add page offset to get full address
    ld1 {v0.4s}, [x0]          // Load 4 floats from vec1 into v0
```
The @PAGE directive is getting the 4KB page number that contains the target `vec1`.
And the `adrp` instruction is what calculates/resolves the address to the base
page address and stores it in `x0`.
And @PAGEOFF is getting the offset within the page and adding this to the `x0`
register to get the full address of `vec1`.

ARM64 instructions are fixed at 32-bit lengths and a 64-bit address cannot be encoded
into the instructions. So this is a way around this limitation and enables addressing
+- 4GB of memory.
