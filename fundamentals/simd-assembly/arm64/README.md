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
