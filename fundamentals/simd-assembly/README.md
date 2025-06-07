## SIMD Assembly Language
This directory contains assembly language code examples that show how SIMD
instructions work. 

### Inspecting registers
The following examples is using the [vector_add.s](src/vector_add.s) example:
```console
(gdb) set print pretty

(gdb) p $xmm0
$3 = {
  v8_bfloat16 = {0, 6, 0, 8, 0, 10, 0, 12},
  v8_half = {0, 2.375, 0, 2.5, 0, 2.5625, 0, 2.625},
  v4_float = {6, 8, 10, 12},
  v2_double = {131072.03161621094, 2097152.5087890625},
  v16_int8 = {0, 0, -64, 64, 0, 0, 0, 65, 0, 0, 32, 65, 0, 0, 64, 65},
  v8_int16 = {0, 16576, 0, 16640, 0, 16672, 0, 16704},
  v4_int32 = {1086324736, 1090519040, 1092616192, 1094713344},
  v2_int64 = {4683743613551640576, 4701758012067414016},
  uint128 = 86732126745120971976272251897545490432
}
````
