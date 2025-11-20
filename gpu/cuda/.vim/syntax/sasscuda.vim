" SUPER LOUD CUDA SASS SYNTAX TEST

if exists("b:current_syntax")
  finish
endif

" Comments like // something
syntax match cudasassComment "//.*$"
hi! link cudasassComment Comment

" Common directives
syntax match cudasassDir "\.\(header\|target\|address_size\|maxnreg\|minnctapersm\)"
hi! link cudasassDir PreProc

" Instructions (sample set – extend as you like)
syntax keyword cudasassInstr MOV LDG STG LDS STS FFMA IMAD IADD SHF LOP3 BRA RET
hi! link cudasassInstr Statement

" Registers: R0, R12, etc.
syntax match cudasassReg "\<R[0-9]\+\>"
hi! link cudasassReg Constant

" Predicates: P0, !P0, @P0, @!P0
syntax match cudasassPred "@\?!\=P[0-9]\+"
hi! link cudasassPred Identifier

let b:current_syntax = "cudasass"
