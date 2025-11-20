" SUPER LOUD PTX SYNTAX TEST

if exists("b:current_syntax")
  finish
endif

" Make comments obvious
syntax match ptxComment "//.*$"
hi! link ptxComment Comment

" Make .entry and .func bright
syntax match ptxEntry "\.entry"
syntax match ptxFunc  "\.func"
hi! link ptxEntry Identifier
hi! link ptxFunc  Function

" Make registers and types stand out
syntax match ptxReg "%[a-zA-Z0-9_]\+"
syntax match ptxType "\.\(u32\|u64\|s32\|s64\|f32\|f64\|pred\)"
hi! link ptxReg  Constant
hi! link ptxType Type

let b:current_syntax = "ptx"

