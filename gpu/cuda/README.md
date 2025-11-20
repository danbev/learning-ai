### CUDA examples


### ptx and sass syntax highlighting
Add the following to .vimrc:
```
au BufNewFile,BufRead *.ptx  set filetype=ptx
au BufNewFile,BufRead *.sass set filetype=sasscuda
```
And then copy the two file in .vim/syntax to `~/.vim/syntax/`.
