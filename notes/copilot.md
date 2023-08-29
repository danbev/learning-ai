## Copilot notes
Copilot uses ChatGPT with a model which is trained on GitHub code comments.
The model originally used was [openai-codex](https://openai.com/blog/openai-codex).

The training of this model is in simple terms the model is passed partial code
snippets/documents and the model is asked to predict the next line of code.
Coplilot also send a context containing data from files that a user has open
which provides more context. This is how it can provide suggestions that are
relevant to the code that we are currently writing which is one of the things
that really surprised me about and I was not expecting.

### Installation
Follow the instructions in [Getting Started](https://github.com/github/copilot.vim#getting-started).
```bash

### Vim configuration
```vim
let g:copilot_filetypes = {
    \ 'gitcommit': v:true,
    \ 'markdown': v:true,
    \ 'yaml': v:true
    \ }

imap <C-p> <Plug>(copilot-next)
imap <C-n> <Plug>(copilot-previous)
imap <C-s> <Plug>(copilot-suggest)
"nmap <C-p> :Copilot split<CR>
"imap <silent> <C-o> :Copilot split<CR>
inoremap <C-o> <Esc>:Copilot split<CR>i
```

### Commands
Can be found using `:help copilot-commands` and are also available in the
[source code](https://github.com/github/copilot.vim/blob/release/doc/copilot.txt).

### Prompting for help

