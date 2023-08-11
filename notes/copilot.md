## Copilot notes

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

imap <C-S-Right-]> <Plug>(copilot-next)
imap <C-S-Right-[> <Plug>(copilot-previous)
imap <C-S-Right-Bslash> <Plug>(copilot-suggest)
```

### Commands
Can be found using `:help copilot-commands` and are also available in the
[source code](https://github.com/github/copilot.vim/blob/release/doc/copilot.txt).
