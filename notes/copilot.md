## Copilot notes
Copilot uses ChatGPT with a model which is trained on GitHub code comments.
The model originally used was [openai-codex](https://openai.com/blog/openai-codex).

The training of this model in simple terms the model is passed partial code
snippets/documents and the model is asked to predict the next line of code.

Coplilot also send a context containing data from files that a user has open
which provides more context. This is how it can provide suggestions that are
relevant to the code that we are currently writing which is one of the things
that really surprised me about and I was not expecting.

### Installation
Follow the instructions in [Getting Started](https://github.com/github/copilot.vim#getting-started).
```bash
$ git clone https://github.com/github/copilot.vim.git \
  ~/.vim/pack/github/start/copilot.vim
```
To update the plugin:
```
$ cd  ~/.vim/pack/github/start/copilot.vim/
$ git pull
```
And then restart vim.

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
We can prompt copilot almost like we can in a chatgpt prompt but that was not
obvious to me initially.
We can try that here in markdown too...

```
# Explain the embeddings are in the context of neural networks
Embeddings are a way to represent words as vectors in a vector space.
The idea is that words that are similar in meaning will be close to each other
in the vector space.
...
```
In the same way we can get explaination api's 
# Show and example of using the github api to get the number of star for a Repo in Rust
```rust
use reqwest::blocking::Client;
use serde::Deserialize;

#[derive(Deserialize)]
struct Repo {
    name: String,
    stargazers_count: u32,
}
    
fn main() {
    let client = Client::new();
    let url = "https://api.github.com/repos/rust-lang/rust";
    let repo: Repo = client.get(url).send().unwrap().json().unwrap();
    println!("{} has {} stars!", repo.name, repo.stargazers_count);
}
``` 
