## Hugging face notes
Note about Hugging Face libraries.

### Accellerate
Is a library which provides distributed learning, which is when you train a
model accross multiple GPUs or nodes.

It also provides mixed precision training which is when you use 16-bit
and 32-bit precision arithmetic.

And it has a data parallelism feature which is when you split the data across
multiple GPUs by dividing the data into batches and then each GPU processes
a batch. Note that this is about data and note the same as the distrubuted
training.

### Models
To add a new model to huggingface.co you first create a new model using their
UI and then you can use normal git command to push.

### Enable files larger than 5GB
To enable files larger than 5GB you need to use the `lfs-enable-largefiles`:
```console
$ huggingface-cli lfs-enable-largefiles . 
```
Also you need to have the `git-lfs` installed.
```console
$ git lfs install
$ git lfs track "*.gguf"
```

### Configure ssh
To configure ssh you need to create a `~/.ssh/config` file with the following:
```console
Host hf.co
    HostName hf.co
    User git
    IdentityFile ~/.ssh/id_file
```
After this you should be able to authenicate to hf:
```console
$ ssh -T git@hf.co
```
And we can update the git remote to use ssh:
```console
$ git remote add origin git@hf.co:danbev/test-model
```
