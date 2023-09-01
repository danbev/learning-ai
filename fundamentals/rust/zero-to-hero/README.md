## Neural Networks:  Zero to Hero
This project is a Rust implementation of the youtube series
[Neural Networks:  Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
by Andrej Karpathy. The goal is to implement the code in the series in Rust.

In the series Mr. Karpathy uses some notebook exploration coding which is very
helpful for learning. I'll try to provide the same thing in the code itself, or 
as debugging sessions where applicable.

## Install
This part is not great but to see plots we depend on Python matplolib and numpy
to be available. This is only for plotting, we will be only be using Rust
libraries for everything else, but visualizing the data is an important part
of the learning process so I wanted to provide something and hope to be
replace it with a pure Rust solution in the future.

```
$ python3 -m penv zeroh
$ source zeroh/bin/activate
(zeroh) $ pip install -r requirements.txt
```
```console
$ sudo dnf install lapack-devel openblas-devel
```

### Part1
This is the first part of the series named
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2).
```console
(zeroh) $ cargo run --bin part1
```
The first plot from into section looks like this:
![image](./plots/part1_intro.svg)

This can also be opened locally using the following command or using a web
browser:
```console
(zeroh) $ xdg-open plots/part1.svg
```
