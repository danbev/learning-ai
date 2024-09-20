## ggml optimizers
This document describes the ggml optimizers that are currently available in ggml.
At the time of this writing there is Adam and LBFGS.

### initialization
There are a number of configuration options for the different type of optimizers.
A default params can be constructed by calling `ggml_opt_default_params` with the type
of optimizer you want to use:
```c
  struct ggml_opt_params opts = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
```
This will return a default set of parameters for the Adam optimizer:
```console
(ggml_opt_params) {
  type = GGML_OPT_TYPE_ADAM
  graph_size = 2048
  n_threads = 1
  past = 0
  delta = 0.00000999999974
  max_no_improvement = 100
  print_forward_graph = true
  print_backward_graph = true
  n_gradient_accumulation = 1
  adam = {
    n_iter = 10000
    sched = 1
    decay = 0
    decay_min_ndim = 2
    alpha = 0.00100000005
    beta1 = 0.899999976
    beta2 = 0.999000012
    eps = 9.99999993E-9
    eps_f = 0.00000999999974
    eps_g = 0.00100000005
    gclip = 0
  }
  lbfgs = {
    m = 0
    n_iter = 0
    max_linesearch = 0
    eps = 0
    ftol = 0
    wolfe = 0
    min_step = 0
    max_step = 0
    linesearch = GGML_LINESEARCH_BACKTRACKING_ARMIJO
  }
}
```
