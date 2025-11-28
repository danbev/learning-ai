### argsort
ggml_argsort returns the indices, not the sorted values themselves!

For example::
```console
mask:  {1, 0, 1, 1, 0, 1, 0}  (values at each vocab position)
       [0, 1, 2, 3, 4, 5, 6]  (vocab indices)
```
If we perform the operation:
```c++
  ggml_argsort(mask, DESC) returns: {0, 2, 3, 5, 1, 4, 6}
```
This is saying:
"The indices that would sort the mask in descending order are: 0, 2, 3, 5, ...".
Those numbers are the original vocabulary positions!

So:
```console
Index 0 (had mask value 1)
Index 2 (had mask value 1)
Index 3 (had mask value 1)
Index 5 (had mask value 1)
Index 1 (had mask value 0)
Index 4 (had mask value 0)
Index 6 (had mask value 0)
```


```c++
ggml_data->logits = ggml_get_rows(ctx, logits_rows, mask_idxs);
```
We get back all 32000 logit values in a different order. The first N will be the
ones we want to keep, followed by 32000-N unwanted values.

