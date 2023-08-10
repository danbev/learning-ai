## XOR Problem 
This is an issue that was discoved with a single layer perceptron.
If we take a look at the truth table for XOR:
```
| A | B | A XOR B |
|---|---|---------|
| 0 | 0 | 0       |
| 0 | 1 | 1       |
| 1 | 0 | 1       |
| 1 | 1 | 0       |
```
If we plot this on a graph we get:
```
        ^
        |
  (0,1) *           *(1.1)
 class 1|            class 0
        |
        |
        *-----------*------>
      (0,0)          (1,0) 
      class 0        class 1
```
And the output of the perceptron will be:
```
 y = f(w1 * x1 + w2 * x2 + b)
```
This will be a linear function and will not be able to separate the two classes.
We can try this by trying to draw a line in the graph above that separates the
two classes. We can do it with a single line but we can do it with two lines
which would mean two perceptrons.
```
        ^
        |        \
  (0,1) *         \*(1.1)
        |  class 1 \ class 0
        |\          \
        | \          \
        *--\--------*------>
      (0,0) \         (1,0) 
      class 0\       
```
