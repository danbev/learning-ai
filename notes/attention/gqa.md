### Grouped Query Attention (GQA)
There is also a grouped query attention (GQA) where we have the same number of
query heads but we group them, and each group has its own key and value matrix.:
```
     +-----+   +-----+          +-----+   +-----+
     | Q'_1|   | Q'_2|          | Q'_3|   | Q'_4|
     |     |   |     |          |     |   |     |
     |     |   |     |          |     |   |     |
     |     |   |     |          |     |   |     | 
     +-----+   +-----+          +-----+   +-----+

          +-----+                     +-----+
          | K   |                     | K   |
          |     |                     |     |
          |     |                     |     |
          |     |                     |     |
          +-----+                     +-----+

          +-----+                     +-----+
          | V   |                     | V   |
          |     |                     |     |
          |     |                     |     |
          |     |                     |     |
          +-----+                     +-----+
```
Notice that GQA is a generalization of MHA and MQA:
* MQA is a special case of GQA where the number of groups is equal to 1.
* MHA is a special case of GQA where the number of groups is equal to the
  number of heads, for example 4 heads and 4 groups.

If we have 8 heads we can then have the following groups:
```
h = 8
8 groups, 4 groups, 2 groups, 1 group
h/n
8/1 = 8
8/2 = 4
8/4 = 2
8/8 = 1
```
