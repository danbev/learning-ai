## Convolutional Neural Network (CNN)
Where a Transformer processes data in parallel, and an RNN processes data
sequentially, a CNN processes data in a sliding window. We can think of this
of as a matrix that slides over the input matrix and performs the dot product
for each filter.

So lets say we have our input matrix, which just like in a transformer the rows
representt tokens of the input sequence and the columns are the embeddings:
```
   +--------------+
   |  |  |  |  |  |
   |--------------|
   |  |  |  |  |  |
   |--------------|
   |  |  |  |  |  |
   |--------------|
   |  |  |  |  |  |
   |--------------|
   |  |  |  |  |  |
   |--------------|
   |  |  |  |  |  |
   +--------------+
```
So a Transformer would process this in parallel using matrix multiplication, 
S = softmax(QKᵀ / √d) V, where Q, K, V are the query, key and value matrices.
For a CNN it will slide a smaller matrix over the above matrix and calculate
the dot product for each position. So lets say we have a 3x3 matrix that we
slide over the above matrix:
```
   +--------------+
   |        |  |  |
   |        ------|    C = Convolution/Filter
   |   C    |  |  |        +--+--+--+    +--+--+--+
   |        ------|    C = |  |  |  |    |  |  |  |
   |        |  |  |        |--+--+--|    |--+--+--|    +--+
   |--------------|        |  |  |  | .  |  |  |  | =  |  |
   |  |  |  |  |  |        |--+--+--|    |--+--+--|    +--+
   |--------------|        |  |  |  |    |  |  |  |
   |  |  |  |  |  |        +--------+    +--------+
   |--------------|
   |  |  |  |  |  |
   +--------------+
```
Now, the Convolution/Filter matrix contains weights learned during training and
the dot product is used to calculate the similarity between the two matrices. If
the result is high that means this implies that the filter and that part of the
input are similar.
Next, we slide the filter over with a stride of 1 in this case:
```
   +--------------+
   |  |        |  |
   |---        ---|    C = Convolution/Filter            result of dot product
   |  |   C    |  |        +--+--+--+    +--+--+--+        |
   |---        ---|    C = |  |  |  |    |  |  |  |        ↓
   |  |        |  |        |--+--+--|    |--+--+--|    +--+--+
   |--------------|        |  |  |  | .  |  |  |  | =  |  |  |
   |  |  |  |  |  |        |--+--+--|    |--+--+--|    +--+--+
   |--------------|        |  |  |  |    |  |  |  |     ↑
   |  |  |  |  |  |        +--------+    +--------+     |
   |--------------|                                     |
   |  |  |  |  |  |                                 From previous step
   +--------------+
```
