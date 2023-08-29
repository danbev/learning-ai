## Vector databases
These are databases that store vector [embeddings](./embeddings.md) of
words/tokens. So these store vectors of numbers, floats that represent these
tokens. And tokens that are similar will have similar vectors and hence will be
closer to each other in the vector space.

So lets say we want to search for something in one of these vector databases.
We would take our query and convert it into an embedding. Then we would search
for the closest vectors to this query vector. And the closest vectors will be
the most similar to the query.
```
                                                     Vector space
 +----------+    +-----------+    +----------+      *   *     ***  *
 | query    | -> | embedding | -> | database |    *   *   *   *  ***  *
 +----------+    +-----------+    +----------+     ***   *x*   *** ** ***
                    |                                     ↑
                  q[...]----------------------------------+
                                                               
```
This is trying to illustrate that the query is converted into an embedding and
then we search for the closest vectors to this query vector. And the closest
vectors will be the most similar to the query which are then returned.

### Embeddings
Lets see what an embedding looks like we can do this in this document by adding
the following comment:
```
# Can  you show me an example of creating an embedding in Python using Huggingface?
```
And the response from copilot was the following:
```
Sure, lets say we want to create an embedding for the word "cat" using
Huggingface. We would do the following:
```

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("bert-base-cased")
input_ids = tokenizer("cat", return_tensors="pt")["input_ids"]
embeddings = model(input_ids)[0]
```
(I added a print statement to print the embeddings at the end.

We can run this using:
```console
$ python3 -i embeddings/python/src/from-copilot.py  
```


Could we perhaps have a database of embeddings for all security adivories and
then when we get a new security advisory we can convert it into an embedding and
then search for the closest vectors to this query vector.

### Searching
So we have a vector database and we want to search it. So we have a query and
we want to find the closest vectors to this query vector.

#### KNN (K-Nearest Neighbors)
This is a way of searching for the closest vectors to a query vector. So we
have a query vector and we want to find the closest vectors to this query.
This is done by calculating the distance between the query vector and all the
vectors in the database. Notice that this says `all` the vectors in the
database. So this is not very efficient. 

#### ANN (Approximate Nearest Neighbors)
This is a another way of searching for the closest vectors to a query vector.





## Qdrant (Rust)
Qdrant is a vector database written in Rust. It is a vector database that is
open source and is written in Rust.

