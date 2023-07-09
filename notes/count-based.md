## Count-based vector encoding
Is also a method of converting test into vectors of numbers, called vector
embeddings. 

In this case en entire sentence is represented with a single vector. The are
multple ways of doing this. 

### Bag of Words

```
Text:
A wizard is never late. He arrives precisely when he means to.

A wizard is never late: 
  A   He  Is When  Never  To  Means ...
[ 1   0   1   0     1      0   0    ...]

                          
He arrives precisely when he means to: 
  A   He  Is When  Never  To  Means ...
[ 0   2   0   1     0      1   1    ...]

```
