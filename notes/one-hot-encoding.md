## One-hot encoding
Lets say you have 10 categories which we are interested in, these could be 10
categories of the contents of images. For example the CIFAR-10 data set has
the following 10 categories:
```
0 Airplane
1 Automobile
2 Bird
3 Cat 
4 Deer
5 Dog
6 Frog
7 Horse
8 Ship
9 Truck
```
These don't really have a natural number representation, so to represent the
Cat we use a vector of 10 elemenet, where the Cat is "turned on" and the rest
are "off":
```
[0 0 0 1 0 0 0 0 0 0]
```
The indexes represent the categories.
