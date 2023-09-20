## Apache Arrow
Apache Arrow is a cross-language development platform for in-memory data.

### Background
Lets take the following table as an example:
```
       +---------+---------+--------+
       |  City   | Country |  Area  |
       +---------+---------+--------+
row 1  | London  | England | 1572   |
row 2  | Berlin  | Germany | 891.85 |
row 3  | Madrid  | Spain   | 605.77 |
row 4  | Rome    | Italy   | 1285   |
       +---------+---------+--------+
```
Now, if we were to store to store this in memory it would normally be layed
out like this:
```
       +---------+
row 1  |  London |
       |  England|
       |  1572   |
row 2  |  Berlin |
       |  Germany|
       |  891.85 |
row 3  |  Madrid |
       |  Spain  |
       |  605.77 |
row 4  |  Rome   |
       |  Italy  |
       |  1285   |
       +---------+
```
This is not very efficient for modern CPU's or GPU's.

In Apache Arrow this would be layed out in memory like this instead:
```
  +---------+
  | London  |  <- City column
  | Berlin  |  <- City column
  | Madrid  |  <- City column
  | Rome    |  <- City column
  | England |  <- Country column
  | Germany |  <- Country column
  | Spain   |  <- Country column
  | Italy   |  <- Country column
  | 1572    |  <- Area column
  | 891.85  |  <- Area column
  | 605.77  |  <- Area column
  | 1285    |  <- Area column
  +---------+
```
Having a memory layout like this defined means that different languages can
have implementations and use the same memory layout. This means that data can
be shared between different languages without having to copy/serialize the data.

### Arrow Table
An Arrow table is a collection of columns. Each column is an Arrow array.
So for the example above we would have an array for the City, another for the
Country and another for the Area.

### Field
This struct defines the metadata of a single column.

### Schema
This is metadata for a data set and it holds a vector of Fields.

### Implelementations
* [Rust implementation](https://github.com/apache/arrow-rs).
