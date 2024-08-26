## Trie/Prefix trees/Digital trees
Is a tree like data structure used to store strings efficiently. Its very good
for string searches, prefix matching, and auto-completion.
Why is it called a trie?  
Because it is a reTRIEval tree.

What does that mean?
It means that it is a tree that is used to retrieve a key in a dataset.

Each node in the tree represents a single character in a string. The root node
is the empty string.

What does a prefix mean in this context?  
A prefix is a string that is a prefix of a longer string. For example, "abc" is
a prefix of "abcdef".

Each node in the tree usually contains a character value and a boolean value
indicating whether the node is the end of a string, and point to the nodes
children.

Notice that the time needed to traverse from the root to the leaf is not
dependent on the size structure. Even if we have a massive number of strings
the time to find a specific one does not increase as thee trie grows.


In this example, we have a trie representing the words "cat", "cow", "dog", and
"dad":
```
                root
                  |
         +--------+--------+
         |                 |
         c-----------------d
        / \               / \
       a   o             o   a
       |   |             |   |
       t*  w*            g*  d*

Legend:
* = end of word
- or | = connection between nodes
```

To search such a trie we follow the nodes. For example, to search for "cat" we
start at the root, follow the "c" node, then the "a" node, and finally the "t"
node. If the "t" node has the end of word flag set, then we have found the word.

Prefix-search is also possible. For example, to find all words starting with
"ca", we would follow the "c" node, then the "a" node, and then return all
words that can be found from that point. 

Tries exploit the fact that many strings, especially in natural language,
share common prefixes. By storing each prefix only once, tries can significantly
reduce space usage compared to storing each string separately.




```
   Operation    | Trie         | Binary Search Tree | Hash Table
   -------------|--------------|---------------------|------------
   Search       | O(m)         | O(m log n)          | O(m)
   Insert       | O(m)         | O(m log n)          | O(m)
   Delete       | O(m)         | O(m log n)          | O(m)
   Prefix Search| O(m + k)     | O(m log n + k)      | O(n)

   m: length of the string
   n: number of keys in the structure
   k: number of matches
```

Tries are generally more memory-efficient when storing a large number of strings
with common prefixes. Hash tables store each string separately, which can be
wasteful for strings with shared prefixes.

```
Trie:
       root
        |
        a
        |
        p
        |
        p
       / \
      l   r
      |   |
      e*  o
          |
          n*

* = end of word

Hash Table:
+-------------------+
| "apple" | hash1   |
+-------------------+
| "apron" | hash2   |
+-------------------+
```

Tries excel at prefix-based operations (like finding all words with a given
prefix), which are O(m + k) where k is the number of matches and m is the length
of the string.

Hash tables are not designed for prefix matching and would require O(n) time to
find all prefix matches, where n is the total number of strings.

Tries provide consistent performance regardless of the input.
Hash tables can suffer from collisions, which in worst-case scenarios can
degrade performance to O(n).

Tries maintain lexicographic ordering of strings naturally.
Hash tables do not preserve any ordering.

Tries support efficient range queries (e.g., find all strings between "apple"
and "apron").
Hash tables are not suited for range queries.


### Double Array Trie (DAT)
A double array trie uses two arrays, BASE and CHECK, to represent a trie
structure compactly.

Lets start with inserting the work 'cat' into a DAT.
```c
    trie_dat* trie = create_trie();

    insert(trie, "cow");
```

```c
void insert(trie_dat* trie, const char* word) {
    int cur_node = ROOT_NODE; // this is the current node, and we start at the ROOT node.
```
And ROOT_NODE is defined as 1, and the trie_dat structure is defined as:
```console
(gdb) p cur_node
$2 = 1

(gdb) ptype trie
type = struct {
    int *base;
    int *check;
    _Bool *terminal;
    int size;
    int capacity;
} *
```
And a trie is created using the create_trie() function:
```c
trie_dat* create_trie() {
    trie_dat* trie = malloc(sizeof(trie_dat));
    trie->size = ROOT_NODE;
    trie->capacity = INITIAL_SIZE;
    trie->base = calloc(trie->capacity, sizeof(int));
    trie->check = calloc(trie->capacity, sizeof(int));
    trie->terminal = calloc(trie->capacity, sizeof(bool));
    return trie;
}
```

We will be focusing on the base and check array in this section.

First, we will iterate over all the individual characters in the word 'cat":
```c
    for (int i = 0; word[i] != '\0'; i++) {
        int char_offset = word[i] - 'a';
```
The above is a common way to convert a character to an index. In this case, we
are converting the character 'a' to 0, 'b' to 1, 'c' to 2, etc. So the offset
for 'c' is 2.

Recall that `cur_node` is the root which is 1 for our trie:
```c
        if (trie->base[cur_node] == 0) {
            trie->base[cur_node] = trie->size;
        }
```
And since this is the first time calling insert there is nothing in the trie at
the moment. So we are setting the base[1] = 1 (this initial size of the trie).

```
BASE[1] = 1
```

Next we have are going to calculate the transition or offset from the base[1] to
the node of the character 'c':
```c
        // Calculate the transition index which uses base[s] + c.
        int t = trie->base[cur_node] + char_offset;
```

```console
(gdb) p trie->base[cur_node] + char_offset
$11 = 3
```
This value is then used as the index into check:
```
        if (trie->check[t] == 0) {
            trie->check[t] = cur_node;
            trie->size++;
        } else if (trie->check[t] != cur_node) {
            // Handle conflicts in base/check
            fprintf(stderr, "Error: Conflict detected while inserting '%s'.\n", word);
            return;
        }
```
And this value of this element will be the cur_node which is 1 in this case:
```
CHECK[3] = 1
```
And the size of the trie is incremented by 1 and will become 2.

The last thing in this loop is:
```c
        cur_node = t;
```
This is setting the current node which 3 to be the current node.

So this is what the arrays look like after we have iterated over the first
character in 'cat':
```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22
BASE  [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...]
CHECK [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...]
```
When we insert a character we calculate some number for it which we use to
insert it. The character itself is never stored. And when we search we
calculate the same number using the character and if the “path” to/from the
parent is valid we have inserted it previously. This is what the check array
is for. So we assign a unique number to each character and this number is used
in both inserts and searches.

So if we wanted to search for the work 'cow' now we would start at the root:
```
char mapping `'c' - 'a' = 2;
base[1] + 2 = 3,
      1 + 2 = 3,

check[3] == 1? Yes, continue
```
So we can see that the check verifies that we indeed have inserted this "path"
before and we can continue to the next character. We currently only have one
node but we we will insert more below.

Next we have the character 'o':
```console
(gdb) p char_offset
$18 = 14
```
And recall that `cur_node` is 3
```c
        if (trie->base[cur_node] == 0) {

```

```console
(gdb) p trie->size
$21 = 2
```
We currently have 2 nodes 'ROOT' and 'c'.

And the last node we inserted was 'c' and the index was 3:
```
(gdb) p cur_node
$22 = 3
```
Then we will set trie->base[3] = 2:
```console
trie->base[cur_node] = trie->size;
```
And we caclulate the transition index:
```c
int t = trie->base[cur_node] + char_offset;
```
```console
(gdb) p char_offset
$123 = 14
(gdb) p trie->base[cur_node]
$124 = 2
(gdb) p trie->base[cur_node] + char_offset
$125 = 16
```
And we will set check[16] = 3
```console
trie->check[t] = cur_node;
```

This will give a arrays that look like this:
```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22
BASE  [ 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...]
CHECK [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0...]
```
Next we have the character 'w' and cur_node is now 16.
```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22, 23  24  25 
BASE  [ 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0 ,  0,  0]
```

```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22, 23  24, 25
CHECK [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,  0,  0, 16]
```
This is the last character in the word 'cow' and we will set the terminal flag:
```
trie->terminal[cur_node] = true;
```
So the arrays look like this after 'cow' has been inserted:
```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22, 23  24  25 
BASE  [ 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,  0,  0,  0]
CHECK [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,  0,  0, 16]
```

Trie Structure with Node Numbers:
```
    1(root)
       |
    3(c)
   / 
16(0)
 |
25(w*)

* = end of word
```

### Packed Double Array Trie (PDAT)
When implementing the double array trie and stepping through the code I noticed
that the arrays, BASE and CHECK are quite sparely populated. This is because
the arrays are allocated based on the number of nodes in the trie. The number of
nodes in the trie is based on the number of characters in the input strings.
This can be quite large and wasteful. Packing the arrays can reduce the memory
footprint of the trie.
implementation. There seems to be many different ways to pack the arrays but
I'll focus on the method used in llama.cpp.

There is an example of Double Array Trie in
[dat](../fundamentals/datastructures/sr/dat.c) which might help to take a look
at to get an understanding of how a DAT works.

So, instead of using two arrays for BASE and CHECK we will now have have a
single uint32_t array which will contain all the information for a node the
trie.
``
Bits    0-7: LCHECK value (8 bits)
Bit       8: LEAF flag (1 bit)
Bit       9: BASE extension flag (1 bit)
Bits  10-30: BASE value or Value (21 bits)
Bit      31: Sign bit for LCHECK or additional VALUE bit
```

### XOR Compressed Double Array (XCDA)
This is a way to compress the double array trie even further. The idea is to
store the XOR of the BASE and CHECK values in a single array. This will reduce
the memory footprint of the trie even further.

The xor_array stores `base ^ check` for each node.
To get base: `xor_array[current] ^ parent`
To get check: `xor_array[next] ^ current`

```
Standard DAT:
Index:   0    1    2    3    4    5    6
base: [  1,   2,   3,   0,   5,   0,   0  ]
check: [ -1,  0,   1,   2,   1,   4,   0  ]

XOR-CDAT:
Index:   0    1    2    3    4    5 
xor_array: [ 1^-1, 2^0, 3^1, 0^2, 5^1, 0^4 ]

Char mapping: a->1, b->2, c->3

Search for "abc" in XOR-CDAT:
1. For 'a': xor_array[0] ^ 0 = 1, xor_array[1] ^ 0 = 2
2. For 'b': xor_array[1] ^ 1 = 3, xor_array[2] ^ 1 = 2
3. For 'c': xor_array[2] ^ 2 = 1, xor_array[3] ^ 2 = 2 (end of word)
```
_wip_

