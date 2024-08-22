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


### XOR-compressed double arrays (XCDA)
Lets start with double array tries.
A double array trie uses two arrays, BASE and CHECK, to represent a trie structure compactly.

So first we start with a normal trie for the words "cat" and "cow":

Trie Structure with Node Numbers:
```
    0(root)
       |
    1(c)
   /     \
2(a)     3(o)
 |         |
4(t*)     5(w*)

* = end of word


```

Double Array Representation:
```
BASE[0] = Has one child which is 'c' (decimal 99).
Lets say that BASE[0] = 1 then we have 1 + 99 = 100. This would mean that then
node for 'c' would be placed at index 100 in the trie. This is far from the
root wasting a lot of space.
If we want the node for 'c' to be at index 1, then we would have to set BASE[0]
to 1 - 99 = -98. This would mean that the node for 'c' would be placed at index
1 in the trie.
So that gives us BASE[0] = -98.
```
BASE[0] + 99 = -98 + 99 = 1
```
In the above trie node 'c' has two children 'a' and 'o'. The value in BASE[1]
should be chosen such that when we add the ASCII value of 'a' to BASE[1] we
get the correct index for 'a'. And likewise for 'o'. 

Let's assume we want the node for 'a' to be at index 2. 'a' = 97.
```
'a' = 97
BASE[1] + 97 = 2
BASE[1] = 2 - 97 = -95

BASE[0] = -98
BASE[1] = -95
```
_wip_
