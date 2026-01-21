## Multi-Head Latent Attention (MLA)
This is an attention mechanism that was introduced by DeepSeek (V2) and its goal
is to reduce the massive memory usage of the KV-cache during long context
inference.

In multi-head attention the key and value matrix must be stored for every token
in the conversation history. Grouped query attention (GQA) tries to reduce this
memory by sharing key and value matrices across multiple query heads. But even
with GQA as the context grows larger and larger the KV-cache can still grow to
hundreds of gigabytes in size, which can make it impossible to run on most
consumer devices. This is where MLA comes in.

The idea is that instead of storing the full matrices for the keys and values
they are projected down into a smaller latent space (hence latent in the name).
So when the model processes a token embedding it projects the Key and Value into
a low-dimensional latent vector using a down projection. And this vector is what
is stored.

In a Standard Transformer (MHA), the Attention block works like this:
1. Input: Takes the vector from the previous layer (h).
2. Expand: Multiplies h by three giant matrices (W_Q, W_K, W_V) to produce three
   huge vectors: Query (Q), Key (K), and Value (V).
3. Cache: Stores the huge K and V vectors (after the have been projected by the
   previous step, in the KV Cache.
4. Attend: Calculates attention scores using Q and the cached K.

In MLA the Attention block works like this:
1. Input: Takes the vector from the previous layer (h).
2. Compress (Down-Project): Multiplies h by a small "Compression Matrix" (W_Down)
   to create a small latent vector (c_KV).
3. Cache: Stores only this small c_KV in the KV Cache (and rope information, but more on this later).
4. Expand (Up-Project): When we need to calculate scores, the model mathematically
   "unpacks" c_KV back into usable Keys and Values (conceptually) to do the math.

Having to compress using a down project matrix multiplication and then later
expand using an up project matrix multiplication does add some computational
overhead. But it was shown that the uncompressing can actually be avoided.  If
we imagine we have:
```console
Query * (W_unzip* compressed_key)
```
Since multiplication is associative we can rearrange this to:
```console
(Query * W_unzip) * compressed_key
```
And this means the same thing. And this called "absorbing" the unzip_matrix. We
can do this as the W_unzip is fixed and known ahead of time.

But we also have to account for the position encoding like RoPE, which tells the
model that "Dan" came before "loves".

First we have two matrices which are `W_down` which is used to compress, and
`W_unzip` which is used to decompress. And the matrix absorption trick happens
during the read step where which we saw earlier.

For RoPE we would have to:
* Rotate the key based on its position R_pos.
* Then compress the rotated key using the down projection.
```console
score = Query * (W_down * (R_pos * Key))
```
Now, like we mentioned earlier we want to use the absorption trick to avoid and
that was possible because we could merge the W_unzip matrix into the Query matrix
and then perform the multiplication with the compressed key. But now we have
the rope information in R_pos which is position dependent, that is is it different
for each token embedding in the sequence. So before we could use Query * W_unzip
and then multiply by the compressed key because W_unzip was fixed. But now with
R_pos it would mean that we would have to include this position information in
the multiplication which would then have to be done for every token in the
sequence which would defeat the purpose of the whole exercise.
For example, if we are currently processing the 3rd token in the sequence we would
need to look back at the 2nd token which means we would need to rotate the Query
vector based on the position of the 2nd token. And this would have to be
done for every token in the sequence which would be very expensive.
```console
Absorbed_Query = Query * W_Unzip * R_pos
```
We would not be able to create a single absorbed query matrix that works for all
tokens because R_pos is different for each token position. So instead we would
need one absorbed query matrix per token position which would be very expensive.
We would be recreating the the matrix for every token in the sequence which is
slower than just unzipping the key directly.

Could we not just RoPE before compressing the key then?  
No, because recall that RoPE relies on pairs and rotates those pairs of values
based on their position. This precise gemetric relationship is crucial for RoPE
to encode the distance. Compressing this is like smearing all the dimensions
togeher and losing the precise positional relationships between the pairs.

The solution to both of these issue is to use a "side car" for the RoPE
information. This allows us to keep the positional information separate from the
content information and we can still use the absorption trick for the content.
And it also preserves the positional information. We have to then store the
rope information in the kv-cache along with the compressed content.

So the cache in memory might looks like this for each token:
```console
[compressed content (512 floats)] + [pos info (64 floats)]
```

So we have two tracks/paths:
```console
Hidden State (h)
     |
     |--- Path 1: Multiply by W_Down_Content -> [Compressed Content] (Big part)
     |
     |--- Path 2: Multiply by W_Down_Rope    -> [Raw Position]       (Tiny part)
```

No-rope:
1. The compressed content path (what we are talking about)
   q_content (absorbed/projected)
   k_content (from the compressed cache C_KV)
   score_content = q_content . k_content

Rope:
2. The uncompressed positional info path (where in the sentence/sequence we are)
    q_rope (a small vector)
    k_rope (a small vector)
    score_rope = q_rope . k_rope

The final attention score is then:
```console
Final Score = score_content + Score_rope
```
So the KV-Cache in DeepSeek-V3 actually stores two things for every token:
1. The compressed content vector (C_KV)
2. The RoPE Key (k_rope) not compressed
