## CLIP Search
This is about using CLIP to search for images. For example the user can send
a query which will be converted to a patch embedding. There would also be a
database of images which also have patch embeddings calculated also in the same
space. Then a similarity search can be performed. Does this sounds familiar?  
Well this is almost what happens with RAG, where we have a few example where a
query is first converted to token embeddings and then a similarity search is
performed against information in a vector database. But in this case images are
used instead of text.
