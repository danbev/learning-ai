## CLIP Learning Transferable Visual Models From Natural Language Supervision
Is a paper from OpenAI that introduces a new way to train models to understand
images. The paper introduces a new model called CLIP (Contrastive Language-Image
Pretraining) that is trained by contrasting images and their descriptions in
natural language. Contrastive is a method that highlights the differences
between two things.

The model is fed with pairs of images and texts during training. Each pair
consists of an image and a correct textual description, along with several
incorrect descriptions. The model's objective is to learn to identify the
correct pairings among the incorrect ones.

The model is trained to map similar images and text descriptions close together
in the embedding space, while pushing dissimilar pairs apart. So a picture of a
cat and a text description of a cat should be close together in the embedding
space.

Another usecase for CLIP is a sementic search for image retrieval. This is where
we have a query and we want to find the most similar images to the query.

Two main components:
1. Image encoder
This part produces image embeddings like patch embeddings from a ViT I think,
like the last layer of a ViT model inference. But it does not have to use ViT
and a CNN could be used as well. The embeddings can be thought of as simliar to
token embeddings but there is no lookup of the id of a token in a vocabulary,
instead the embeddings use the raw pixel values of the image.

2. Text encoder
This part produces text embeddings from a transformer model designed for NLP.

During training, CLIP learns to align the embeddings from the image encoder with
the embeddings from the text encoder. 

The process involves calculating embeddings (vector representations) for both
the images and the text descriptions. The model is trained to minimize the
distance between embeddings of correct image-text pairs while maximizing the
distance between embeddings of incorrect pairs. This is known as a contrastive
loss function, specifically designed to handle such tasks.

One thing to note here is that if you have a batch of images and text pairs
that are all about cats you might get a model that does not work as expected.
This is because if we use a batch and want to find the most similar image to
text and the rest we treat as negative examples (but the actually are not) so
there are ways where a queue/cache (called momentum queues) of images and text
pairs are used instead to avoid taking negative examples from the same batch
(and possibly same category). Normally the selection of batch entries is random
but this can happen.

Now, in the following the text input is a description of the image and the
image input:
```
   +-------------+      +----------------+
   | Text inputs |----->| Text encoder   |---> Sequence of token embeddings
   +-------------+      +----------------+

   +-------------+      +----------------+
   | Image inputs|----->| Image Encoder  |---> Sequence of patch embeddings
   +-------------+      +----------------+
```
Note that we have pairs of texts and images, and we generate a sequence of
token embeddings for each input text, and a sequence of patch embeddings for
each input image.

These are then taken and used to calculate the cosine similarity between the
text and image embeddings:
```

                     Token embeddings

            +----------------------------------+
Patch       |P₁T₁|P₁T₂|  ...              |PₘTₙ|
embeddings  +---+---+---+---+---+---+---+---+--+
            |P₂T₁|P₂T₂|  ...              |PₘTₙ|
            +---+---+---+---+---+---+---+---+--+
            |             .                    |
            |             .                    |
            |             .                    |
            |PₘT₁|PₘT₂|  ...              |PₘTₙ|
            +---+---+---+---+---+---+---+---+--+
```
The goal of training is to have the highest cosine similarity between the
correct image-text pair and the lowest cosine similarity between the incorrect
image-text pairs. So this means that the diagonal of the matrix of cosine
similarities should be the highest and the off-diagonal elements should be the
lower.

The output of training is a model can then be used for zero-shot classification,
where it can classify images based on textual descriptions, and vice versa. This
is done by comparing the image's embedding with embeddings of textual
descriptions of potential categories. The category whose text embedding is
closest to the image's embedding (as measured by cosine similarity or another
distance metric) is chosen as the classification. 

It can also be used to search for images based on textual descriptions. It takes
the textual description and generates an embedding for the query and the
searches for images with similar embeddings in the database.

This is a new way to train models that can understand images and it is a step
away from the traditional way of training models to understand images which was
to use labeled images.

### Vision Tower
This is a name for the vision encoder part of the CLIP model. This could be ViT
for example.

The "tower" terminology signifies that each of these components is a distinct
part of the overall architecture, with each "tower" focusing on a different
modality (vision for images and language for text). Perhaps "tower" comes from
that these parts are fairly independent and have distinct roles.

"In machine learning, particularly with complex models like CLIP, the metaphor
of a tower helps convey the idea of a component designed for a specific type of
data processing, stacked layer upon layer to transform input data into a
high-level representation. Each "tower" within a model like CLIP is specialized
(e.g., one for images and another for text), yet they work together within the
overarching architecture to achieve a common goal, such as aligning visual and
textual representations."


### CLIP vs VIT
CLIP and VIT are not the same thing as I understand it. CLIP which stands for
contrastive language-image pretraining can use a vision transformer to process
the images but CLIP itself is the complete concept of the training process.
At inference time, like what is being worked on in llama.cpp, it is the ViT
component of CLIP thas being used and perhaps using clip as the prefix for
methods/structs/tensor can be a litle misleading. Would this be better off
named something like 'vit' or `vision_model`, `vision_layer` etc.

### CLIP image preprocessing

* Resize/crop the images to 224x224 pixels (which is what was used in the paper)
* Normalize the pixel values to be in the range [0, 1]
* Convert to RGB format if needed
