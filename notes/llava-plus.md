## Large Language and Vision Assistent that Plugs And Learns to Use Skills (LLaVA-Plus)
So in this case there is a set of skills that extend LLaVA with the ability to
use external APIs as tools (skills).
These APIs can alse large multimodal models like CLIP, DALL-E, etc. So this
a model that is finetuned for instruction following and it is trained to use
the API as tools enbling the model to be grounded.

Extended skills:
* External knowledge which enables the RAG (for images) which is done using [CLIP search](clip-search.md).
* Generation, for this [Stabile Diffusion](stabile-diffusion.md) is used.
* Visual prompting which uses [SAM](sam.md)

ChatGPT is also used to glue these skills together, like selecting the top-k
matching images from the similarity search for example.

### Visual Prompting
This is visual input (the prompting part) which have been edited to hightlight
something of importance to the user or thing that is prompting. This could
segmenting the image like creating/drawing a box around something of interest.
One might ask the box to be removed by the model, or changed in some other way.
