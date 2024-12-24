## Image preprocessing
This document is about image preprocessing in the context of multi-modal models, like
Llama 3.2 Vision Instruct.


### Image loading
Image loading in llama.cpp uses stb_image.h:
```c++
    int inW, inH, inC;
    unsigned char * input = stbi_load(fname, &inW, &inH, &inC, 3);
```
The last parameter is the number of channels, which is always 3 for RGB images.
And the width, height, and the channels are stored in inW, inH, and inC respectively. So
inC would be 1 for a grayscale image, and 3 for an RGB image, and 4 for an RGBA image.


### Tiling
Tiling is something that is used for large images so that they don't get scaled down
into to an image that is too small, which might distort the image information and cause
the model to not be able to process it properly, or with enough accuracy. Wide images
can become squished, and tall images can become stretched and be come unrecognizable. Text
migth become unreadable, and objects might become unrecognizable.

So what is done is the larger image is split into multiple images of the size that the model
expects. For example, if the model expects (was trained on) 560x560 images images that
are larger can be split into multiple 560x560 images. This is where the concept of
tiling comes in. This allows us to keep the original proportions of the image and keep
a natural view of the image.

As an example, in Llama 3.2 Vision Instruct, the model expects 560x560 images and has
the following aspect ration configuration:
```
"supported_aspect_ratios": [
      [ 1, 1 ],
      [ 1, 2 ],
      [ 1, 3 ],
      [ 1, 4 ],
      [ 2, 1 ],
      [ 2, 2 ],
      [ 3, 1 ],
      [ 4, 1 ]
    ],
```
So it has 8 different aspect ratio configurations that it can handle. The first number
is the width, and the second number is the height. So for example, the first configuration
is 1x1, which means that the image is square. The second configuration is 1x2, which means
that the image is twice as tall as it is wide. 

Aspect Ratio Configurations (each box represents a 560x560 tile)
```

1. 1x1 (Index 1)     2. 1x2 (Index 2)     3. 1x3 (Index 3)     4. 1x4 (Index 4)
+-------+            +-------+            +-------+            +-------+
|       |            |   1   |            |   1   |            |   1   |
|   1   |            +-------+            +-------+            +-------+
|       |            |   2   |            |   2   |            |   2   |
+-------+            +-------+            +-------+            +-------+
                                          |   3   |            |   3   |
                                          +-------+            +-------+
                                                               |   4   |
                                                               +-------+

5. 2x1 (Index 5)     6. 2x2 (Index 6)     7. 3x1 (Index 7)
+-------+-------+    +-------+-------+    +-------+-------+-------+
|   1   |   2   |    |   1   |   2   |    |   1   |   2   |   3   |
+-------+-------+    +-------+-------+    +-------+-------+-------+
                     |   3   |   4   |
                     +-------+-------+

8. 4x1 (Index 8)
+-------+-------+-------+-------+
|   1   |   2   |   3   |   4   |
+-------+-------+-------+-------+

```
Each tile represents a 560x560 pixel area. Numbers indicate processing order (left-to-right, top-to-bottom)
