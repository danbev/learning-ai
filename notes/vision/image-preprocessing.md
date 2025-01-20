## Image preprocessing
This document is about image preprocessing in the context of multi-modal models, like
Llama 3.2 Vision Instruct.

The image encoder is a pre-trained ViT-H/14 model which produces image patch
embeddings. These embeddings are then used in the cross-attention for the
language model. This is a little different from other models where the image
patch embeddings are projected into the same embedding space as the text
embeddings and then are used as normal tokens to the transformer model.

### Pre-processor config
Some models have this information in `preprocessor_config.json` which is needed
for the pre-processing of images, for example Llama 3.2 Vision Instruct has the
following:
```console
{
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "MllamaImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "max_image_tiles": 4,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 560,
    "width": 560
  }
}
```
Using this information we can see that rescaling is used:
```
"do_rescale": true,
"rescale_factor": 0.00392156862745098,
```
This is 1/255 which we can verify using:
```console
1/255 = 0.00392156862745098
```
This will recale to a range of [0,1] from [0,255].

This means the preprocessing pipeline does:
1. Convert to RGB if needed (`do_convert_rgb`: true)
2. Rescale pixels using `rescale_factor`(`do_rescale`: true)
```c++
    pixel = pixel * 0.00392156862745098  // Same as pixel / 255.0f
    or 
    pixel = pixel / 255.0f
```
3. Normalize using mean/std (`do_normalize`: true)
```c++
    normalized = (pixel - mean) / std
```
4. Resize/pad to specified size (`do_resize": true, `do_pad`: true)

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
can become squished, and tall images can become stretched and become unrecognizable. Text
migth become unreadable, and objects might become unrecognizable.

So what is done is the larger image is split into multiple images of the size that the model
expects. For example, if the model expects (was trained on) 560x560 images, images that
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
|       |            |   1   |(560x560)   |   1   |(560x560)   |   1   |(560x560)
|   1   |            +-------+            +-------+            +-------+
|       |            |   2   |(560x560)   |   2   |(560x560)   |   2   |(560x560)
+-------+            +-------+            +-------+            +-------+
(560x560)                                 |   3   |(560x560)   |   3   |(560x560)
                                          +-------+            +-------+
                                                               |   4   |(560x560)
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
Each tile represents a 560x560 pixel area and in the code the `tile_size` represents the dimension
so 560 in this case. Numbers indicate processing order (left-to-right, top-to-bottom).

```c++
static std::pair<int,int> get_optimal_canvas(int w, int h, int n_tiles, int tile_size) {
    printf("get_optimal_canvas: w=%d, h=%d, n_tiles=%d, tile_size=%d\n", w, h, n_tiles, tile_size);
    // This is the size that the model expects.
    int model_dim = tile_size;

    // Calculate the width to height ratio.
    //        10
    // +-----------------+
    // |                 |
    // |                 | 5         10 / 5 = 2
    // |                 |
    // +-----------------+
    //     5
    // +--------+
    // |        |
    // |        |          10         5 / 10 = 0.5
    // |        |
    // |        |
    // |        |
    // |        |
    // +--------+
    float aspect = static_cast<float>(w) / h;

    // If the width or height is less than the min dimension
    if (w < model_dim || h < model_dim) {
        if (aspect > 1.0f) {
	        //      Wide image
	        //  +-----------------+          Example: 300x200 image
	        //  |                 |          aspect = 300 / 200 = 1.5
	        //  |                 |          w = 560 (max of 300 and 560)
	        //  |                 |          h = 373 (560 / 1.5)
	        //  +-----------------+
	        // Set width to model_dim or width.
            w = std::max(w, model_dim);
	        // Calculate a new height based on the aspect ratio.
            h = static_cast<int>(w / aspect);
        } else {
            //	    Tall image
            //	    +-----+
            //	    |     |
            //	    |     |
            //	    |     |
            //	    |     |
            //	    +-----+
            //
            // Set height to model_dim or height.
            h = std::max(h, model_dim);
            // Calculate a new width based on the aspect ratio.
            w = static_cast<int>(h * aspect);
        }
    }

    int max_canvas_w = std::min(w, n_tiles * tile_size);
    int max_canvas_h = std::min(h, n_tiles * tile_size);

    return {max_canvas_w, max_canvas_h};
}
```
So lets say we have 4 tiles and the model expects images to be 560x560. So the maximum
canvas would be 4 * 560 = 2240 pixels.

A canvas is like a drawing canvas where the tiles will be placed. The size of the canvas
is determined by how many tiles we need. 
For example, I have an image that is 1280x853 and 3 channels, the canvas for this would be
1280x853 (since it is smaller than 2240).
```
                  1280
     +------------------------------+
     |                              |
     |                              |
     |                              |  853
     |                              |
     |                              |
     |                              |
     +------------------------------+
```

```c++
static std::pair<int,int> scale_to_fit_canvas(int w, int h, int canvas_w, int canvas_h, int tile_size) {
    double scale = std::min(double(canvas_w) / double(w), double(canvas_h) / double(h));

    if (scale > 1.0) {
	// Don't upscale the image if it is smaller than the canvas.
        scale = 1.0;
    }

    // Apply scaling to get new width and height.
    int new_w = int(std::floor(w * scale));
    int new_h = int(std::floor(h * scale));

    // Round down to multiples of tile_size
    new_w = (new_w / tile_size) * tile_size;
    new_h = (new_h / tile_size) * tile_size;

    // Ensure that the new dimensions are at least tile_size(model dimension size)
    if (new_w < tile_size) new_w = tile_size;
    if (new_h < tile_size) new_h = tile_size;

    return {new_w, new_h};
}
```
So for the example of the 1280x853 image, the scaling factor will be 1. Notice that we will
round this down:
```
width:
  1280 / 560 = 2
  2 * 560    = 1120
height:
  853 / 560 = 1
  1 * 560   = 560

                  1120
     +------------------------------+
     |                              |
     |                              |
     |                              |  560
     |                              |
     |                              |
     |                              |
     +------------------------------+

Scaled size: 1120 x 560
```

```c++
    // Resize to Width/Height/Channel
    std::vector<unsigned char> resized_hwc(final_w * final_h * 3);
    stbir_resize_uint8_linear(
        input, i_w, i_h, i_w * 3,
        resized_hwc.data(),
        final_w, final_h, final_w * 3,
        STBIR_RGB
    );
```
This is creating a vector of size 1280x560x3 = 1881600. Notice that we then call `stbir_resize_uint8_linear`
and we specify the input image, the input width, height, and channels. Then we specify where output
data should be stored and the width and height, and channels for the output image. So this is where
the scaling of the image happens.

Then we are going to split this resized image into tiles;
```c++
static std::vector<std::vector<unsigned char>> subdivide_into_tiles(
    const unsigned char* resized_hwc,
    int final_w,
    int final_h,
    int tile_size
) {
    // Number horizontal tiles (x-axis)
    int tiles_x = final_w / tile_size;
    // Number of vertical tiles (y-axis)
    int tiles_y = final_h / tile_size;

    std::vector<std::vector<unsigned char>> tiles;
    tiles.reserve(tiles_x * tiles_y);

    // iterate over the y axis (rows)
    for (int ty = 0; ty < tiles_y; ty++) {
        // iterate over the x axis (columns)
        for (int tx = 0; tx < tiles_x; tx++) {
            // Create a vector to store the one tile
            std::vector<unsigned char> tile_data(tile_size * tile_size * 3);

            for (int tile_row = 0; tile_row < tile_size; tile_row++) {
                for (int tile_col = 0; tile_col < tile_size; tile_col++) {
                    int src_x = tx * tile_size + tile_col;
                    int src_y = ty * tile_size + tile_row;

                    int src_idx = (src_y * final_w + src_x) * 3;
                    int dst_idx = (tile_row * tile_size + tile_col) * 3;

                    // copy 3 channels
                    tile_data[dst_idx + 0] = resized_hwc[src_idx + 0]; // Red
                    tile_data[dst_idx + 1] = resized_hwc[src_idx + 1]; // Green
                    tile_data[dst_idx + 2] = resized_hwc[src_idx + 2]; // Blue
                }
            }
            tiles.push_back(std::move(tile_data));
        }
    }

    return tiles;
}
```
So after this we will have a vector of vectors of unsigned chars, where each vector represents
a tile of the image. And the size of each tile is 560x560x3 = 940800 bytes. And the values will
be in `[R G B] [R G B]...`.

