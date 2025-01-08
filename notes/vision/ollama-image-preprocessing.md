## Ollama image preprocessing

In model/mllama/imageproc.go we have the following:
```go
func Preprocess(imageData io.Reader) ([]float32, map[string]any, error) {

    img, format, err := image.Decode(imageData)

    newImage, aspectRatio := resizeImage(img, format, outputSize, maxTiles)

    newImage = padImage(newImage, outputSize, aspectRatio)

    tiles := splitToTiles(newImage, aspectRatio)

    data := packImages(newImage, aspectRatio)

    aspectRatioIndex := slices.Index(getSupportedAspectRatios(maxTiles), aspectRatio) + 1
```
This does not looks so bad but we also need to take a look at the resizeImage function:

```go
func resizeImage(img image.Image, format string, outputSize image.Point, maxImageTiles int) (image.Image, image.Point) {
	if format == "png" {
		img = imageproc.Composite(img)
	}
	slog.Info("Resize image", "outputSize", outputSize, "maxImageTiles", maxImageTiles)

	b := img.Bounds()
	tileSize := outputSize.Y

	canvasSize := getOptimalTiledCanvas(b.Max, maxImageTiles, tileSize)
	aspectRatio := image.Point{canvasSize.X / tileSize, canvasSize.Y / tileSize}
	slog.Info("Aspect ratio", "aspectRatio", aspectRatio)
	newSize := getImageSizeFitToCanvas(b.Max, canvasSize, tileSize)

	return imageproc.Resize(img, newSize, imageproc.ResizeBilinear), aspectRatio
}
```

```go
func getOptimalTiledCanvas(imageSize image.Point, maxImageTiles, tileSize int) image.Point {
        slog.Info("Get optimal tiled canvas", "imageSize", imageSize, "maxImageTiles", maxImageTiles, "tileSize", tileSize)
	possibleTileArrangements := getSupportedAspectRatios(maxImageTiles)
    ...
```


```go
func getSupportedAspectRatios(maxTiles int) []image.Point {
        slog.Info("Get supported aspect ratios", "maxTiles", maxTiles)
	ratios := []image.Point{}

	for w := range maxTiles {
		for h := range maxTiles {
			if (w+1)*(h+1) <= maxTiles {
				ratios = append(ratios, image.Point{w + 1, h + 1})
			}
		}
	}

        slog.Info("Get supported aspect ratios", "ratios", ratios)
	return ratios
}
```

```console
level=INFO source=imageproc.go:34 msg="Get supported aspect ratios"
ratios="[(1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (3,1) (4,1)]"
```
```console
0 [1x1] [1x2] [1x3] [1x4]
1 [2x1] [2x2]
2 [3x1]
3 [4x1]
```

### getOptimalTiledCanvas
```console
level=INFO source=imageproc.go:49 msg="Get optimal tiled canvas"
imageSize=(1500,1749) maxImageTiles=4 tileSize=560


level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(560,560)
level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(560,1120)
level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(560,1680)
level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(560,2240)
level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(1120,560)
level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(1120,1120)
level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(1680,560)
level=INFO source=imageproc.go:57 msg="Possible canvas size" pcs=(2240,560)
```

```go
func getOptimalTiledCanvas(imageSize image.Point, maxImageTiles, tileSize int) image.Point {
    slog.Info("Get optimal tiled canvas", "imageSize", imageSize, "maxImageTiles", maxImageTiles, "tileSize", tileSize)
	possibleTileArrangements := getSupportedAspectRatios(maxImageTiles)

	possibleCanvasSizes := []image.Point{}

	for _, pta := range possibleTileArrangements {
		possibleCanvasSizes = append(possibleCanvasSizes, image.Point{pta.X * tileSize, pta.Y * tileSize})
	}
```
The `possibleCanvasSizes` is a dynamic array of image.Point and in go the builtin function
append is used to add elements to the array. So this is creating new elements for each of the
supported aspect ratios. And we know that the tileSize is 560 for ollama so this will produce:
```console
(1,1) (560,560)
(1,2) (560,1120)
(1,3) (560,1680)
(1,4) (560,2240)
(2,1) (1120,560)
(2,2) (1120,1120)
(3,1) (1680,560)
(4,1) (2240,560)
```

Next, we want to figure out how much the image would need to be scaled to fit into
each possible canvas:
```go
	scales := []float64{}

	for _, pcs := range possibleCanvasSizes {
		scaleHeight := float64(pcs.Y) / float64(imageSize.Y)
		scaleWidth := float64(pcs.X) / float64(imageSize.X)

		if scaleWidth > scaleHeight {
			scales = append(scales, scaleHeight)
		} else {
			scales = append(scales, scaleWidth)
		}
	}
```

```console
pcs=(560,560)   scaleHeight=0.32018296169239563 scaleWidth=0.37333333333333335
pcs=(560,1120)  scaleHeight=0.6403659233847913  scaleWidth=0.37333333333333335
pcs=(560,1680)  scaleHeight=0.9605488850771869  scaleWidth=0.37333333333333335
pcs=(560,2240)  scaleHeight=1.2807318467695825  scaleWidth=0.37333333333333335
pcs=(1120,560)  scaleHeight=0.32018296169239563 scaleWidth=0.7466666666666667
pcs=(1120,1120) scaleHeight=0.6403659233847913  scaleWidth=0.7466666666666667
pcs=(1680,560)  scaleHeight=0.32018296169239563 scaleWidth=1.12
pcs=(2240,560)  scaleHeight=0.32018296169239563 scaleWidth=1.4933333333333334

msg=Scale index=0 value=0.32018296169239563
msg=Scale index=1 value=0.37333333333333335
msg=Scale index=2 value=0.37333333333333335
msg=Scale index=3 value=0.37333333333333335
msg=Scale index=4 value=0.32018296169239563
msg=Scale index=5 value=0.6403659233847913
msg=Scale index=6 value=0.32018296169239563
msg=Scale index=7 value=0.32018296169239563
````

```go

	var minUpscale float64
	var maxDownscale float64
	var upscale bool

	for _, s := range scales {
		if s > 1.0 {
			upscale = true
			if minUpscale == 0 {
				minUpscale = s
			} else {
				minUpscale = math.Min(minUpscale, s)
			}
		} else {
			maxDownscale = math.Max(maxDownscale, s)
		}
	}

	selectedScale := maxDownscale
	if upscale {
		selectedScale = minUpscale
	}

	var selectedCanvas image.Point
	for n, pcs := range possibleCanvasSizes {
		if scales[n] == selectedScale {
			// choose the smallest possible canvas
			if selectedCanvas.X == 0 && selectedCanvas.Y == 0 {
				selectedCanvas = pcs
			} else if pcs.X*pcs.Y < selectedCanvas.X*selectedCanvas.Y {
				selectedCanvas = pcs
			}
		}
	}
	slog.Info("Selected canvas", "selectedCanvas", selectedCanvas)
	return selectedCanvas
}
```
```console
img="Selected canvas" selectedCanvas=(1120,1120)
```


Back in resizeImage we then have:
```go
func resizeImage(img image.Image, format string, outputSize image.Point, maxImageTiles int) (image.Image, image.Point) {
    ...
	canvasSize := getOptimalTiledCanvas(b.Max, maxImageTiles, tileSize)
-->	aspectRatio := image.Point{canvasSize.X / tileSize, canvasSize.Y / tileSize}
	slog.Info("Aspect ratio", "aspectRatio", aspectRatio)
```
So this is takeing (1120,1120) and dividing it by 560 to get the aspect ratio:
```console
1120 / 560 = 2
1120 / 560 = 2
```

```console
msg="Aspect ratio" aspectRatio=(2,2)
```
Then we have:
```go
	newSize := getImageSizeFitToCanvas(b.Max, canvasSize, tileSize)
```
```go
func getImageSizeFitToCanvas(imageSize, canvasSize image.Point, tileSize int) image.Point {
	targetWidth := clip(imageSize.X, tileSize, canvasSize.X)
	targetHeight := clip(imageSize.Y, tileSize, canvasSize.Y)

	scaleWidth := float64(targetWidth) / float64(imageSize.X)
	scaleHeight := float64(targetHeight) / float64(imageSize.Y)

	var w, h int

	if scaleWidth < scaleHeight {
		w = targetWidth
		h = min(int(math.Floor(float64(imageSize.Y)*scaleWidth)), targetHeight)
	} else {
		w = min(int(math.Floor(float64(imageSize.X)*scaleHeight)), targetWidth)
		h = targetHeight
	}

	return image.Point{w, h}
}
```
```go
func clip(a, a_min, a_max int) int {
	if a < a_min {
		return a_min
	} else if a > a_max {
		return a_max
	}

	return a
}
```
```console
msg="Get image size fit to canvas" imageSize=(1500,1749) canvasSize=(1120,1120) tileSize=560
msg="Get image size fit to canvas" imageSize=(1500,1749) canvasSize=(1120,1120) tileSize=560
targetWidth=1120 targetHeight=1120
```

Then we have:
```go
	newSize := getImageSizeFitToCanvas(b.Max, canvasSize, tileSize)
	slog.Info("New scaled image size", "newSize", newSize)
```
```console
msg="New scaled image size" newSize=(960,1120)
```
And the final line in `resizeImage` is:
```go
	return imageproc.Resize(img, newSize, imageproc.ResizeBilinear), aspectRatio
```
Notice that this function returns a tuple with the new image and the aspect ratio. And
we are passing in the newSize, and the alginment is `imageproc.ResizeBilinear`.

```go
func Resize(img image.Image, newSize image.Point, method int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, newSize.X, newSize.Y))

	kernels := map[int]draw.Interpolator{
		ResizeBilinear:        draw.BiLinear,
		ResizeNearestNeighbor: draw.NearestNeighbor,
		ResizeApproxBilinear:  draw.ApproxBiLinear,
		ResizeCatmullrom:      draw.CatmullRom,
	}

	kernel, ok := kernels[method]
	if !ok {
		panic("no resizing method found")
	}

	kernel.Scale(dst, dst.Rect, img, img.Bounds(), draw.Over, nil)
	slog.Info("Resize image", "method", method, "newSize", newSize, "dst", dst.Bounds())

	return dst
}
```
```console
msg="Resize image" method=0 newSize=(960,1120) dst=(0,0)-(960,1120)
```

### padImage
```go
    newImage = padImage(newImage, outputSize, aspectRatio)
    slog.Info("Padded image", "bounds", newImage.Bounds())
    writeDebugImage(newImage, "ollama_padded.bin")
```
```go
func padImage(img image.Image, outputSize, aspectRatio image.Point) image.Image {
	slog.Info("Pad image", "aspectRatio", aspectRatio)
	paddedSize := image.Point{
		X: outputSize.X * aspectRatio.X,
		Y: outputSize.Y * aspectRatio.Y,
	}
	slog.Info("Padded size", "paddedSize", paddedSize)

	dst := image.NewRGBA(image.Rect(0, 0, paddedSize.X, paddedSize.Y))
	slog.Info("Dst bounds", "dstBounds", dst.Bounds())
	draw.Draw(dst, img.Bounds(), img, image.Point{0, 0}, draw.Over)

	return dst
}
```
```console
msg="Pad image" aspectRatio=(2,2)
msg="Padded size" paddedSize=(1120,1120)
```

Next the images are packed (recall that we now have 2x2 tiles so we have 4 560x560 images):
```go
    data := packImages(newImage, aspectRatio)
```

```go
func packImages(img image.Image, aspectRatio image.Point) []float32 {
	slog.Info("Pack images", "aspectRatio", aspectRatio)
	subImages := splitToTiles(img, aspectRatio)
	slog.Info("Sub images", "subImages", subImages)

	var pixelVals []float32

	rescale := true
	channelFirst := true

	for _, subImg := range subImages {
		vals := imageproc.Normalize(subImg, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, rescale, channelFirst)
		pixelVals = append(pixelVals, vals...)
	}

	return pixelVals
}
```
```console
msg="Pack images" aspectRatio=(2,2)
```
Then we call splitToTiles:
```go
    // Get tiles and write debug info for each
	subImages := splitToTiles(img, aspectRatio)
```

```go
func splitToTiles(img image.Image, numTilesSize image.Point) []image.Image {
	b := img.Bounds()
    slog.Info("Split to tiles", "numTilesSize", numTilesSize, "bounds", b)
	width := b.Max.X - b.Min.X
	height := b.Max.Y - b.Min.Y
	tileHeight := height / numTilesSize.Y
	tileWidth := width / numTilesSize.X

	images := []image.Image{}

	for h := range numTilesSize.Y {
		for w := range numTilesSize.X {
			rect := image.Rect(tileWidth*w, tileHeight*h, tileWidth*(w+1), tileHeight*(h+1))
			images = append(images, img.(interface {
				SubImage(image.Rectangle) image.Image
			}).SubImage(rect))
		}
	}

	return images
}
```
image.Bounds() returnes a image.Rectangle which describes the rectangular area that defines
the valid pixel coordinates of the image.
```go
type Rectangle struct {
    Min, Max Point
}
```

```console
msg="Split to tiles" numTilesSize=(2,2) bounds=(0,0)-(1120,1120)
```
So this means that the rectangle is from (0,0) to (1120,1120):
```
 0,0
   +-----------------+
   |                 |
   |                 |
   |                 |
   |                 |
   |                 |
   +-----------------+ 1120,1120
```
```go
	width := b.Max.X - b.Min.X
	height := b.Max.Y - b.Min.Y
	tileHeight := height / numTilesSize.Y
	tileWidth := width / numTilesSize.X
```
So this is doing:
```
	width := 1120 - 0
    height := 1120 - 0
    tileHeight := 1120 / 2 = 560
    tileWidth := 1120 / 2 = 560
```
And we can verify this:
```console
msg="Image size" width=1120 height=1120
msg="Tile size" tileWidth=560 tileHeight=560
```
Next we have:
```go
	images := []image.Image{}

	for h := range numTilesSize.Y {
		for w := range numTilesSize.X {
			rect := image.Rect(tileWidth*w, tileHeight*h, tileWidth*(w+1), tileHeight*(h+1))
			images = append(images, img.(interface {
				SubImage(image.Rectangle) image.Image
			}).SubImage(rect))
		}
	}

	return images
```
So this is first creating an empty array of image.Image (one for each numTilesSize=(2,2)) and
numTileSize is of type Point:
```go 
type Point struct {
    X, Y int
}
```
So the above is iterating of the 2 y values and then for each or them the x values:
createing rectangles:
```
msg="Tile rect" rect=(0,0)-(560,560)
    0     560
  0 +-------+
    |       |
    |       |
560 +-------+

msg="Tile rect" rect=(560,0)-(1120,560)
    0     560     1120
  0 +-------+-------+
    |       |       |
    |       |       |
560 +-------+-------+

msg="Tile rect" rect=(0,560)-(560,1120)
    0     560     1120
  0 +-------+-------+
    |       |       |
    |       |       |
560 +-------+-------+
    |       |
    |       |
1120+-------+

msg="Tile rect" rect=(560,560)-(1120,1120)
    0     560     1120
  0 +-------+-------+
    |       |       |
    |       |       |
560 +-------+-------+
    |       |       |
    |       |       |
1120+-------+-------+
```
For each of these the following is done:
```go
			images = append(images, img.(interface {
				SubImage(image.Rectangle) image.Image
			}).SubImage(rect))
```
This is first asserting that `img` satisfies the interface:
```go
interface {
    SubImage(image.Rectangle) image.Image
}
```
This means that img must have a method SubImage(rect image.Rectangle) image.Image.
This allows the calling of SubImage(rect) which is one of the above rectangles. This method
extracts a rectanglular region from the original image. The original image in question here
is the resized and padded original image. Which is now split into tiles going fromt the top
left corner to the bottom right corner.

So with the sub images we will then do the following:
```go
	subImages := splitToTiles(img, aspectRatio)
	var pixelVals []float32

	rescale := true
	channelFirst := true

	for _, subImg := range subImages {
		vals := imageproc.Normalize(subImg, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, rescale, channelFirst)
		pixelVals = append(pixelVals, vals...)
	}

	return pixelVals
```
Notice that rescalse and channelFirst are both true. channelFirst means that all the Red values will
come first, then all the Green values and finally all the Blue values.
Also notice that imageproc.ClipDefaultMean and imageproc.ClipDefaultSTD are used for normalization:
```go
	ClipDefaultMean      = [3]float32{0.48145466, 0.4578275, 0.40821073}
	ClipDefaultSTD       = [3]float32{0.26862954, 0.26130258, 0.27577711}
```
So this will normalize each sub image separately and then append the values to the pixelVals array.
```go
func Normalize(img image.Image, mean, std [3]float32, rescale bool, channelFirst bool) []float32 {
    slog.Info("Normalize subimage", "mean", mean, "std", std, "channelFirst", channelFirst)
	var pixelVals []float32

	bounds := img.Bounds()
	if channelFirst {
		var rVals, gVals, bVals []float32
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				c := img.At(x, y)
				r, g, b, _ := c.RGBA()
				var rVal, gVal, bVal float32
				if rescale {
					rVal = float32(r>>8) / 255.0
					gVal = float32(g>>8) / 255.0
					bVal = float32(b>>8) / 255.0
				}

				rVal = (rVal - mean[0]) / std[0]
				gVal = (gVal - mean[1]) / std[1]
				bVal = (bVal - mean[2]) / std[2]

				rVals = append(rVals, rVal)
				gVals = append(gVals, gVal)
				bVals = append(bVals, bVal)
			}
		}

		pixelVals = append(pixelVals, rVals...)
		pixelVals = append(pixelVals, gVals...)
		pixelVals = append(pixelVals, bVals...)
	} else {
        ...
	}
	return pixelVals
}
```
So this is going to go through each sub image and for each pixel, which will be a vaue between 0 and 255
for each of the Red, Green and Blue channels, it will first rescale the value to be between 0 and 1 which
is down by dividing by 255. After that the mean is subtracted and the result devided by the standard deviation. And notice that there are arrays for the Red, Green and Blue values which are then appended to the pixelVals array which is the channel first part.
And finally all the Red values are appented, then all the Green values and finally all the Blue values.
```
	aspectRatioIndex := slices.Index(getSupportedAspectRatios(maxTiles), aspectRatio) + 1
	slog.Info("AspectRationIndex:", "aspectRatioIndex", aspectRatioIndex)
```
```console
msg="Resized image" aspectRatio=(2,2) bounds=(0,0)-(960,1120)
...
msg="Get supported aspect ratios" ratios="[(1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (3,1) (4,1)]"
                                             0     1     2     3     4     5     6     7

msg=AspectRationIndex: aspectRatioIndex=6
```
So we can see that this is looking up the index of (2,2) which is 5 and then adding 1 to get 6.
