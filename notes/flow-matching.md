## Flow Matching
Lets take a look at this from the view point of a 2d grid. Lets say we have
a destination point (a vector), and another point which is a randomally chosen
point in the grid.

Now, the grid is a vector field, which means that for any position in the grid
there is a function that returns a vector which is the velocity of that point.

If we look at linear interpolation (lerp) we have:
```console
x_t = (1 - t) * x_0 + t * x_1
```
So lets say we are at the start, so time is 0:
```console
x_t = (1 - 0) * x_0 + 0 * x_1
x_t = 1 * x_0 + 0
x_t = x_0
```
This would be the complete noise.


And if we look at half way through time would then be 0.5:
```console
x_t = (1 - 0.5) * x_0 + 0.5 * x_1
x_t = 0.5 * x_0 + 0.5 * x_1
```
In this case we have a mix of both. This would be a blurry image.

And at the end of the time, so time is 1:
```console
x_t = (1 - 1) * x_0 + 1 * x_1
x_t = 0 * x_0 + 1 * x_1
x_t = x_1
```
And this would be the complete image.


So lets say we have our noise as [0, 0] and the target image as [10, 10].
If we go from t=0 => 0.1 => 0.2 => 0.3 => 0.4 => 0.5 => 0.6 => 0.7 => 0.8 => 0.9
we get:
```console
t=0: x_t = (1 - 0) * [0, 0] + 0 * [10, 10] = [0, 0]
t=0.1: x_t = (1 - 0.1) * [0, 0] + 0.1 * [10, 10] = [1, 1]
t=0.2: x_t = (1 - 0.2) * [0, 0] + 0.2 * [10, 10] = [2, 2]
t=0.3: x_t = (1 - 0.3) * [0, 0] + 0.3 * [10, 10] = [3, 3]
t=0.4: x_t = (1 - 0.4) * [0, 0] + 0.4 * [10, 10] = [4, 4]
t=0.5: x_t = (1 - 0.5) * [0, 0] + 0.5 * [10, 10] = [5, 5]
t=0.6: x_t = (1 - 0.6) * [0, 0] + 0.6 * [10, 10] = [6, 6]
t=0.7: x_t = (1 - 0.7) * [0, 0] + 0.7 * [10, 10] = [7, 7]
t=0.8: x_t = (1 - 0.8) * [0, 0] + 0.8 * [10, 10] = [8, 8]
t=0.9: x_t = (1 - 0.9) * [0, 0] + 0.9 * [10, 10] = [9, 9]
t=1.0: x_t = (1 - 1.0) * [0, 0] + 1.0 * [10, 10] = [10, 10]
```
And if we were to plot this, we would see a straight line from [0, 0] to [10, 10].
Notice that if we take the derivative of this, we get:
```console
 d
-- = [(1 - t) * x_0 + t * x_1] = -x_0 + x_1 = x_1 - x_0
dt
```

And also imagine that we have a function, that takes a vector and returnes a
new vector, which contains the velocity of that point. 

So we can pass in a vector and we will get a vector back, for examples we might
get [2, 2] back.  This tells us how to move forward, we could move 2 units in
the x direction and 2 units in the y direction. Now, if we calculate the
magnitude of this vector we get:
```console
magnitude = sqrt(2^2 + 2^2) = sqrt(8) = 2.83
```
And this is the speed at which we are moving. If we had [4, 4] we would get:
```console
magnitude([4, 4]) = sqrt(4² + 4²) = sqrt(32) ≈ 5.66
```
A larger vector means move further per unit time.

So this vector field is what the neural network is learning. It is learning to
take a vector and return a new vector that tells us how to move forward.
The network's job is: given any point xt and time t, predict the velocity
vector at that point.

This sounds like we could just sample a random point and then follow the 
vector field, but if we want to generate a cat how do we know which direction
to go?
So how this works is that we also pass in the token embedding vector for the
prompt, which is the conditioning:
```console
Vec2 network_predict(vec2 xt, float t, embedding prompt);
```
So the vector field is different for different prompts. The hidden space itself
does not change, a cat is still in the same space as it would be regardless of
the prompt. It is the velocities that changes.
_wip_
