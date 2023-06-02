## AI/ML background
This document contains notes about concepts related to AI/ML. 

### ...
Lets say that we have a neural network with two input values:
```
 +----+             +----+
 | x₁ |------------>| y₁ |------> y'₁
 +----+             +----+
    |                 ↑
    ↓         +-------+
    +---------|-------+
    +---------+       |
    ↑                 ↓
 +----+             +----+
 | x₂ |------------>| y₁ |------> y'₂
 +----+             +----+
```
In this case y'₁ will be:
```
y'₁ = activation_function(w₁₁x₁ + w₁₂x₂ + b₁)
y'₂ = activation_function(w₂₁x₁ + w₂₂x₂ + b₁)
```
If we write this in vector an matrix form we get:
```
 xs = [x₁ x₂]
 ws = [w₁₁ w₁₂]
      [w₂₁ w₂₂]
```
And the layer can then be seen as a matrix multiplication and addition of the
bias:
```
 [w₁₁ w₁₂][x₁] + [b₁] = [w₁₁x₁ + w₁₂x₂ + b₁] = [y'₁]
 [w₂₁ w₂₂][x₂] + [b₂]   [w₂₁x₁ + w₂₁x₂ + b₂]   [y'₂]
```
So what if we only had one neuron in the hidden layer:
```
 +----+             +----+
 | x₁ |------------>| y₁ |------> y'₁
 +----+             +----+
                      ↑
              +-------+
              |        
    +---------+        
    ↑                  
 +----+
 | x₂ |
 +----+
```
We have to remember that we have a cooefficiant for each x value:
```
y₁ = w₁*x₁ + w₂*x₂ + b₁
```
So this would become:
```
 [w₁₁ w₁₂][x₁] + [b₁] = [w₁₁x₁ + w₁₂x₂ + b₁] = [y'₁]
          [x₂]          
```
Notice here that matrix multiplication still works as the number of colums in
the matrix match the number of rows of the input vector.

### Learning (creating the ML model)
Tranditionally programming would specify rules in functions and data would be
the input. And the output the computation of the rules on the data.

Tranditional programming:
```
Data        +----------+          Answers
----------->|   Rules  |--------->
            |          |
            +----------+
```

With machine learning we provide answers and data to the function and the output
are the rules.

Machine learning:
```
Data        +----------+          Rules (Model)
----------->| Train    |--------->
Anwers      | Model    |
            +----------+
```

So we take data and then we label the data points, that is we
specify what they are. For example, lets say that we want to recoginize human
emotions by looking at photos of faces (expressions). We would train a model
by passing it images faces that are each tagged/labeled with a specific emotion,
like happy, sad, angry, etc. Notice that we this data must have been classified
by someone/something to add those tags/lables. So our model is something that
has been trained to recoginize facial expressions.

And we can use that to try to figure out the expression/emotions of unseen
images of faces, and this is called `inference`.

The training can be done on a CPU, a GPU (Graphical Processing Unit), or a TPU
(Tensor Processing Unit). 

Inference:
```
Data        +----------+          Answer
----------->| Model    |---------> (Happy | Sad | Angry | ...)
            |          |
            +----------+
```
So data could be a photo of a human face which is passed to the model that we
have trained to classify emotions.

I think that model inference can be performed on an constrained device if it has
enough storage and processing powers. So we could imagine running the inference
close to the user, which could be a system. Like we might be able to place the
inference on an IoT device and have the inference done there with out having to
send the data to a central server to be processed.

### Linear regression
Lets take a simple example where we have the following known data points:
```
xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
```
The known values are the xs and ys. That is, for each x value we know the
expected y value.
```
Input Layer      Layer 1          Output Layer
   X --------->  y = mx + b  ---> Y_guess <---------> Y_actual
```
                                        
The first time we run this the values of `m` and `b` are just guesses, for
example:
```
   -1.0 --->  y = 10*(-1.0) + 10  ---> 0  <------> -3.0
    0.0 --->  y = 10*( 0.0) + 10  ---> 10 <------> -1.0
    1.0 --->  y = 10*( 1.0) + 10  ---> 20 <------>  1.0
    2.0 --->  y = 10*( 2.0) + 10  ---> 30 <------>  3.0
    3.0 --->  y = 10*( 3.0) + 10  ---> 40 <------>  5.0
    4.0 --->  y = 10*( 4.0) + 10  ---> 50 <------>  7.0
```
The loss function is what determines the calculated y values with the know
y values. And depending on the values returned from the loss function the
parameters to the function, called the optimizer, in the layer are adjusted.
The this repeates until the values from the loss function are close to the
expected know values.


A more realistic example could be using the Fashion MNIST where each image is
28x28 pixels (784), and each pixel can have a value of between 0-255 (our x
values):
```
Pixel    Value
    0    254
    1    202
    2    5
  ...    ...
  784    18
```
And these are to be classified into 10 different categories (our y values).

Now, in the previous example we had something like this:
```
             +-----------+
X -------->  | Learn m,b | --------> Y = mX+b
             +-----------+
```
Where we had a single value for x. I this new case we 784 input values instead
of one value:
```
0   --------|   n₀(m₀,b₀)  |                                          y₀
1   --------|   n₁(m₁,b₁)  |                                          y₁
2   --------|-> n₁(m₂,b₂)->|--------> y'₀ + y'₁+ y'₂ + ... y'₈ <----> y₂
...
783 --------|   n₈(m₈,b₈)  |                                          y₈
```
So we know the number of neurons in the input layer which is 28x28 (784), and
we know that the output layer will have 10 neurons. But what about the layer, or
layers in between?  By the way these are called hidden layers as they are not
"visible" to the caller, who only "sees" the input the output layers. The number
hidden layers and how many neurons they contains can be arbitarlity chosen and
is part of training I think to figure out they appropriate number.

