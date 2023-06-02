## AI/ML background
This document contains notes about concepts related to AI/ML. 

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

