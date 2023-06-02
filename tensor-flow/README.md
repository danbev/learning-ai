## Tensor flow example
Tensor flow is a platform for creating and using ML models. It contains a number
common algorithms and patterns for ML and also supports ways to deploy models
to the web, cloud, and to embedded systems.

### Create virtual environment
```console
$ python3 -m venv tf 
```

### Activate the environment
```console
$ source tf/bin/activate
```

### Install requirements
```console
$ source tf/bin/activate
(tf) pip install -r requirements
```

### Install Tensor flow (first time only)
```console
(tf) python -m pip install tensorflow
```
Run pip freeze to generate a requirements.txt file to check in.
```console
(tf) $ pip freeze > requirements.txt
```

### Training models
Training the model can be done by simply writing all the code ourselfes, or by
using built-in estimators, or alternatively using [Keras] 

[Keras]: https://keras.io/


### Data Services
This is an API in Tensor Flow that helps manage data. For training we will need
a lot of example data and this needs to be managed somehow.
This service also contains/includes preprocessed datasets which can be used.

### Serving
Is a Tensor Flow API to provide model inference over HTTP.
For embedded devices there is Tensor Flow Lite that provides tools more model
inference on Android, iOS and Linux based embedded systems. For more constrained
embedded devices, which might not have an operating system and hence are not
using Linux, there is Tensor Flow Lite Micro (TFLM) which is related to TinyML.
For Node.js there is TensorFlow.js which provides the ability to train and
execute models.
