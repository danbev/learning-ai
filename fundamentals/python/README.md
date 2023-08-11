## AI/ML Fundamentals in Python
The idea of the code is for learning and exploration and as a reference for
later to refresh my knowledge later.

### Virtual environment
First, create a virtual environment:
```console
$ python3 -m venv fund
```

To activate it, run the following command:
```console
$ source fund/bin/activate
```

Then install the requirements:
```console
(fund) $ pip install -r requirements.txt
```

### Autograd (automatic gradient)
This example comes from a Andrej Karpathy's youtube video on
[autograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=577s) modified with
comments and a few changes. This automatic gradient functionality is something
that I believe is used in most machine learning libraries, like tensorflow and
pytorch. Going through this example is a good way to understand how it works.

### Bigrams
This example comes from a Andrej Karpathy's youtube video on
[bigrams](https://www.youtube.com/watch?v=PaCmpygFfXo) modified with comments
and a few changes. This is a good start to learning about language models.
