## vLLM example
Examples of using the vLLM library.

### Building
```console
$ python -m venv venv
$ source venv/bin/activate
$ pip install vllm

### Running
```console
$ python src/simple-prompt.py
```

### Documentation
```console
$ python -m pydoc vllm.LLM
```
Or interactively:
```console
$ python
>>> from vllm import VLLM
>>> help(VLLM)
```

