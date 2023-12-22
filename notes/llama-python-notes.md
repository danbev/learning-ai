Using Fedora 39 which comes with Python 3.12.0 trying to create a virtual
environment to run convert-hf-to-gguf.py fails with the following error:

```console
$ python3 -m venv venv2
$ source venv2/bin/activate
(venv2)
```

The version of pip is as follows:
```console
(venv2) $ pip --version
pip 23.2.1 from /home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip (python 3.12)
```

Then, installing the requirement fails:
```console
(venv2) $ pip install -r requirements-hf-to-gguf.txt 
Collecting numpy==1.24.4 (from -r requirements.txt (line 1))
  Using cached numpy-1.24.4.tar.gz (10.9 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
ERROR: Exception:
Traceback (most recent call last):
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/cli/base_command.py", line 180, in exc_logging_wrapper
    status = run_func(*args)
             ^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/cli/req_command.py", line 248, in wrapper
    return func(self, options, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/commands/install.py", line 377, in run
    requirement_set = resolver.resolve(
                      ^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 92, in resolve
    result = self._result = resolver.resolve(
                            ^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
    state = resolution.resolve(requirements, max_rounds=max_rounds)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
    self._add_to_criteria(self.state.criteria, r, parent=None)
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria
    if not criterion.candidates:
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__
    return bool(self._sequence)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 155, in __bool__
    return any(self)
           ^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 143, in <genexpr>
    return (c for c in iterator if id(c) not in self._incompatible_ids)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 47, in _iter_built
    candidate = func()
                ^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 208, in _make_candidate_from_link
    self._link_candidate_cache[link] = LinkCandidate(
                                       ^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 293, in __init__
    super().__init__(
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 156, in __init__
    self.dist = self._prepare()
                ^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 225, in _prepare
    dist = self._prepare_distribution()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 304, in _prepare_distribution
    return preparer.prepare_linked_requirement(self._ireq, parallel_builds=True)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/operations/prepare.py", line 538, in prepare_linked_requirement
    return self._prepare_linked_requirement(req, parallel_builds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/operations/prepare.py", line 653, in _prepare_linked_requirement
    dist = _get_prepared_distribution(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/operations/prepare.py", line 69, in _get_prepared_distribution
    abstract_dist.prepare_distribution_metadata(
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/distributions/sdist.py", line 48, in prepare_distribution_metadata
    self._install_build_reqs(finder)
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/distributions/sdist.py", line 118, in _install_build_reqs
    build_reqs = self._get_build_requires_wheel()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/distributions/sdist.py", line 95, in _get_build_requires_wheel
    return backend.get_requires_for_build_wheel()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_internal/utils/misc.py", line 697, in get_requires_for_build_wheel
    return super().get_requires_for_build_wheel(config_settings=cs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 166, in get_requires_for_build_wheel
    return self._call_hook('get_requires_for_build_wheel', {
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 321, in _call_hook
    raise BackendUnavailable(data.get('traceback', ''))
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Traceback (most recent call last):
  File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 77, in _build_backend
    obj = import_module(mod_path)
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib64/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1381, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1354, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1304, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1381, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1354, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1325, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 929, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 994, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/pip-build-env-27ay0hcs/overlay/lib/python3.12/site-packages/setuptools/__init__.py", line 10, in <module>
    import distutils.core
ModuleNotFoundError: No module named 'distutils'


[notice] A new release of pip is available: 23.2.1 -> 23.3.2
[notice] To update, run: pip install --upgrade pip

```

I've upgraded pip just in case but the error remains
```console
(venv2) $ pip install --upgrade pip
```

distutils is a core Python module used in building and installing Python
packages. It's crucial for the functioning of pip and setuptools when compiling
and installing packages from source.

Python 3.12 remove distutils since it has been deprecated since 3.10.

setuptools now also provides distutils.
So we should be able to install setuptools:
```console
(venv2) python -m pip install --upgrade setuptools
```

```console
(venv2) $ pip list
Package    Version
---------- -------
pip        23.3.2
setuptools 69.0.2
```
So we can then run the install again:
```console
(venv2) $ pip install -r requirements-hf-to-gguf.txt 
Collecting numpy==1.24.4 (from -r requirements.txt (line 1))
  Using cached numpy-1.24.4.tar.gz (10.9 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [33 lines of output]
      Traceback (most recent call last):
        File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 353, in <module>
          main()
        File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 335, in main
          json_out['return_val'] = hook(**hook_input['kwargs'])
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 112, in get_requires_for_build_wheel
          backend = _build_backend()
                    ^^^^^^^^^^^^^^^^
        File "/home/danielbevenius/work/ai/llama.cpp/venv2/lib64/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 77, in _build_backend
          obj = import_module(mod_path)
                ^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/lib64/python3.12/importlib/__init__.py", line 90, in import_module
          return _bootstrap._gcd_import(name[level:], package, level)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<frozen importlib._bootstrap>", line 1381, in _gcd_import
        File "<frozen importlib._bootstrap>", line 1354, in _find_and_load
        File "<frozen importlib._bootstrap>", line 1304, in _find_and_load_unlocked
        File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
        File "<frozen importlib._bootstrap>", line 1381, in _gcd_import
        File "<frozen importlib._bootstrap>", line 1354, in _find_and_load
        File "<frozen importlib._bootstrap>", line 1325, in _find_and_load_unlocked
        File "<frozen importlib._bootstrap>", line 929, in _load_unlocked
        File "<frozen importlib._bootstrap_external>", line 994, in exec_module
        File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
        File "/tmp/pip-build-env-r1_zob0f/overlay/lib/python3.12/site-packages/setuptools/__init__.py", line 16, in <module>
          import setuptools.version
        File "/tmp/pip-build-env-r1_zob0f/overlay/lib/python3.12/site-packages/setuptools/version.py", line 1, in <module>
          import pkg_resources
        File "/tmp/pip-build-env-r1_zob0f/overlay/lib/python3.12/site-packages/pkg_resources/__init__.py", line 2172, in <module>
          register_finder(pkgutil.ImpImporter, find_on_path)
                          ^^^^^^^^^^^^^^^^^^^
      AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```

```console
(venv2) $ pip install --upgrade setuptools wheel
Requirement already satisfied: setuptools in ./venv2/lib64/python3.12/site-packages (69.0.2)
Collecting wheel
  Using cached wheel-0.42.0-py3-none-any.whl.metadata (2.2 kB)
Using cached wheel-0.42.0-py3-none-any.whl (65 kB)
Installing collected packages: wheel
Successfully installed wheel-0.42.0
(venv2) $ pip list
Package    Version
---------- -------
pip        23.3.2
setuptools 69.0.2
wheel      0.42.0
```

```console
$ virtualenv --upgrade-embed-wheels
```

```console
pip install --upgrade virtualenv
```

pip3.12 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu


I ended up installing python 3.11 after spending way too much time on this:
```console
$ sudo dnf install python3.11
```
And then using python 3.11 to create the new virtual environment:
```
$ python3.11 -m venv venv3
```
After that is was possible to install the requirements.
