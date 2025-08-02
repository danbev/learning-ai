## Pybind11 Example
This example is only here to better understand how pytorch c/c++ functions work
as it also used PyBind11 to create a Python module like nn.functional does.

Understanding this in PyTorch was the motivation for exploring this:
```console
linear = add_docstr(
    torch._C._nn.linear,  # ← C++ function already exists
    r"""
    linear(input, weight, bias=None) -> Tensor
    Applies a linear transformation...
    """
)
```

### Building
```console
$ make build
```
This will produce a shared library:
```console
build/pybind_example.cpython-312-x86_64-linux-gnu.so 
```

### Running
If we start a Python REPL in the `build` directory, we can import the module:
```console
(venv) $ cd build/
(venv) $ python3
Python 3.12.3 (main, Jun 18 2025, 17:59:45) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pybind_example
>>> pybind_example.add(1, 3)
4
```

### Show preprocessor output
```console
$ make show-preprocessor-output
```
So lets look how the `PYBIND11_MODULE` macro gets expanded:
```console
# 2 "src/module.cpp" 2

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

static int pybind11_exec_pybind_example(PyObject *);

extern "C" [[maybe_unused]] __attribute__((visibility("default"))) PyObject *PyInit_pybind_example();
extern "C" __attribute__((visibility("default"))) PyObject *PyInit_pybind_example() { 
    {
        const char *compiled_ver = "3" "." "12";
        const char *runtime_ver = Py_GetVersion();
        size_t len = std::strlen(compiled_ver);
        if (std::strncmp(runtime_ver, compiled_ver, len) != 0 || (runtime_ver[len] >= '0' && runtime_ver[len] <= '9')) {
            PyErr_Format(PyExc_ImportError, "Python version mismatch: module was compiled for Python %s, " "but the interpreter version is incompatible: %s.", compiled_ver, runtime_ver);
            return nullptr;
        }
    }
    (pybind11::detail::get_num_interpreters_seen() += 1);
    {
        pybind11::detail::get_internals_pp_manager().unref();
        pybind11::detail::get_internals();
    }
    static ::pybind11::detail::slots_array slots = ::pybind11::detail::init_slots( &pybind11_exec_pybind_example);
    static PyModuleDef def{{ { { 1 }, (nullptr) }, nullptr, 0, nullptr, }, "pybind_example", nullptr, 0, nullptr, slots.data(), nullptr, nullptr, nullptr};
    return PyModuleDef_Init(&def);
}

static void pybind11_init_pybind_example(::pybind11::module_ &);
int pybind11_exec_pybind_example(PyObject * pm) {
    try {
        auto m = pybind11::reinterpret_borrow<::pybind11::module_>(pm);
        pybind11_init_pybind_example(m);
        return 0;
    } catch (pybind11::error_already_set & e) {
        pybind11::raise_from(e, PyExc_ImportError, "initialization failed");
    } catch (const std::exception &e) {
        ::pybind11::set_error(PyExc_ImportError, e.what());
    }
    return -1;
}

void pybind11_init_pybind_example(::pybind11::module_ & m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: pybind_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

}
```
So  when we do:
```python
import pybind_example
```
Python's import machinery:
- Searches sys.path for pybind_example.*
- Finds pybind_example.cpython-312-x86_64-linux-gnu.so
- Recognizes it as a C extension module
- Uses dlopen() to load the shared library
- Looks for symbol "PyInit_pybind_example"
```
So it will then call `PyInit_pybind_example` which will setup the module
for execution.
Python's module system then calls the bridge function `pybind11_exec_pybind_example`

When m.def("add", &add, "...") executes:
- Creates wrapper function that converts Python ↔ C++ types
- Registers in module's method table with name "add"
- Stores docstring for help() function
- Links to your C++ function add(int, int)
