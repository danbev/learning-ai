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

### Debugging
```console
$ gdb python3
#
# Defining a custom gdb command (basically an alias to json dump())
Reading symbols from python3...
(No debugging symbols found in python3)
(gdb) set args test/pybind-example-test.py
(gdb) break add
(gdb) br PyInit_pybind_example
Function "add" not defined.
Breakpoint 1 (add) pending.
(gdb) r
Starting program: /usr/bin/python3 test/pybind-example-test.py
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
✓ Import successful!

Available in module:
  add: <class 'builtin_function_or_method'>
  subtract: <class 'builtin_function_or_method'>

Breakpoint 1, add (i=10, j=20) at /home/danbev/work/ai/learning-ai/fundamentals/pytorch/pybind-example/src/module.cpp:4
4	    return i + j;
```

```console
(gdb) bt
#0  add (i=10, j=20) at /home/danbev/work/ai/learning-ai/fundamentals/pytorch/pybind-example/src/module.cpp:4
#1  0x00007ffff7ac070f in pybind11::detail::argument_loader<int, int>::call_impl<int, int (*&)(int, int), 0ul, 1ul, pybind11::detail::void_type>(int (*&)(int, int), std::integer_sequence<unsigned long, 0ul, 1ul>, pybind11::detail::void_type&&) && (
    this=0x7fffffffce10, f=@0xc4a4f8: 0x7ffff7a93d81 <add(int, int)>)
    at /home/danbev/work/ai/learning-ai/fundamentals/pytorch/pybind-example/pybind11/include/pybind11/cast.h:2132
#2  0x00007ffff7abd48c in pybind11::detail::argument_loader<int, int>::call<int, pybind11::detail::void_type, int (*&)(int, int)>(int (*&)(int, int)) && (this=0x7fffffffce10, f=@0xc4a4f8: 0x7ffff7a93d81 <add(int, int)>)
    at /home/danbev/work/ai/learning-ai/fundamentals/pytorch/pybind-example/pybind11/include/pybind11/cast.h:2100
#3  0x00007ffff7ab8348 in pybind11::cpp_function::initialize<int (*&)(int, int), int, int, int, pybind11::name, pybind11::scope, pybind11::sibling, char [86]>(int (*&)(int, int), int (*)(int, int), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, char const (&) [86])::{lambda(pybind11::detail::function_call&)#1}::operator()(pybind11::detail::function_call&) const (
    __closure=0x0, call=...)
    at /home/danbev/work/ai/learning-ai/fundamentals/pytorch/pybind-example/pybind11/include/pybind11/pybind11.h:429
#4  0x00007ffff7ab83b1 in pybind11::cpp_function::initialize<int (*&)(int, int), int, int, int, pybind11::name, pybind11::scope, pybind11::sibling, char [86]>(int (*&)(int, int), int (*)(int, int), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, char const (&) [86])::{lambda(pybind11::detail::function_call&)#1}::_FUN(pybind11::detail::function_call&) ()
    at /home/danbev/work/ai/learning-ai/fundamentals/pytorch/pybind-example/pybind11/include/pybind11/pybind11.h:400
#5  0x00007ffff7aa4ac8 in pybind11::cpp_function::dispatcher (self=0x7ffff7b19f90, args_in=0x7ffff7133a00, kwargs_in=0x0)
    at /home/danbev/work/ai/learning-ai/fundamentals/pytorch/pybind-example/pybind11/include/pybind11/pybind11.h:1064
#6  0x0000000000581d4f in ?? ()
#7  0x0000000000548f85 in _PyObject_MakeTpCall ()
#8  0x00000000005d6b2f in _PyEval_EvalFrameDefault ()
#9  0x00000000005d500b in PyEval_EvalCode ()
#10 0x00000000006081e2 in ?? ()
#11 0x00000000006b5033 in ?? ()
#12 0x00000000006b4d9a in _PyRun_SimpleFileObject ()
#13 0x00000000006b4bcf in _PyRun_AnyFileObject ()
#14 0x00000000006bcc35 in Py_RunMain ()
#15 0x00000000006bc71d in Py_BytesMain ()
#16 0x00007ffff7c2a1ca in __libc_start_call_main (main=main@entry=0x518950, argc=argc@entry=2, argv=argv@entry=0x7fffffffd808)
    at ../sysdeps/nptl/libc_start_call_main.h:58
#17 0x00007ffff7c2a28b in __libc_start_main_impl (main=0x518950, argc=2, argv=0x7fffffffd808, init=<optimized out>,
    fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7fffffffd7f8) at ../csu/libc-start.c:360
#18 0x00000000006575a5 in _start ()
```
There is quite a bit of overhead in the call stack, and for function that does
very little like `add(int i, int j)`, this overhead can be significant. But when
the c++ function does more work, the overhead becomes negligible, and this is
where the performance of PyBind11 shines.
```console
#0  add (i=10, j=20)                    ← YOUR CODE (~1-5ns)
#1  argument_loader::call_impl          ← Type conversion (~30-50ns)
#2  argument_loader::call               ← Argument processing (~20-30ns)
#3  cpp_function::initialize            ← Wrapper logic (~20-40ns)
#4  cpp_function::dispatcher            ← Pybind11 dispatch (~30-60ns)
#5  _PyObject_MakeTpCall                ← Python C API (~20-40ns)
#6  _PyEval_EvalFrameDefault            ← Python interpreter (~10-20ns)
```
This is a reason that PyTorch operates on whole tensors and not individual
elements.

Python → C++ conversion requires:
- Type checking: Is this really an int?
- Object conversion: Python int → C++ int
- Memory management: Reference counting, cleanup
- Error handling: Exception translation
- Result conversion: C++ int → Python int
- Object creation: New Python integer object

Conceptually, Python does this:
```python
def python_call_add(py_int1, py_int2):
    # 1. Validate types
    if not isinstance(py_int1, int) or not isinstance(py_int2, int):
        raise TypeError("Arguments must be integers")

    # 2. Convert Python objects to C++ values
    cpp_int1 = extract_c_int_from_python_object(py_int1)
    cpp_int2 = extract_c_int_from_python_object(py_int2)

    # 3. Call your function (same as GDB!)
    cpp_result = add(cpp_int1, cpp_int2)

    # 4. Convert result back to Python object
    py_result = create_python_int_object(cpp_result)

    return py_result
```
