.PHONY: build
build: 
	cmake -S . -B build
	cmake --build build -v

show-preprocessor-output:
	${CXX} -Ipybind11/include $(shell python3-config --includes) -E src/module.cpp

show-symbols:
	nm -C build/pybind_example.cpython-312-x86_64-linux-gnu.so | c++filt

show-dyn-symbols:
	nm -D build/pybind_example.cpython-312-x86_64-linux-gnu.so | c++filt

.PHONY: test
test:
	./test/pybind-example-test.py

.PHONY: clean
clean: 
	${RM} -rf build
