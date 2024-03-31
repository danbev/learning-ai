CC = gcc
CFLAGS = -g -Wall -Iggml/include
LDFLAGS = -Lggml/build/src/

.PHONY: ggml-init
ggml-init:
	cd ggml && \
	rm -rf build && mkdir build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j8

.PHONY: ggml-show-build-settings
ggml-show-build-settings:
	cmake -L ggml/build

.PHONY: ggml
ggml:
	cd ggml/build && make -j8

all: $(TARGET)

graph: src/graph.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

tensor: src/tensor.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

matrix-mul: src/matrix-mul.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

get-rows: src/get_rows.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

reshape: src/reshape.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

transpose: src/transpose.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

view: src/view.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

permute: src/permute.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lggml -lm

show-add-dot:
	dot -Tpng add.dot -o add.dot.png && xdg-open add.dot.png

show-mul-dot:
	dot -Tpng mul.dot -o mul.dot.png && xdg-open mul.dot.png

show-rows-dot:
	dot -Tpng get-rows.dot -o get-rows.dot.png && xdg-open get-rows.dot.png

update-ggml:
	git submodule update --remote --merge ggml

.PHONY: clean
clean:
	${RM} -f graph tensor graph matrix-mul


clean-ggml:
	${RM} -rf ggml/build
