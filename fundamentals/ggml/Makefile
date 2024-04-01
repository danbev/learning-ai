CC = gcc
CFLAGS = -g -Wall -Iggml/include
LDFLAGS = -Lggml/build/src/

SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/*.c)
TARGETS := $(patsubst $(SRCDIR)/%.c, %, $(SOURCES))

.PHONY: all clean

all: $(TARGETS)

$(TARGETS): % : $(SRCDIR)/%.c | bindir
	$(CC) $(CFLAGS) $(LDFLAGS) -lggml -o ${BINDIR}/$@ $<

bindir: bin

bin: 
	@mkdir -p $(BINDIR)

clean:
	${RM} -rf $(BINDIR)

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

show-add-dot:
	dot -Tpng add.dot -o add.dot.png && xdg-open add.dot.png

show-mul-dot:
	dot -Tpng mul.dot -o mul.dot.png && xdg-open mul.dot.png

show-rows-dot:
	dot -Tpng get-rows.dot -o get-rows.dot.png && xdg-open get-rows.dot.png

update-ggml:
	git submodule update --remote --merge ggml

clean-ggml:
	${RM} -rf ggml/build
