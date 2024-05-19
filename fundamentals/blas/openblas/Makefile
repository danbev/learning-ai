CC = gcc
SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/*.c)
TARGETS := $(patsubst $(SRCDIR)/%.c, %, $(SOURCES))

all: $(TARGETS)

$(TARGETS): % : $(SRCDIR)/%.c | bindir
	$(CC) -o ${BINDIR}/$@ $< -lopenblas

bindir: bin

bin: 
	@mkdir -p $(BINDIR)

.PHONY: clean
clean:
	@${RM} -rf $(BINDIR)
