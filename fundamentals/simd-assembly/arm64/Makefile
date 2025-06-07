CC = clang

SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/*.s)
TARGETS := $(patsubst $(SRCDIR)/%.s, %, $(SOURCES))

CFLAGS = -g -O0

.PHONY: all clean

all: $(TARGETS)

$(TARGETS): % : $(SRCDIR)/%.s | bindir
	$(CC) $(CFLAGS) -g -o $(BINDIR)/$@ $<

bindir: bin

bin:
	@mkdir -p $(BINDIR)

clean:
	$(RM) -rf $(BINDIR)
