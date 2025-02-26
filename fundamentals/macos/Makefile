CXXFLAGS = -g -Wall -std=c++20
OUT = bin

$(OUT)/features: src/features.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OUT): 
	mkdir -p $(OUT)

.PHONY: clean
clean:
	${RM} -r $(OUT)
