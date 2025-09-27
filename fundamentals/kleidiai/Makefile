KLEIDIAI_DIR := kleidiai/install
CXXFLAGS := -I$(KLEIDIAI_DIR)/include/kai/ukernels \
            -I$(KLEIDIAI_DIR)/include/kai/ukernels/matmul \
            -I$(KLEIDIAI_DIR)/include/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp \
			-I$(KLEIDIAI_DIR)/include/kai/ukernels/matmul/pack \
			-I$(KLEIDIAI_DIR)/include \
			-march=native
LDFLAGS := -L$(KLEIDIAI_DIR)/lib -lkleidiai

kleidiai-clone:
	git clone https://github.com/ARM-software/kleidiai.git

kleidiai-update:
	cd kleidiai && git pull

kleidiai-build:
	cd kleidiai && cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=install \
      -S . -B build && \
	cmake --build ./build && \
	cmake --install ./build


simple: src/simple.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

features: src/features.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


smmla: src/smmla.s
	clang -mcpu=native -g3 -o $@ $<

.PHONY: clean
clean:
	${RM} simple features smmla
