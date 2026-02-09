CXX = g++
CXXFLAGS = -I/opt/xilinx/xrt/include -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -pthread

npu_check: src/npu_check.cpp
	@${CXX} $< ${CXXFLAGS} -o $@

run_check: npu_check
	@env LD_LIBRARY_PATH=/opt/xilinx/xrt/lib ./npu_check

.PHONY: clean
clean:
	@${RM} -rf npu_check
