OPENVINO_DIR = /home/danielbevenius/work/ai/openvino/openvino_dist

OPENVINO_INCLUDE=${OPENVINO_DIR}/runtime/include
OPENVINO_LIBRARY_PATH=${OPENVINO_DIR}/runtime/lib/intel64

OPENVINO_PKG_CONFIG_PATH=${OPENVINO_LIBRARY_PATH}/pkgconfig
export PKG_CONFIG_PATH=${OPENVINO_PKG_CONFIG_PATH}

CXX = g++
CXXFLAGS = `pkg-config --cflags openvino`
CFLAGS = `pkg-config --cflags openvino`
LDFLAGS = `pkg-config --libs openvino`

print-version: src/print-version.cpp
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o $@ $<

print-version-c: src/print-version.c
	${CC} ${CFLAGS} ${LDFLAGS} -o $@ $<


openvino-pkg-config-libs:
	@env PKG_CONFIG_PATH=${OPENVINO_PKG_CONFIG_PATH} pkg-config --libs openvino

openvino-pkg-config-cflags:
	@env PKG_CONFIG_PATH=${OPENVINO_PKG_CONFIG_PATH} pkg-config --cflags openvino

.PHONY: clean
clean:
	${RM} print-version
