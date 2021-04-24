NVCC=nvcc

SRC_DIR=src/
INC_DIR=include/
CUDA_FLAG= -std=c++14 -I ${INC_DIR}

TARGET=main
TARGET_SRC=${SRC_DIR}${TARGET}.cu
TARGET_OBJ=$(TARGET_SRC:.cu=.o)

CXX_SRC_LIB=${TARGET_SRC} ${SRC_DIR}tensor.cu
OBJ_SRC_LIB=$(CXX_SRC_LIB:.cu=.o)

all: ${TARGET}

${TARGET}: ${OBJ_SRC_LIB}
	${NVCC} ${CUDA_FLAG} -o $@ $^

${TARGET_OBJ}:
	${NVCC} ${CUDA_FLAG} -c ${TARGET_SRC} -o ${TARGET_OBJ}

${SRC_DIR}tensor.o:
	${NVCC} ${CUDA_FLAG} -c ${SRC_DIR}tensor.cu -o ${SRC_DIR}tensor.o

clean:
	rm ${TARGET}
	rm ${OBJ_SRC_LIB}
