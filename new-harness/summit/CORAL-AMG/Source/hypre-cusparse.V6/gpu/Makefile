include ../Makefile.include

.SUFFIXES:
.SUFFIXES: .cu .o
 
RANLIB = ranlib

#CUDA_PATH = /usr/local/cuda
CUDA_PATH=$(dir $(shell which nvcc))../

NVCC := nvcc 
NVCC_FLAGS = -Xptxas -dlcm=cg -m64 -use_fast_math -O3 \
	-gencode=arch=compute_70,code=sm_70 \
	$(OPT_FLAGS)
# release mode
NVCC_FLAGS += -DNDEBUG

# debug mode
#NVCC_FLAGS += -g -G 

INCLUDES := -I$(CUDA_PATH)/include
#LIBRARIES := -L$(CUDA_PATH)/lib64 -lcublas -lcusparse -lcudart 

SOURCES =\
 gpu_matrix.cu\
 gpu_matvec.cu\
 gpu_vector.cu\
 gpu_utilities.cu

OBJECTS = ${SOURCES:.cu=.o}

all: libamg_gpu.a

clean:
	@rm -f *.o
 
veryclean: clean
	@rm -f libamg_gpu.a
 
# assemble the library

libamg_gpu.a: ${OBJECTS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJECTS}
	${RANLIB} $@

# build object files

gpu_matrix.o: gpu_matrix.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

gpu_matvec.o: gpu_matvec.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

gpu_vector.o: gpu_vector.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

gpu_utilities.o: gpu_utilities.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

