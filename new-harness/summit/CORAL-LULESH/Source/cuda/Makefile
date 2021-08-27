MPI_HOME = ${MPI_ROOT}

MPI_FLAGS = -DUSE_MPI -I$(MPI_HOME)/include

NVCC		= nvcc
FLAGS		= -arch=sm_70
DFLAGS	= $(MPI_FLAGS) -lineinfo -G -g
RFLAGS 	= $(MPI_FLAGS) -O3 -DNDEBUG -Xptxas -v

LINKFLAGS = -lmpi_ibm -L$(MPI_HOME)/lib -lopen-pal -lopen-rte -lhwloc# -L/home/hfwen/debug/trap -ltrap -Wl,--rpath=/home/hfwen/debug/trap

all: release

debug: LINKFLAGS +=

release: 	FLAGS += $(RFLAGS)
debug: 		FLAGS += $(DFLAGS)

release: lulesh
debug: lulesh

lulesh: allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o
	$(NVCC) $(LINKFLAGS) allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o lulesh

allocator.o: allocator.cu vector.h
	$(NVCC) $(FLAGS) allocator.cu -I ./ -c -o allocator.o

lulesh.o: lulesh.cu util.h vector.h allocator.h
	$(NVCC) $(FLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o lulesh.o

lulesh-comms.o: lulesh-comms.cu
	$(NVCC) $(FLAGS) lulesh-comms.cu -I ./ -c -o lulesh-comms.o

lulesh-comms-gpu.o: lulesh-comms-gpu.cu
	$(NVCC) $(FLAGS) lulesh-comms-gpu.cu -I ./ -c -o lulesh-comms-gpu.o

clean:
	rm -rf allocator.o lulesh-comms.o lulesh-comms-gpu.o lulesh.o lulesh xyz.asc regNumList.txt
