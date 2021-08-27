#default build suggestion of MPI + OPENMP with gcc on Livermore machines you might have to change the compiler name

SHELL = /bin/sh
.SUFFIXES: .cc .o

LULESH_EXEC = lulesh2.0

MPI_HOME = ${MPI_ROOT}
MPI_INC = $(MPI_HOME)/include
MPI_LIB =

SERCXX = xlC_r -DUSE_MPI=0
MPICXX = $(MPI_HOME)/bin/mpicxx -DUSE_MPI=1 #-DOMPI_SKIP_MPICXX
CXX = $(MPICXX)

SOURCES2.0 = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc
OBJECTS2.0 = $(SOURCES2.0:.cc=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -g -Ofast   -qarch=pwr9 -qtune=pwr9 -qsmp=omp -qoffload -qdebug=inlinegpurt  -qtgtarch=sm_70 -DUSE_UVM -DUSE_NVTX -DCalcHydroConstraintForElems_SHORTCUT -DCOMM_OPT -DUSE_DEVICE=1 -std=c++11 -I${CUDA_DIR}/include 
LDFLAGS = -g -o  -Ofast -qarch=pwr9 -qtune=pwr9 -qsmp=omp -qoffload -qdebug=inlinegpurt  -qtgtarch=sm_70 -std=c++11 -L${CUDA_DIR}/lib64 -lcuda -lcudart ${CUDA_DIR}/lib64/libnvToolsExt.so


#Below are reasonable default flags for a serial build
#CXXFLAGS = -g -O3 -I. -Wall
#LDFLAGS = -g -O3 

#common places you might find silo on the Livermore machines.
#SILO_INCDIR = /opt/local/include
#SILO_LIBDIR = /opt/local/lib
#SILO_INCDIR = ./silo/4.9/1.8.10.1/include
#SILO_LIBDIR = ./silo/4.9/1.8.10.1/lib

#If you do not have silo and visit you can get them at:
#silo:  https://wci.llnl.gov/codes/silo/downloads.html
#visit: https://wci.llnl.gov/codes/visit/download.html

#below is and example of how to make with silo, hdf5 to get vizulization by default all this is turned off.  All paths are Livermore specific.
#CXXFLAGS = -g -DVIZ_MESH -I${SILO_INCDIR} -Wall -Wno-pragmas
#LDFLAGS = -g -L${SILO_LIBDIR} -Wl,-rpath -Wl,${SILO_LIBDIR} -lsiloh5 -lhdf5

.cc.o: lulesh.h
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(LULESH_EXEC)

lulesh2.0: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -lm -o $@

clean:
	/bin/rm -f *.o *~ $(OBJECTS) $(LULESH_EXEC)
	/bin/rm -rf *.dSYM

tar: clean
	cd .. ; tar cvf lulesh-2.0.tar LULESH-2.0 ; mv lulesh-2.0.tar LULESH-2.0

