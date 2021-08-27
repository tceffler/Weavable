MPI_HOME = ${MPI_ROOT}

MPI_FLAGS = -DUSE_MPI -I$(MPI_HOME)/include

SHELL = /bin/sh
.SUFFIXES: .cc .o

LULESH_EXEC = lulesh

MPI_INC = ${MPI_ROOT}/include
MPI_LIB = ${MPI_ROOT}/lib

SOURCES = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc
OBJECTS = $(SOURCES:.cc=.o)

CXXFLAGS = $(MPI_FLAGS) -O3 -DNDEBUG -fopenmp -I. -Wall
#CXXFLAGS = $(MPI_FLAGS) -g -fopenmp -I. -Wall
LDFLAGS = -g -O3 -fopenmp


.cc.o: lulesh.h
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(LULESH_EXEC)

lulesh: $(OBJECTS)
	@echo "Linking"
	$(CXX) $(OBJECTS) $(LDFLAGS) -lm -o $@

clean:
	rm -f *.o *.dSYM *~ $(OBJECTS) $(LULESH_EXEC)


