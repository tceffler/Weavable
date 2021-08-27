# basic makefile - yes, this makefile is not written well.  it was 
# a quick hack.  feel free to fix it and contribute it back if it 
# offends you - hacks like this don't bother me.

#CC = mpixlc_r -qasm=gcc
#CC = gcc
#--> flags for BGP
#CFLAGS =-qasm=gcc  -I../common -O0 -qunroll -DBGP

CC = xlc
CC_THREADED = xlc_r
CFLAGS = -Ofast -qunroll -qasm=gcc -I../common -D__USE_GNU -D_GNU_SOURCE 

all: ftq fwq t_ftq t_fwq

single: ftq fwq

threaded: t_ftq t_fwq

# Fixed TIME quanta benchmark without threads
ftq: ftq.h ftq.c
	$(CC) $(CFLAGS) ftq.c -o ftq

# Fixed TIME quanta benchmark for use with mutiple threads
t_ftq: ftq.h ftq.c
	$(CC_THREADED) $(CFLAGS) -D_WITH_PTHREADS_ -DCORE63 ftq.c -o t_ftq -lpthread

# Fixed WORK quanta benchmark without threads
fwq: ftq.h fwq.c
	$(CC) $(CFLAGS) fwq.c -o fwq

# Fixed WORK quanta benchmark without threads assembly language
# output. This is most useful to view and verify the loop you think
# you are running is the loop the cores/threads are actually
# executing.
fwq.s: ftq.h fwq.c
	$(CC) $(CFLAGS) -S fwq.c

# Fixed WORK quanta benchmark for use with mutiple threads
t_fwq: ftq.h fwq.c
	$(CC_THREADED) $(CFLAGS) -D_WITH_PTHREADS_ fwq.c -o t_fwq -lpthread

clean:
	rm -f ftq.o ftq t_ftq fwq fwq.s t_fwq
