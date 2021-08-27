/*
 * includes and platform specific defines for setting up the code moved
 * here to clean up the core of the code.
 */
#ifndef __FTQ_H__
#define __FTQ_H__

#include <assert.h>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
/* use cycle timers from FFTW3 (http://www.fftw.org/).  this
 * defines a "ticks" typedef, usually unsigned long long, that is
 * used to store timings.  currently this code will NOT work if the
 * ticks typedef is something other than an unsigned long long.
 */
#include "cycle.h"

#ifdef _WITH_PTHREADS_
#include <pthread.h>
#endif


#endif /* __FTQ_H__ */
