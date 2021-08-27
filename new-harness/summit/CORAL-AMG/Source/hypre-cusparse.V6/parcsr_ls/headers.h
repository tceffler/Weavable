/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "parcsr_ls.h"
/*#include "par_csr_block_matrix.h"*/

#include <sys/time.h>
typedef struct CTimer {
  struct timeval timerStart, timerEnd;
} CTimer;

//static struct timeval timerStart, timerEnd;
static void startTimer(CTimer *timer)
{
  gettimeofday(&(timer->timerStart), NULL);
}
static double getET(CTimer *timer)
{
  gettimeofday(&(timer->timerEnd),NULL);
  double et=(timer->timerEnd.tv_sec+timer->timerEnd.tv_usec*0.000001)-
    (timer->timerStart.tv_sec+timer->timerStart.tv_usec*0.000001);
  return et;
}
