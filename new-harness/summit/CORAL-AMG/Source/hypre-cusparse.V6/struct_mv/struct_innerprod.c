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


/******************************************************************************
 *
 * Structured inner product routine
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_StructInnerProd
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_USE_PTHREADS
double          *local_result_ref[hypre_MAX_THREADS];
#endif

double           final_innerprod_result;


double
hypre_StructInnerProd(  hypre_StructVector *x,
                        hypre_StructVector *y )
{
   double           local_result;
   double           process_result;
                   
   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;
                   
   int              xi;
   int              yi;
                   
   double          *xp;
   double          *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;
                   
   int              i;
   int              loopi, loopj, loopk;
#ifdef HYPRE_USE_PTHREADS
   int              threadid = hypre_GetThreadID();
#endif

   local_result = 0.0;
   process_result = 0.0;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxGetSize(box, loop_size);

#ifdef HYPRE_USE_PTHREADS
   local_result_ref[threadid] = &local_result;
#endif

         hypre_BoxLoop2Begin(loop_size,
                             x_data_box, start, unit_stride, xi,
                             y_data_box, start, unit_stride, yi);

	 hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
            {
               local_result += xp[xi] * yp[yi];
            }
         hypre_BoxLoop2End(xi, yi);
      }

#ifdef HYPRE_USE_PTHREADS
   if (threadid != hypre_NumThreads)
   {
      for (i = 0; i < hypre_NumThreads; i++)
         process_result += *local_result_ref[i];
   }
   else
      process_result = *local_result_ref[threadid];
#else
   process_result = local_result;
#endif


   MPI_Allreduce(&process_result, &final_innerprod_result, 1,
                 MPI_DOUBLE, MPI_SUM, hypre_StructVectorComm(x));


#ifdef HYPRE_USE_PTHREADS
   if (threadid == 0 || threadid == hypre_NumThreads)
#endif
   hypre_IncFLOPCount(2*hypre_StructVectorGlobalSize(x));

   return final_innerprod_result;
}
