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
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "headers.h"
#include <assert.h>

#include "../gpu/gpu_utilities.h"
#include "../gpu/gpu_vector.h"
#include <cuda.h>
#include <stdio.h>
enum memoryType {memoryTypeHost = 0,
                 memoryTypeDevice = 1,
                 memoryTypeHostPinned = 2,
                 memoryTypeManaged = 3};
typedef enum memoryType memoryType_t;
inline memoryType_t queryPointer(const void *ptr)
{
  CUpointer_attribute attr[] = {CU_POINTER_ATTRIBUTE_CONTEXT,
                                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                CU_POINTER_ATTRIBUTE_IS_MANAGED};
  CUcontext context = NULL;
  CUmemorytype mem_type;
  int is_managed = 0;
  void* data[] = {&context, &mem_type, &is_managed};

  CUresult err = cuPointerGetAttributes(3, attr, data, (CUdeviceptr)ptr);
  if (err != CUDA_SUCCESS) {
    printf("queryPointer: error %d\n", err);
  }

  memoryType_t type;

  if (context == NULL) {
    type = memoryTypeHost;
  } else {
    if (mem_type == CU_MEMORYTYPE_DEVICE) {
      if (is_managed)
        type = memoryTypeManaged;
      else
        type = memoryTypeDevice;
    } else {
      type = memoryTypeHostPinned;
    }
  }

  return type;
}


/*--------------------------------------------------------------------------
 * hypre_VectorSyncHost
 * Copy from device -> host
 *--------------------------------------------------------------------------*/
void hypre_VectorSyncHost( hypre_Vector *vector)
{
  if (GPU_SOLVE_PHASE && vector->size > 0) {
    // device data not initialized -- failure
    if (!vector->initialized) {
      printf("[E]: Trying to sync non-existent device data!\n");
      exit(0);
    }

    // copy device -> host
    if (vector->d_data == NULL) {
      printf("[E]: Trying to sync non-existent device data (%d)!\n",vector->initialized);
      exit(0);
    }
    device_memcpy(vector->data, vector->d_data, sizeof(double)*vector->size, D2H);
    device_checkErrors();
  }
}

/*--------------------------------------------------------------------------
 * hypre_VectorSyncHost
 * Copy from host -> device
 *--------------------------------------------------------------------------*/
void hypre_VectorSyncDevice( hypre_Vector *vector)
{
  if (GPU_SOLVE_PHASE && vector->size > 0) {
    // device data not initialized -- allocate
    if (!vector->initialized) {
      // device data pointer should be NULL in this case
      if (vector->d_data != NULL) {
        printf("[E]: Vector uninitialized but non-NULL device pointer detected\n");
        exit(0);
      }

      device_malloc((void **)&vector->d_data, sizeof(double)*vector->size);
      if (vector->d_data != NULL) vector->initialized = 1;
    }

    device_memcpyAsync(vector->d_data, vector->data, sizeof(double)*vector->size, H2D);
  }
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCreate( int size )
{
   hypre_Vector  *vector;

   vector = hypre_CTAlloc(hypre_Vector, 1);

   hypre_VectorData(vector) = NULL;
   hypre_VectorSize(vector) = size;

   hypre_VectorNumVectors(vector) = 1;
   hypre_VectorMultiVecStorageMethod(vector) = 0;

   /* set defaults */
   hypre_VectorOwnsData(vector) = 1;
   vector->initialized = 0;
   vector->d_data = NULL;

/*
   if (GPU_SOLVE_PHASE && size > 0) {
     device_malloc((void **)&vector->d_data, size*sizeof(double));
     vector->initialized = 1;
   }
*/

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultiVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqMultiVectorCreate( int size, int num_vectors )
{
   hypre_Vector *vector = hypre_SeqVectorCreate(size);
   hypre_VectorNumVectors(vector) = num_vectors;
   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_SeqVectorDestroy( hypre_Vector *vector )
{
  int  ierr=0;

  if (vector)
  {
    if ( hypre_VectorOwnsData(vector) )
    {
      // unregister pinned memory
      // if (GPU_SOLVE_PHASE && (hypre_VectorSize(vector) > 0)) {
      //   memoryType_t mem_type = queryPointer(hypre_VectorData(vector));
      //   if (mem_type == memoryTypeHostPinned) {
      //     device_hostUnregister(hypre_VectorData(vector));
      //   }
      // }

      hypre_TFree(hypre_VectorData(vector));

      // free device memory
      if (vector->initialized && GPU_SOLVE_PHASE) {
        device_free(vector->d_data);
      }
      vector->initialized = 0;
    }
    hypre_TFree(vector);
  }

  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_SeqVectorInitialize( hypre_Vector *vector )
{
  int  size = hypre_VectorSize(vector);
  int  ierr = 0;
  int  num_vectors = hypre_VectorNumVectors(vector);
  int  multivec_storage_method = hypre_VectorMultiVecStorageMethod(vector);

  vector->initialized = 0;
  vector->d_data = NULL;

  if ( ! hypre_VectorData(vector) )
  {
    // hypre_VectorData(vector) = hypre_CTAlloc(double, num_vectors*size);
    hypre_VectorData(vector) = hypre_TAlloc(double, num_vectors*size);
    memset(hypre_VectorData(vector), sizeof(double)*num_vectors*size, 0);

    // register pinned memory

    if (GPU_SOLVE_PHASE && vector->size > 0) {
      // device_hostRegister(hypre_VectorData(vector), sizeof(double)*num_vectors*size, 0);   
      // should always be NULL, but worth checking
      if (vector->d_data == NULL ) {
        device_malloc((void **)&vector->d_data, sizeof(double)*num_vectors*size);
        if (vector->d_data != NULL) vector->initialized = 1;
      }
    }
  }

  if ( multivec_storage_method == 0 )
  {
    hypre_VectorVectorStride(vector) = size;
    hypre_VectorIndexStride(vector) = 1;
  }
  else if ( multivec_storage_method == 1 )
  {
    hypre_VectorVectorStride(vector) = 1;
    hypre_VectorIndexStride(vector) = num_vectors;
  }
  else
    ++ierr;


  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_SeqVectorSetDataOwner( hypre_Vector *vector,
                          int           owns_data   )
{
   int    ierr=0;

   hypre_VectorOwnsData(vector) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * ReadVector
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorRead( char *file_name )
{
   hypre_Vector  *vector;

   FILE    *fp;

   double  *data;
   int      size;
   
   int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &size);

   vector = hypre_SeqVectorCreate(size);
   hypre_SeqVectorInitialize(vector);

   data = hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   /* multivector code not written yet >>> */
   hypre_assert( hypre_VectorNumVectors(vector) == 1 );

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrint
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorPrint( hypre_Vector *vector,
                   char         *file_name )
{
   FILE    *fp;

   double  *data;
   int      size, num_vectors, vecstride, idxstride;
   
   int      i, j;

   int      ierr = 0;

   num_vectors = hypre_VectorNumVectors(vector);
   vecstride = hypre_VectorVectorStride(vector);
   idxstride = hypre_VectorIndexStride(vector);

   // make sure the data is synced if appropriate
   if (GPU_SOLVE_PHASE) {
     if (vector->initialized) hypre_VectorSyncHost(vector);
   }

   /*----------------------------------------------------------
    * Print in the data
    *----------------------------------------------------------*/

   data = hypre_VectorData(vector);
   size = hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   if ( hypre_VectorNumVectors(vector) == 1 )
   {
      fprintf(fp, "%d\n", size);
   }
   else
   {
      fprintf(fp, "%d vectors of size %d\n", num_vectors, size );
   }

   if ( num_vectors>1 )
   {
      for ( j=0; j<num_vectors; ++j )
      {
         fprintf(fp, "vector %d\n", j );
         for (i = 0; i < size; i++)
         {
            fprintf(fp, "%.14e\n",  data[ j*vecstride + i*idxstride ] );
         }
      }
   }
   else
   {
      for (i = 0; i < size; i++)
      {
         fprintf(fp, "%.14e\n", data[i]);
      }
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorSetConstantValues( hypre_Vector *v,
                               double        value )
{
   nvtxRangePush(__FUNCTION__);

   double  *vector_data = hypre_VectorData(v);
   int      size        = hypre_VectorSize(v);
           
   int      i;
           
   int      ierr  = 0;

   size *=hypre_VectorNumVectors(v);

   if (GPU_SOLVE_PHASE && v->size > 0) {
     if (!v->initialized) {
       device_malloc((void**)&v->d_data, sizeof(double)*size);
       device_memcpy(v->d_data, vector_data, sizeof(double)*size, H2D);
       if (v->d_data != NULL) v->initialized = 1;
     }

     if (v->initialized) {
       device_SeqVectorSetConstantValues(value, v->d_data, size);
       //hypre_VectorSyncHost(v);

       nvtxRangePop();
       return ierr;
     }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
   for (i = 0; i < size; i++)
      vector_data[i] = value;

   nvtxRangePop();
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetRandomValues
 *
 *     returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorSetRandomValues( hypre_Vector *v,
                             int           seed )
{
   double  *vector_data = hypre_VectorData(v);
   int      size        = hypre_VectorSize(v);
           
   int      i;
           
   int      ierr  = 0;
   hypre_SeedRand(seed);

   size *=hypre_VectorNumVectors(v);

/* RDF: threading this loop may cause problems because of hypre_Rand() */
   for (i = 0; i < size; i++)
      vector_data[i] = 2.0 * hypre_Rand() - 1.0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopy
 * copies data from x to y
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorCopy( hypre_Vector *x,
                  hypre_Vector *y )
{
   nvtxRangePush(__FUNCTION__);

   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;
           
   int      ierr = 0;

   size *=hypre_VectorNumVectors(x);

   if (GPU_SOLVE_PHASE && x->size > 0) {
     if (!x->initialized) {
       device_malloc((void**)&x->d_data, sizeof(double)*size);
       device_memcpy(x->d_data, x_data, sizeof(double)*size, H2D);
       if (x->d_data != NULL) x->initialized = 1;
     }
     if (!y->initialized) {
       device_malloc((void**)&y->d_data, sizeof(double)*size);
       device_memcpy(y->d_data, y_data, sizeof(double)*size, H2D);
       if (y->d_data != NULL) y->initialized = 1;
     }

     if (x->initialized && y->initialized) {
       //hypre_VectorSyncDevice(x);

       device_SeqVectorCopy(x->d_data, y->d_data, size);

       nvtxRangePop();
       return ierr;

       //hypre_VectorSyncHost(y);
     }
   }


#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
   for (i = 0; i < size; i++)
      y_data[i] = x_data[i];

   nvtxRangePop();
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCloneDeep( hypre_Vector *x )
{
   int      size   = hypre_VectorSize(x);
   int      num_vectors   = hypre_VectorNumVectors(x);
   hypre_Vector * y = hypre_SeqMultiVectorCreate( size, num_vectors );

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

   hypre_SeqVectorInitialize(y);
   hypre_SeqVectorCopy( x, y );

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneShallow
 * Returns a complete copy of x - a shallow copy, pointing the data of x
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCloneShallow( hypre_Vector *x )
{
   int      size   = hypre_VectorSize(x);
   int      num_vectors   = hypre_VectorNumVectors(x);
   hypre_Vector * y = hypre_SeqMultiVectorCreate( size, num_vectors );

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

   hypre_VectorData(y) = hypre_VectorData(x);
   hypre_SeqVectorSetDataOwner( y, 0 );
   hypre_SeqVectorInitialize(y);

   // this should be fine...
   y->d_data = x->d_data;
   y->initialized = x->initialized;

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorScale( double        alpha,
                   hypre_Vector *y     )
{
   nvtxRangePush(__FUNCTION__);

   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(y);
           
   int      i;
           
   int      ierr = 0;

   size *=hypre_VectorNumVectors(y);

   if (GPU_SOLVE_PHASE && y->size > 0) {
     if (!y->initialized) {
       device_malloc((void**)&y->d_data, sizeof(double)*size);
       device_memcpy(y->d_data, y_data, sizeof(double)*size, H2D);
       if (y->d_data != NULL) y->initialized = 1;
     }

     if (y->initialized) {
       //hypre_VectorSyncDevice(y);
       device_SeqVectorScale(alpha, y->d_data, size);
       //hypre_VectorSyncHost(y);

       nvtxRangePop();
       return ierr;
     }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
   for (i = 0; i < size; i++)
      y_data[i] *= alpha;

   nvtxRangePop();
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

int
hypre_SeqVectorAxpy( double        alpha,
            hypre_Vector *x,
            hypre_Vector *y     )
{
   nvtxRangePush(__FUNCTION__);

   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;
           
   int      ierr = 0;

   size *=hypre_VectorNumVectors(x);

   if (GPU_SOLVE_PHASE && x->size > 0) {
     if (!x->initialized) {
       device_malloc((void**)&x->d_data, sizeof(double)*size);
       device_memcpy(x->d_data, x_data, sizeof(double)*size, H2D);
       if (x->d_data != NULL) x->initialized = 1;
     }
     if (!y->initialized) {
       device_malloc((void**)&y->d_data, sizeof(double)*size);
       device_memcpy(y->d_data, y_data, sizeof(double)*size, H2D);
       if (y->d_data != NULL) y->initialized = 1;
     }

     if (x->initialized && y->initialized) {
       //hypre_VectorSyncDevice(x);
       //hypre_VectorSyncDevice(y);
       device_SeqVectorAxpy(alpha,x->d_data,y->d_data,size);

       //hypre_VectorSyncHost(y);
       nvtxRangePop();
       return ierr;
     }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
   for (i = 0; i < size; i++)
      y_data[i] += alpha * x_data[i];

   nvtxRangePop();
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_SeqVectorInnerProd( hypre_Vector *x,
                          hypre_Vector *y )
{
   nvtxRangePush(__FUNCTION__);

   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   int      size   = hypre_VectorSize(x);
           
   int      i;

   double      result = 0.0;

   size *=hypre_VectorNumVectors(x);

   if (GPU_SOLVE_PHASE && x->size > 0) {
     if (!x->initialized) {
       device_malloc((void**)&x->d_data, sizeof(double)*size);
       device_memcpy(x->d_data, x_data, sizeof(double)*size, H2D);
       if (x->d_data != NULL) x->initialized = 1;
     }
     if (!y->initialized) {
       device_malloc((void**)&y->d_data, sizeof(double)*size);
       device_memcpy(y->d_data, y_data, sizeof(double)*size, H2D);
       if (y->d_data != NULL) y->initialized = 1;
     }

     double dev_result = 0.;
     if (x->initialized && y->initialized) {
       //hypre_VectorSyncDevice(x);
       // if (x != y) hypre_VectorSyncDevice(y);
       dev_result = device_SeqVectorInnerProd(x->d_data,y->d_data,size);

       nvtxRangePop();
       return dev_result;
     }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:result) schedule(static)
#endif
   for (i = 0; i < size; i++)
      result += y_data[i] * x_data[i];

   nvtxRangePop();
   return result;
}

/*--------------------------------------------------------------------------
 * hypre_VectorSumElts:
 * Returns the sum of all vector elements.
 *--------------------------------------------------------------------------*/

double
hypre_VectorSumElts( hypre_Vector *vector )
{
   double sum = 0;
   double * data = hypre_VectorData( vector );
   int size = hypre_VectorSize( vector );
   double * d_data = hypre_DVectorData( vector );
   int i;

   if (GPU_SOLVE_PHASE && vector->size > 0) {
     if (!vector->initialized) {
       device_malloc((void **)&vector->d_data, sizeof(double)*size);
       if (vector->d_data != NULL) vector->initialized = 1;
     }

     if (vector->initialized) {
       //hypre_VectorSyncDevice(vector);
       sum = device_VectorSumElts(vector->d_data, size);
       // compute sum
       return sum;
     }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:sum) schedule(static)
#endif
   for ( i=0; i<size; ++i ) sum += data[i];

   return sum;
}


