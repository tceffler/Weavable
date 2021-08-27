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
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"
#include <assert.h>

#include "../gpu/gpu_utilities.h"
#include "../gpu/gpu_matrix.h"

extern MPI_Comm HP_MPI_COMM;


/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixMatvec( double           alpha,
              hypre_CSRMatrix *A,
              hypre_Vector    *x,
              double           beta,
              hypre_Vector    *y     )
{
  double     *A_data   = hypre_CSRMatrixData(A);
  int        *A_i      = hypre_CSRMatrixI(A);
  int        *A_j      = hypre_CSRMatrixJ(A);
  int         num_rows = hypre_CSRMatrixNumRows(A);
  int         num_cols = hypre_CSRMatrixNumCols(A);

  int my_id;
  MPI_Comm_rank(HP_MPI_COMM, &my_id);  
  // // // printf("P(%d): \n", myid);
  // if (myid == 7) {
  //   printf("rows = %d, nnz = %d, nnz/rows = %d\n",
  //          num_rows, A->num_nonzeros, A->num_nonzeros/num_rows);
  // }

  int        *A_rownnz = hypre_CSRMatrixRownnz(A);
  int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

  double     *x_data = hypre_VectorData(x);
  double     *y_data = hypre_VectorData(y);

  if (GPU_SOLVE_PHASE) { 
    if (!x->initialized) hypre_VectorSyncDevice(x);
    if (!y->initialized) hypre_VectorSyncDevice(y);
    x_data = hypre_DVectorData(x);
    y_data = hypre_DVectorData(y);
  }

  int         x_size = hypre_VectorSize(x);
  int         y_size = hypre_VectorSize(y);
  int         num_vectors = hypre_VectorNumVectors(x);
  int         idxstride_y = hypre_VectorIndexStride(y);
  int         vecstride_y = hypre_VectorVectorStride(y);
  int         idxstride_x = hypre_VectorIndexStride(x);
  int         vecstride_x = hypre_VectorVectorStride(x);

  double      temp, tempx;

  int         i, j, jj;

  int         m;

  double     xpar=0.7;

  int         ierr = 0;


  /*---------------------------------------------------------------------
   *  Check for size compatibility.  Matvec returns ierr = 1 if
   *  length of X doesn't equal the number of columns of A,
   *  ierr = 2 if the length of Y doesn't equal the number of rows
   *  of A, and ierr = 3 if both are true.
   *
   *  Because temporary vectors are often used in Matvec, none of 
   *  these conditions terminates processing, and the ierr flag
   *  is informational only.
   *--------------------------------------------------------------------*/

  hypre_assert( num_vectors == hypre_VectorNumVectors(y) );

  if (num_cols != x_size)
    ierr = 1;

  if (num_rows != y_size)
    ierr = 2;

  if (num_cols != x_size && num_rows != y_size)
    ierr = 3;

  if (GPU_SOLVE_PHASE) {
    // run GPU SpMV
    // if (my_id == 7) {
    //   double *h_x = (double*)malloc(A->num_cols*sizeof(double));
    //   double *h_y = (double*)malloc(A->num_rows*sizeof(double));
    //   device_memcpy(h_x, x, A->num_cols*sizeof(double), D2H);
    //   device_memcpy(h_y, y, A->num_rows*sizeof(double), D2H);
    //   printf("num_rows = %d, num_cols = %d, nnz = %d, alpha = %f, beta = %f\n", 
    //          A->num_rows, A->num_cols, A->num_nonzeros, alpha, beta);
    //   FILE *fp = fopen("level0_r7.mtx", "wb");
    //   fwrite(&(A->num_rows), sizeof(int), 1, fp);
    //   fwrite(&(A->num_cols), sizeof(int), 1, fp);
    //   fwrite(&(A->num_nonzeros), sizeof(int), 1, fp);
    //   fwrite(&alpha, sizeof(double), 1, fp);
    //   fwrite(&beta, sizeof(double), 1, fp);
    //   fwrite(A->i, sizeof(int), A->num_rows+1, fp);
    //   fwrite(A->j, sizeof(int), A->num_nonzeros, fp);
    //   fwrite(A->data, sizeof(double), A->num_nonzeros, fp);
    //   fwrite(h_x, sizeof(double), A->num_cols, fp);
    //   fwrite(h_y, sizeof(double), A->num_rows, fp);
    //   fclose(fp);
    //   free(h_x);
    //   free(h_y);
    // }
    device_spmv(A, num_rows, num_cols, hypre_CSRMatrixNumNonzeros(A), alpha, A_i, A_j, A_data, x_data, x_size, beta, y_data, y_size, 0);
  } else {

    /*-----------------------------------------------------------------------
     * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
     *-----------------------------------------------------------------------*/

    if (alpha == 0.0)
    {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static) 
#endif
      for (i = 0; i < num_rows*num_vectors; i++)
        y_data[i] *= beta;

      return ierr;
    }

    /*-----------------------------------------------------------------------
     * y = (beta/alpha)*y
     *-----------------------------------------------------------------------*/

    temp = beta / alpha;

    if (temp != 1.0)
    {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static) 
#endif
        for (i = 0; i < num_rows*num_vectors; i++)
          y_data[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static) 
#endif
        for (i = 0; i < num_rows*num_vectors; i++)
          y_data[i] *= temp;
      }
    }

    /*-----------------------------------------------------------------
     * y += A*x
     *-----------------------------------------------------------------*/

    if (num_rownnz < xpar*(num_rows))
    {
      /* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,j,m,tempx) schedule(static)
#endif
      for (i = 0; i < num_rownnz; i++)
      {
        m = A_rownnz[i];

        /*
         * for (jj = A_i[m]; jj < A_i[m+1]; jj++)
         * {
         *         j = A_j[jj];   
         *  y_data[m] += A_data[jj] * x_data[j];
         * } */
        if ( num_vectors==1 )
        {
          tempx = y_data[m];
          for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
            tempx +=  A_data[jj] * x_data[A_j[jj]];
          y_data[m] = tempx;
        }
        else
          for ( j=0; j<num_vectors; ++j )
          {
            tempx = y_data[ j*vecstride_y + m*idxstride_y ];
            for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
              tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
            y_data[ j*vecstride_y + m*idxstride_y] = tempx;
          }
      }

    }
    else
    {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,temp,j) schedule(static)
#endif
      for (i = 0; i < num_rows; i++)
      {
        if ( num_vectors==1 )
        {
          temp = y_data[i];
          for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            temp += A_data[jj] * x_data[A_j[jj]];
          y_data[i] = temp;
        }
        else
          for ( j=0; j<num_vectors; ++j )
          {
            temp = y_data[ j*vecstride_y + i*idxstride_y ];
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
              temp += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
            }
            y_data[ j*vecstride_y + i*idxstride_y ] = temp;
          }
      }
    }

    /*-----------------------------------------------------------------
     * y = alpha*y
     *-----------------------------------------------------------------*/

    if (alpha != 1.0)
    {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
      for (i = 0; i < num_rows*num_vectors; i++)
        y_data[i] *= alpha;
    }
  }
  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *  This version is using a different (more efficient) threading scheme

 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixMatvecT( double           alpha,
               hypre_CSRMatrix *A,
               hypre_Vector    *x,
               double           beta,
               hypre_Vector    *y     )
{
   double     *A_data    = hypre_CSRMatrixData(A);
   int        *A_i       = hypre_CSRMatrixI(A);
   int        *A_j       = hypre_CSRMatrixJ(A);
   int         num_rows  = hypre_CSRMatrixNumRows(A);
   int         num_cols  = hypre_CSRMatrixNumCols(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);

  if (GPU_SOLVE_PHASE) {
    x_data = hypre_DVectorData(x);
    y_data = hypre_DVectorData(y);
  }

   int         x_size = hypre_VectorSize(x);
   int         y_size = hypre_VectorSize(y);
   int         num_vectors = hypre_VectorNumVectors(x);
   int         idxstride_y = hypre_VectorIndexStride(y);
   int         vecstride_y = hypre_VectorVectorStride(y);
   int         idxstride_x = hypre_VectorIndexStride(x);
   int         vecstride_x = hypre_VectorVectorStride(x);

   double      temp;

   double      *y_data_expand = NULL;
   int         offset = 0;
#ifdef HYPRE_USING_OPENMP
   int         my_thread_num = 0;
#endif
   
   int         i, j, jv, jj;
   int         num_threads;

   int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

    hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;


  if (GPU_SOLVE_PHASE) {
    // run GPU SpMV
    device_spmv(A, num_rows, num_cols, hypre_CSRMatrixNumNonzeros(A), alpha, A_i, A_j, A_data, x_data, x_size, beta, y_data, y_size, 1);
  } else {
    /*-----------------------------------------------------------------------
     * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
     *-----------------------------------------------------------------------*/

    if (alpha == 0.0)
    {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
      for (i = 0; i < num_cols*num_vectors; i++)
        y_data[i] *= beta;

      return ierr;
    }

    /*-----------------------------------------------------------------------
     * y = (beta/alpha)*y
     *-----------------------------------------------------------------------*/

    temp = beta / alpha;

    if (temp != 1.0)
    {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
        for (i = 0; i < num_cols*num_vectors; i++)
          y_data[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
        for (i = 0; i < num_cols*num_vectors; i++)
          y_data[i] *= temp;
      }
    }

    /*-----------------------------------------------------------------
     * y += A^T*x
     *-----------------------------------------------------------------*/

    //printf("Hypre transpose spmv: %d x %d, x: %d, y: %d, alpha: %lg, beta: %lg\n",num_rows,num_cols,x_size,y_size,alpha,beta);
    num_threads = hypre_NumThreads();
    if (num_threads > 1)
    {
      y_data_expand = hypre_CTAlloc(double, num_threads*y_size);

      if ( num_vectors==1 )
      {

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,jj,j, my_thread_num, offset)    
        {                                      
          my_thread_num = omp_get_thread_num();
          offset =  y_size*my_thread_num;
#pragma omp for schedule(static)
#endif
          for (i = 0; i < num_rows; i++)
          {
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
              j = A_j[jj];
              y_data_expand[offset + j] += A_data[jj] * x_data[i];
            }
          }
#ifdef HYPRE_USING_OPENMP
          /* implied barrier */           
#pragma omp for schedule(static)
#endif
          for (i = 0; i < y_size; i++)
          {
            for (j = 0; j < num_threads; j++)
            {
              y_data[i] += y_data_expand[j*y_size + i];
              /*y_data_expand[j*y_size + i] = 0; //zero out for next time */
            }
          }
#ifdef HYPRE_USING_OPENMP
        } /* end parallel region */
#endif         
        hypre_TFree(y_data_expand);
      }
      else
      {
        /* MULTIPLE VECTORS NOT THREADED YET */
        for (i = 0; i < num_rows; i++)
        {
          for ( jv=0; jv<num_vectors; ++jv )
          {
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
              j = A_j[jj];
              y_data[ j*idxstride_y + jv*vecstride_y ] +=
                A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x];
            }
          }
        }
      }

      hypre_TFree(y_data_expand);
    }
    else 
    {
      for (i = 0; i < num_rows; i++)
      {
        if ( num_vectors==1 )
        {
          for (jj = A_i[i]; jj < A_i[i+1]; jj++)
          {
            j = A_j[jj];
            y_data[j] += A_data[jj] * x_data[i];
          }
        }
        else
        {
          for ( jv=0; jv<num_vectors; ++jv )
          {
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
              j = A_j[jj];
              y_data[ j*idxstride_y + jv*vecstride_y ] +=
                A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
            }
          }
        }
      }
    }
    if (alpha != 1.0)
    {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
      for (i = 0; i < num_cols*num_vectors; i++)
        y_data[i] *= alpha;
    }
  }
   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/
                                                                                                              
int
hypre_CSRMatrixMatvec_FF( double           alpha,
              hypre_CSRMatrix *A,
              hypre_Vector    *x,
              double           beta,
              hypre_Vector    *y,
              int             *CF_marker_x,
              int             *CF_marker_y,
              int fpt )
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         num_rows = hypre_CSRMatrixNumRows(A);
   int         num_cols = hypre_CSRMatrixNumCols(A);
                                                                                                              
   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   int         x_size = hypre_VectorSize(x);
   int         y_size = hypre_VectorSize(y);
                                                                                                              
   double      temp;
                                                                                                              
   int         i, jj;
                                                                                                              
   int         ierr = 0;
                                                                                                              
                                                                                                              
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
                                                                                                              
    if (num_cols != x_size)
              ierr = 1;
                                                                                                              
    if (num_rows != y_size)
              ierr = 2;
                                                                                                              
    if (num_cols != x_size && num_rows != y_size)
              ierr = 3;
                                                                                                              
   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/
                                                                                                              
    if (alpha == 0.0)
    {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
       for (i = 0; i < num_rows; i++)
          if (CF_marker_x[i] == fpt) y_data[i] *= beta;
                                                                                                              
       return ierr;
    }
                                                                                                              
   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/
                                                                                                              
   temp = beta / alpha;
                                                                                                              
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] *= temp;
      }
   }
                                                                                                              
   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/
                                                                                                              
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,temp) schedule(static)
#endif 
   for (i = 0; i < num_rows; i++)
   {
      if (CF_marker_x[i] == fpt)
      {
         temp = y_data[i];
         for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            if (CF_marker_y[A_j[jj]] == fpt) temp += A_data[jj] * x_data[A_j[jj]];
         y_data[i] = temp;
      }
   }
                                                                                                              
                                                                                                              
   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/
                                                                                                              
   if (alpha != 1.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
      for (i = 0; i < num_rows; i++)
         if (CF_marker_x[i] == fpt) y_data[i] *= alpha;
   }
                                                                                                              
   return ierr;
}

