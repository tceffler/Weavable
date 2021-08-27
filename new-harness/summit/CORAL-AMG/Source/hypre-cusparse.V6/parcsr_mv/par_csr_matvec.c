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

extern MPI_Comm HP_MPI_COMM;


/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMatvec( double           alpha,
              	 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *x,
                 double           beta,
                 hypre_ParVector    *y     )
{
   nvtxRangePushColor(__FUNCTION__, 0xFFFFFF00);

   hypre_ParCSRCommHandle	**comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix      *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Vector         *x_local  = hypre_ParVectorLocalVector(x);   
   hypre_Vector         *y_local  = hypre_ParVectorLocalVector(y);   
   HYPRE_BigInt         num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt         num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   static hypre_Vector      *x_tmp;
   HYPRE_BigInt        x_size = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt        y_size = hypre_ParVectorGlobalSize(y);
   int        num_vectors = 1;
   int	      num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int        ierr = 0;
   int	      num_sends, i, j, jv, index, start;

   /*int        vecstride = hypre_VectorVectorStride( x_local );
   int        idxstride = hypre_VectorIndexStride( x_local );*/

   double     *x_tmp_data, **x_buf_data;
   double     *x_local_data = hypre_VectorData(x_local);
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
   /*hypre_assert( idxstride>0 );*/

    if (num_cols != x_size)
              ierr = 11;

    if (num_rows != y_size)
              ierr = 12;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 13;

#if 0
      x_tmp = hypre_SeqVectorCreate( num_cols_offd );
      hypre_SeqVectorInitialize(x_tmp);
#else
//NEW :
    static int x_tmp_FLAG = 0;
    static int x_tmp_capacity = -1;

    if ( num_cols_offd > x_tmp_capacity) {
      if (x_tmp_FLAG > 0){
       hypre_SeqVectorDestroy(x_tmp);
        x_tmp = NULL;
      }
      x_tmp = hypre_SeqVectorCreate( num_cols_offd );
      hypre_SeqVectorInitialize(x_tmp);
      x_tmp_capacity = num_cols_offd;
      x_tmp_FLAG = 1;
    }
    if (GPU_SOLVE_PHASE && !x_tmp->initialized) {
      device_malloc((void **)&x_tmp->d_data, x_tmp_capacity*sizeof(double));
      if (x_tmp->d_data != NULL) x_tmp->initialized = 1;
    }
    hypre_VectorSize(x_tmp) = num_cols_offd;
//END of NEW
#endif
    x_tmp_data = hypre_VectorData(x_tmp);
    
    nvtxRangePushA("comm_handle alloc");
    comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*,num_vectors);
    nvtxRangePop();

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(A);
#else
      hypre_MatvecCommPkgCreate(A);
#endif
      comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   x_buf_data = /*hypre_CTAlloc*/ hypre_TAlloc( double*, num_vectors );
   for ( jv=0; jv<num_vectors; ++jv )
      x_buf_data[jv] = /*hypre_CTAlloc*/ hypre_TAlloc (double, hypre_ParCSRCommPkgSendMapStart
                                    (comm_pkg, num_sends));

   // initialize vectors if necessary
   if (GPU_SOLVE_PHASE) {
     if (!x_local->initialized) hypre_VectorSyncDevice(x_local);
     if (!y_local->initialized) hypre_VectorSyncDevice(y_local); 
   }

   // assemble send buffer & launch diag spmv
   if (GPU_SOLVE_PHASE) {
     int send_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
     int *send_maps = NULL;

     if (!device_has_send_maps(diag, 0)) {
       // fill in host send maps
       send_maps = hypre_CTAlloc(int, send_size);
       index = 0;
       for (i = 0; i < num_sends; i++)
       {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
           send_maps[index++]
             = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         #if 0
         printf("MPI:   num_sends=%d, mes_size[%d] = %d\n",num_sends,i,hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start);
         #endif
       }
     }

     // fill in comm buffers on gpu
     device_set_stream(1);
     device_create_comm_buffer(diag, send_size, send_maps, x_buf_data[0], x_local->d_data);
   
     // launch diag spmv to occupy gpu
     device_set_stream(2);
     hypre_CSRMatrixMatvec(alpha, diag, x_local, beta, y_local);

     // make sure we copied comm buffer
     device_set_stream(1);
     device_sync_stream();

     if (send_maps != NULL) {
       // remove temp host maps
       hypre_TFree(send_maps);
     }
   }
   else {
   /*if ( num_vectors==1 )*/
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            x_buf_data[0][index++] 
               = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         #if 0
         printf("MPI:   num_sends=%d, mes_size[%d] = %d\n",num_sends,i,hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start); 
         #endif
      }
   }
   /*else
      for ( jv=0; jv<num_vectors; ++jv )
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               x_buf_data[jv][index++] 
                  = x_local_data[
                     jv*vecstride +
                     idxstride*hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j) ];
         }
      }

   hypre_assert( idxstride==1 );*/
   /* >>> ... The assert is because the following loop only works for 'column' storage of a multivector <<<
      >>> This needs to be fixed to work more generally, at least for 'row' storage. <<<
      >>> This in turn, means either change CommPkg so num_sends is no.zones*no.vectors (not no.zones)
      >>> or, less dangerously, put a stride in the logic of CommHandleCreate (stride either from a
      >>> new arg or a new variable inside CommPkg).  Or put the num_vector iteration inside
      >>> CommHandleCreate (perhaps a new multivector variant of it).
   */
   }

   for ( jv=0; jv<num_vectors; ++jv )
   {
      comm_handle[jv] = hypre_ParCSRCommHandleCreate
         ( 1, comm_pkg, x_buf_data[jv], &(x_tmp_data[jv*num_cols_offd]) );
   }

   if (!GPU_SOLVE_PHASE) {
     // launch diag spmv on cpu
     hypre_CSRMatrixMatvec(alpha, diag, x_local, beta, y_local);
   }

   // nvtxRangePushColor("MPI_Barrier", 0xFFFF0000);
   // MPI_Barrier(hypre_ParCSRCommPkgComm(comm_pkg));
   // nvtxRangePop();         
   for ( jv=0; jv<num_vectors; ++jv )
   {
      hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
   }
   hypre_TFree(comm_handle);

   // sync ext data on device after MPI transfer
   if (GPU_SOLVE_PHASE) {
     device_set_stream(1);
     hypre_VectorSyncDevice(x_tmp);
     device_sync_stream();

     // launch offd spmv
     device_set_stream(2);
   }
   if (num_cols_offd) hypre_CSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);    

   if (GPU_SOLVE_PHASE) {
     device_sync_stream();

     // set NULL stream for other work
     device_set_stream(0);
   }

#if 0 
   hypre_SeqVectorDestroy(x_tmp);
   x_tmp = NULL;
#endif
   for ( jv=0; jv<num_vectors; ++jv ) hypre_TFree(x_buf_data[jv]);
   hypre_TFree(x_buf_data);
 
   nvtxRangePop(); 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMatvecT( double           alpha,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *x,
                  double           beta,
                  hypre_ParVector    *y     )
{
   nvtxRangePushColor(__FUNCTION__, 0xFFFFA500);

   hypre_ParCSRCommHandle	**comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   static hypre_Vector *y_tmp;
   int           vecstride = hypre_VectorVectorStride( y_local );
   int           idxstride = hypre_VectorIndexStride( y_local );
   double       *y_tmp_data, **y_buf_data;
   double       *y_local_data = hypre_VectorData(y_local);

   HYPRE_BigInt         num_rows  = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt         num_cols  = hypre_ParCSRMatrixGlobalNumCols(A);
   int	       num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_BigInt         x_size = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt         y_size = hypre_ParVectorGlobalSize(y);
   int         num_vectors = hypre_VectorNumVectors(y_local);

   int         i, j, jv, index, start, num_sends;

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
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
   /*-----------------------------------------------------------------------
    *-----------------------------------------------------------------------*/

    comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*,num_vectors);

#if 0
    y_tmp = hypre_SeqVectorCreate(num_cols_offd);
    hypre_SeqVectorInitialize(y_tmp);
    if (num_cols_offd == 0) printf("num_cols_offd=%d\n",num_cols_offd);
#else
    static int y_tmp_FLAG = 0;
    static int y_tmp_capacity = 0;
    
    if ( (num_cols_offd > y_tmp_capacity) || (y_tmp_FLAG == 0) ) {
      if (y_tmp_FLAG > 0){
       hypre_SeqVectorDestroy(y_tmp);
        y_tmp = NULL;
      }
      y_tmp = hypre_SeqVectorCreate( num_cols_offd );
      hypre_SeqVectorInitialize(y_tmp);
      y_tmp_capacity = num_cols_offd;
      y_tmp_FLAG = 1;
    }
    if (GPU_SOLVE_PHASE && !y_tmp->initialized) {
      device_malloc((void **)&y_tmp->d_data, y_tmp_capacity*sizeof(double));
      if (y_tmp->d_data != NULL) y_tmp->initialized = 1;
    }
    hypre_VectorSize(y_tmp) = num_cols_offd;
#endif


   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(A);
#else
      hypre_MatvecCommPkgCreate(A);
#endif
      comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   y_buf_data = hypre_CTAlloc( double*, num_vectors );
   for ( jv=0; jv<num_vectors; ++jv )
      y_buf_data[jv] = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart
                                     (comm_pkg, num_sends));
   y_tmp_data = hypre_VectorData(y_tmp);
   y_local_data = hypre_VectorData(y_local);

   hypre_assert( idxstride==1 ); /* >>> only 'column' storage of multivectors implemented so far */

   // initialize vectors if necessary
   if (GPU_SOLVE_PHASE) {
     if (!x_local->initialized) hypre_VectorSyncDevice(x_local);
     if (!y_local->initialized) hypre_VectorSyncDevice(y_local);
   }

   // matvecT on off-diag
   if (num_cols_offd) hypre_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);

   if (GPU_SOLVE_PHASE) {
     if (num_cols_offd) {
       // copy offd part of the vector to host
       device_set_stream(1);
       if (y_tmp->initialized && y_tmp->d_data != NULL) {
         device_memcpyAsync(y_tmp_data, y_tmp->d_data, sizeof(double)*y_tmp->size, D2H);
       } else {
         printf("[E]: vector not initialized\n");
       }
     }
     // hypre_VectorSyncHost(y_tmp);
     //
     // now call the diag matvecT
     device_set_stream(2);
     hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);

     if (num_cols_offd) {
       // make sure the copy down has completed
       device_set_stream(1);
       device_sync_stream();
     }
   }

   /**
    * async copy of results from offd matvecT while performing diag matvecT
    */
   for ( jv=0; jv<num_vectors; ++jv )
   {
      /* >>> this is where we assume multivectors are 'column' storage */
      comm_handle[jv] = hypre_ParCSRCommHandleCreate
         ( 2, comm_pkg, &(y_tmp_data[jv*num_cols_offd]), y_buf_data[jv] );
   }

   // if in the GPU_SOLV_PHASE, this operation has already been submitted
   if (!GPU_SOLVE_PHASE) {
     hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
   }

   for ( jv=0; jv<num_vectors; ++jv )
   {
      hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
   }
   hypre_TFree(comm_handle);

   if (GPU_SOLVE_PHASE) {
     // ensure diag matvecT has completed
     device_set_stream(2);
     device_sync_stream();

     // set matrix->send_maps = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)
     // set matrix->send_data = y_buf_data
     int myid;
     MPI_Comm_rank(HP_MPI_COMM, &myid);
     int recv_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
     int *recv_maps = NULL;

     if (!device_has_send_maps(diag, 1) && recv_size > 0) {
       // fill in host recv maps
       recv_maps = hypre_CTAlloc(int, recv_size);
       index = 0;
       for (i = 0; i < num_sends; i++)
       {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
           recv_maps[index] = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
           index++;
         }
       }
       // copy this map to the device and associate with a matrix
       device_set_comm_map(diag,recv_size,recv_maps, 1);
       hypre_TFree(recv_maps);
     }
     // at this point, the matrix should have the relevant map set and y_buf_data should have been received
     // we can now assemble the final result
     if (recv_size > 0) {
       device_assemble_transpose_result(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag), recv_size, y_local->d_data, y_buf_data[0]);
     }
     // here, we should have the final result
   }

   // if we haven't done all this on the device, do it now on host
   if (!GPU_SOLVE_PHASE) {
     /**
      * perform Axpy on local data with received data
      */
     if ( num_vectors==1 )
     {
       index = 0;
       for (i = 0; i < num_sends; i++)
       {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
           y_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
             += y_buf_data[0][index++];
       }

     } else {
       for ( jv=0; jv<num_vectors; ++jv )
       {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
           start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
           for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
             y_local_data[ jv*vecstride +
               idxstride*hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j) ]
               += y_buf_data[jv][index++];
         }
       }
     }
   }

//NEW
#if 0	
   hypre_SeqVectorDestroy(y_tmp);
   y_tmp = NULL;
#endif
//END NEW
   for ( jv=0; jv<num_vectors; ++jv ) hypre_TFree(y_buf_data[jv]);
   hypre_TFree(y_buf_data);

   nvtxRangePop();
   return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/
                                                                                                              
int
hypre_ParCSRMatrixMatvec_FF( double           alpha,
                 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *x,
                 double           beta,
                 hypre_ParVector    *y,
                 int                *CF_marker,
                 int fpt )
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommHandle       *comm_handle;
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix      *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Vector         *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector         *y_local  = hypre_ParVectorLocalVector(y);
   HYPRE_BigInt         num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt         num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
                                                                                                              
   hypre_Vector      *x_tmp;
   HYPRE_BigInt        x_size = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt        y_size = hypre_ParVectorGlobalSize(y);
   HYPRE_BigInt        num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int        ierr = 0;
   int        num_sends, i, j, index, start, num_procs;
   int        *int_buf_data = NULL;
   int        *CF_marker_offd = NULL;
                                                                                                              
                                                                                                              
   double     *x_tmp_data = NULL;
   double     *x_buf_data = NULL;
   double     *x_local_data = hypre_VectorData(x_local);
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
                                                                                                              
   MPI_Comm_size(comm,&num_procs);
                                                                                                              
   if (num_cols != x_size)
              ierr = 11;
                                                                                                              
   if (num_rows != y_size)
              ierr = 12;
                                                                                                              
   if (num_cols != x_size && num_rows != y_size)
              ierr = 13;
                                                                                                              
   if (num_procs > 1)
   {
      if (num_cols_offd)
      {
         x_tmp = hypre_SeqVectorCreate( num_cols_offd );
         hypre_SeqVectorInitialize(x_tmp);
         x_tmp_data = hypre_VectorData(x_tmp);
      }
                                                                                                              
   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
      if (!comm_pkg)
      {
#ifdef HYPRE_NO_GLOBAL_PARTITION
         hypre_NewCommPkgCreate(A);
#else
         hypre_MatvecCommPkgCreate(A);
#endif
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
                                                                                                              
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      if (num_sends)
         x_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart
                                    (comm_pkg, num_sends));
                                                                                                              
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            x_buf_data[index++]
               = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
      comm_handle = hypre_ParCSRCommHandleCreate ( 1, comm_pkg, x_buf_data, x_tmp_data );
   }
   hypre_CSRMatrixMatvec_FF( alpha, diag, x_local, beta, y_local, CF_marker, CF_marker, fpt);
                                                                                                              
   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
                                                                                                              
      if (num_sends)
         int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart
                                    (comm_pkg, num_sends));
      if (num_cols_offd) CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            int_buf_data[index++]
               = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11,comm_pkg,int_buf_data,CF_marker_offd );
                                                                                                              
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
                                                                                                              
      if (num_cols_offd) hypre_CSRMatrixMatvec_FF( alpha, offd, x_tmp, 1.0, y_local,
        CF_marker, CF_marker_offd, fpt);
                                                                                                              
      hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      hypre_TFree(x_buf_data);
      hypre_TFree(int_buf_data);
      hypre_TFree(CF_marker_offd);
   }
                                                                                                              
   return ierr;
}
