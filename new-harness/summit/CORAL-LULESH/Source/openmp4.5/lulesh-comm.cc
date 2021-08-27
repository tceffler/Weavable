#include "lulesh.h"

// If no MPI, then this whole file is stubbed out
#if USE_MPI

#include <mpi.h>
#include <string.h>

/* Comm Routines */

#define ALLOW_UNPACKED_PLANE false
#define ALLOW_UNPACKED_ROW   false
#define ALLOW_UNPACKED_COL   false


#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif

#if _OPENMP
# include <omp.h>
#endif

/*
   There are coherence issues for packing and unpacking message
   buffers.  Ideally, you would like a lot of threads to 
   cooperate in the assembly/dissassembly of each message.
   To do that, each thread should really be operating in a
   different coherence zone.

   Let's assume we have three fields, f1 through f3, defined on
   a 61x61x61 cube.  If we want to send the block boundary
   information for each field to each neighbor processor across
   each cube face, then we have three cases for the
   memory layout/coherence of data on each of the six cube
   boundaries:

      (a) Two of the faces will be in contiguous memory blocks
      (b) Two of the faces will be comprised of pencils of
          contiguous memory.
      (c) Two of the faces will have large strides between
          every value living on the face.

   How do you pack and unpack this data in buffers to
   simultaneous achieve the best memory efficiency and
   the most thread independence?

   Do do you pack field f1 through f3 tighly to reduce message
   size?  Do you align each field on a cache coherence boundary
   within the message so that threads can pack and unpack each
   field independently?  For case (b), do you align each
   boundary pencil of each field separately?  This increases
   the message size, but could improve cache coherence so
   each pencil could be processed independently by a separate
   thread with no conflicts.

   Also, memory access for case (c) would best be done without
   going through the cache (the stride is so large it just causes
   a lot of useless cache evictions).  Is it worth creating
   a special case version of the packing algorithm that uses
   non-coherent load/store opcodes?
*/

/******************************************/


/* doRecv flag only works with regular block structure */
void CommRecv(Domain& domain, int msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz, bool doRecv, bool planeOnly) {

   #ifdef USE_NVTX
   nvtxRangeId_t nvtx_CommRecv = nvtxRangeStartA("CommRecv");
   #endif

   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;

   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;

   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   for (Index_t i=0; i<26; ++i) {
      domain.recvRequest[i] = MPI_REQUEST_NULL ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   /* post receives */

   /* receive data from neighboring domain faces */
   if (planeMin && doRecv) {
      /* contiguous memory */
      int fromRank = myRank - domain.tp()*domain.tp() ;
      int recvCount = dx * dy * xferFields ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (planeMax) {
      /* contiguous memory */
      int fromRank = myRank + domain.tp()*domain.tp() ;
      int recvCount = dx * dy * xferFields ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (rowMin && doRecv) {
      /* semi-contiguous memory */
      int fromRank = myRank - domain.tp() ;
      int recvCount = dx * dz * xferFields ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (rowMax) {
      /* semi-contiguous memory */
      int fromRank = myRank + domain.tp() ;
      int recvCount = dx * dz * xferFields ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (colMin && doRecv) {
      /* scattered memory */
      int fromRank = myRank - 1 ;
      int recvCount = dy * dz * xferFields ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (colMax) {
      /* scattered memory */
      int fromRank = myRank + 1 ;
      int recvCount = dy * dz * xferFields ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      ++pmsg ;
   }

   if (!planeOnly) {
      /* receive data from domains connected only by an edge */
      if (rowMin && colMin && doRecv) {
         int fromRank = myRank - domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMin && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && colMax) {
         int fromRank = myRank + domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMax && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && colMin) {
         int fromRank = myRank + domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMin && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && colMax && doRecv) {
         int fromRank = myRank - domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMax && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      /* receive data from domains connected only by a corner */
      if (rowMin && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 0, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMin && planeMax) {
         /* corner at domain logical coord (0, 0, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 0, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax) {
         /* corner at domain logical coord (1, 0, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 1, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax) {
         /* corner at domain logical coord (0, 1, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 1, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax) {
         /* corner at domain logical coord (1, 1, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
   }
   #ifdef USE_NVTX
   nvtxRangeEnd(nvtx_CommRecv);
   #endif

}

/******************************************/

void CommSend(Domain& domain, int msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly)
{
   #ifdef USE_NVTX
   nvtxRangeId_t nvtx_CommSend = nvtxRangeStartA("CommSend");
   #endif


   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   MPI_Status status[26] ;
   Real_t *destAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax, not_planeOnly;
   not_planeOnly = !planeOnly;

   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   for (Index_t i=0; i<26; ++i) {
      domain.sendRequest[i] = MPI_REQUEST_NULL ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   
   Real_t *ptr_fi[xferFields];
   for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Domain_member src = fieldData[fi] ;
      ptr_fi[fi] = &(domain.*src)(0);
   }
   #pragma omp target enter data map(to:ptr_fi[0:xferFields]) if(USE_DEVICE)

/*** IBM: preparing for OpenMP parallel communication  ***/
   int pmsg_array[26], emsg_array[26], cmsg_array[26];
   for (Index_t i = 0; i < 26; ++i)  pmsg_array[i]=0;
   for (Index_t i = 0; i < 26; ++i)  emsg_array[i]=0;
   for (Index_t i = 0; i < 26; ++i)  cmsg_array[i]=0;

   if (planeMin | planeMax) {
      if (planeMin)                  {pmsg_array[0] = pmsg++;}
      if (planeMax && doSend)        {pmsg_array[1] = pmsg++;}
   }
   if (rowMin | rowMax) {
      if (rowMin)                    {pmsg_array[2] = pmsg++;}
      if (rowMax && doSend)          {pmsg_array[3] = pmsg++;}
   }
   if (colMin | colMax) {
      if (colMin)                    {pmsg_array[4] = pmsg++;}
      if (colMax && doSend)          {pmsg_array[5] = pmsg++;}
   }
   if (!planeOnly){
     if (rowMin && colMin)                       {emsg_array[6] = emsg++; pmsg_array[6] = pmsg;}
     if (rowMin && planeMin)                     {emsg_array[7] = emsg++; pmsg_array[7] = pmsg;}
     if (colMin && planeMin)                     {emsg_array[8] = emsg++; pmsg_array[8] = pmsg;}
     if (rowMax && colMax && doSend)             {emsg_array[9] = emsg++; pmsg_array[9] = pmsg;}
     if (rowMax && planeMax && doSend)           {emsg_array[10] = emsg++; pmsg_array[10] = pmsg;}
     if (colMax && planeMax  && doSend)          {emsg_array[11] = emsg++; pmsg_array[11] = pmsg;}
     if (rowMax && colMin && doSend)             {emsg_array[12] = emsg++; pmsg_array[12] = pmsg;}
     if (rowMin && planeMax && doSend)           {emsg_array[13] = emsg++; pmsg_array[13] = pmsg;}
     if (colMin && planeMax  && doSend)          {emsg_array[14] = emsg++; pmsg_array[14] = pmsg;}
     if (rowMin && colMax)                       {emsg_array[15] = emsg++; pmsg_array[15] = pmsg;}
     if (rowMax && planeMin)                     {emsg_array[16] = emsg++; pmsg_array[16] = pmsg;}
     if (colMax && planeMin)                     {emsg_array[17] = emsg++; pmsg_array[17] = pmsg;}
     if (rowMin && colMin && planeMin)           {cmsg_array[18] = cmsg++; pmsg_array[18] = pmsg; emsg_array[18] = emsg;}
     if (rowMin && colMin && planeMax && doSend) {cmsg_array[19] = cmsg++; pmsg_array[19] = pmsg; emsg_array[19] = emsg;}
     if (rowMin && colMax && planeMin)           {cmsg_array[20] = cmsg++; pmsg_array[20] = pmsg; emsg_array[20] = emsg;}
     if (rowMin && colMax && planeMax && doSend) {cmsg_array[21] = cmsg++; pmsg_array[21] = pmsg; emsg_array[21] = emsg;}
     if (rowMax && colMin && planeMin)           {cmsg_array[22] = cmsg++; pmsg_array[22] = pmsg; emsg_array[22] = emsg;}
     if (rowMax && colMin && planeMax && doSend) {cmsg_array[23] = cmsg++; pmsg_array[23] = pmsg; emsg_array[23] = emsg;}
     if (rowMax && colMax && planeMin)           {cmsg_array[24] = cmsg++; pmsg_array[24] = pmsg; emsg_array[24] = emsg;}
     if (rowMax && colMax && planeMax && doSend) {cmsg_array[25] = cmsg++; pmsg_array[25] = pmsg; emsg_array[25] = emsg;}
   }

   /* post sends */
   #pragma omp parallel sections private(pmsg,emsg,cmsg,destAddr) num_threads(1)
   {
   #pragma omp section
   {
   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dy ;

      if (planeMin) {
         pmsg = pmsg_array[0];

         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64) 
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<sendCount; ++i) {
               destAddr[i+sendCount*fi] = ptr_fi[fi][i];//(domain.*src)(i) ;
            }
            //destAddr += sendCount ;
         }
         //destAddr -= xferFields*sendCount ;

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         //++pmsg ;
      }
      if (planeMax && doSend) {
         pmsg = pmsg_array[1];
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<sendCount; ++i) {
               destAddr[i+sendCount*fi] = /*(domain.*src)*/ ptr_fi[fi][dx*dy*(dz - 1) + i] ;
            }
            //destAddr += sendCount ;
         }
         //destAddr -= xferFields*sendCount ;

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         //++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dz ;

      if (rowMin) {
         pmsg = pmsg_array[2];
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  destAddr[i*dx+j + sendCount*fi] = /*(domain.*src)*/ ptr_fi[fi][i*dx*dy + j] ;
               }
            }
            //destAddr += sendCount ;
         }
         //destAddr -= xferFields*sendCount ;

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         //++pmsg ;
      }
      if (rowMax && doSend) {
         pmsg = pmsg_array[3];
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  destAddr[i*dx+j + sendCount*fi] = /*(domain.*src)*/ ptr_fi[fi][dx*(dy - 1) + i*dx*dy + j] ;
               }
            }
            //destAddr += sendCount ;
         }
         //destAddr -= xferFields*sendCount ;

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         //++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dy * dz ;

      if (colMin) {
         pmsg = pmsg_array[4];
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  destAddr[i*dy + j + sendCount*fi] = /*(domain.*src)*/ ptr_fi[fi][i*dx*dy + j*dx] ;
               }
            }
            //destAddr += sendCount ;
         }
         //destAddr -= xferFields*sendCount ;

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         //++pmsg ;
      }
      if (colMax && doSend) {
         pmsg = pmsg_array[5];
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  destAddr[i*dy + j + sendCount*fi] = /*(domain.*src)*/ ptr_fi[fi][dx - 1 + i*dx*dy + j*dx] ;
               }
            }
            //destAddr += sendCount ;
         }
         //destAddr -= xferFields*sendCount ;

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         //++pmsg ;
      }
   }
   }
   //if (!planeOnly) {
      
   #pragma omp section
   {
      if (rowMin && colMin && not_planeOnly) {
         pmsg = pmsg_array[6];
         emsg = emsg_array[6];
         int toRank = myRank - domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i + dz*fi] = /*(domain.*src)*/  ptr_fi[fi][i*dx*dy] ;
            }
            //destAddr += dz ;
         }
         //destAddr -= xferFields*dz ;
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMin && planeMin && not_planeOnly) {
         pmsg = pmsg_array[7];
         emsg = emsg_array[7];
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            ///Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
               destAddr[i + dx*fi] = /*(domain.*src)*/  ptr_fi[fi][i] ;
            }
            //destAddr += dx ;
         }
         //destAddr -= xferFields*dx ;
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (colMin && planeMin && not_planeOnly) {
         pmsg = pmsg_array[8];
         emsg = emsg_array[8];
         int toRank = myRank - domain.tp()*domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i + dy*fi] = /*(domain.*src)*/ ptr_fi[fi][i*dx] ;
            }
            //destAddr += dy ;
         }
         //destAddr -= xferFields*dy ;
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && colMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[9];
         emsg = emsg_array[9];
         int toRank = myRank + domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i + dz*fi] = /*(domain.*src)*/ ptr_fi[fi][dx*dy - 1 + i*dx*dy] ;
            }
            //destAddr += dz ;
         }
         //destAddr -= xferFields*dz ;
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[10];
         emsg = emsg_array[10];
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
              destAddr[i + fi*dx] = /*(domain.*src)*/ ptr_fi[fi][dx*(dy-1) + dx*dy*(dz-1) + i] ;
            }
            //destAddr += dx ;
         }
         //destAddr -= xferFields*dx ;
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
        // ++emsg ;
      }
   }
   #pragma omp section
   {
      if (colMax && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[11];
         emsg = emsg_array[11];
         int toRank = myRank + domain.tp()*domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i + fi*dy] = /*(domain.*src)*/ ptr_fi[fi][dx*dy*(dz-1) + dx - 1 + i*dx] ;
            }
            //destAddr += dy ;
         }
         //destAddr -= xferFields*dy ;
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && colMin && doSend && not_planeOnly) {
         pmsg = pmsg_array[12];
         emsg = emsg_array[12];
         int toRank = myRank + domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i + fi*dz] = /*(domain.*src)*/ ptr_fi[fi][dx*(dy-1) + i*dx*dy] ;
            }
            //destAddr += dz ;
         }
         //destAddr -= xferFields*dz ;
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMin && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[13];
         emsg = emsg_array[13];
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
               destAddr[i + fi*dx] = /*(domain.*src)*/ ptr_fi[fi][dx*dy*(dz-1) + i] ;
            }
            //destAddr += dx ;
         }
         //destAddr -= xferFields*dx ;
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (colMin && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[14];
         emsg = emsg_array[14];
         int toRank = myRank + domain.tp()*domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i + fi*dy] = /*(domain.*src)*/ptr_fi[fi][dx*dy*(dz-1) + i*dx] ;
            }
            //destAddr += dy ;
         }
         //destAddr -= xferFields*dy ;
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMin && colMax && not_planeOnly) {
         pmsg = pmsg_array[15];
         emsg = emsg_array[15];
         int toRank = myRank - domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i + fi*dz] = /*(domain.*src)*/ ptr_fi[fi][dx - 1 + i*dx*dy] ;
            }
            //destAddr += dz ;
         }
         //destAddr -= xferFields*dz ;
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && planeMin && not_planeOnly) {
         pmsg = pmsg_array[16];
         emsg = emsg_array[16];
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
               destAddr[i + fi*dx] = /*(domain.*src)*/ ptr_fi[fi][dx*(dy - 1) + i] ;
            }
            //destAddr += dx ;
         }
         //destAddr -= xferFields*dx ;
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
        // ++emsg ;
      }
   }
   #pragma omp section
   {
      if (colMax && planeMin && not_planeOnly) {
         pmsg = pmsg_array[17];
         emsg = emsg_array[17];
         int toRank = myRank - domain.tp()*domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE ) is_device_ptr(destAddr) thread_limit(64)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            //Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i + fi*dy] = /*(domain.*src)*/ ptr_fi[fi][dx - 1 + i*dx] ;
            }
            //destAddr += dy ;
         }
         //destAddr -= xferFields*dy ;
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         //++emsg ;
      }
   }
   #pragma omp section
   {
      if (rowMin && colMin && planeMin && not_planeOnly) {
         pmsg = pmsg_array[18];
         emsg = emsg_array[18];
         cmsg = cmsg_array[18];
         /* corner at domain logical coord (0, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL] ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32) 
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/  ptr_fi[fi][0] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   #pragma omp section
   {
      if (rowMin && colMin && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[19];
         emsg = emsg_array[19];
         cmsg = cmsg_array[19];
         /* corner at domain logical coord (0, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*(dz - 1) ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   #pragma omp section
   {
      if (rowMin && colMax && planeMin && not_planeOnly) {
         pmsg = pmsg_array[20];
         emsg = emsg_array[20];
         cmsg = cmsg_array[20];
         /* corner at domain logical coord (1, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx - 1 ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   #pragma omp section
   {
      if (rowMin && colMax && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[21];
         emsg = emsg_array[21];
         cmsg = cmsg_array[21];
         /* corner at domain logical coord (1, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && colMin && planeMin && not_planeOnly) {
         pmsg = pmsg_array[22];
         emsg = emsg_array[22];
         cmsg = cmsg_array[22];
         /* corner at domain logical coord (0, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*(dy - 1) ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && colMin && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[23];
         emsg = emsg_array[23];
         cmsg = cmsg_array[23];
         /* corner at domain logical coord (0, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && colMax && planeMin && not_planeOnly) {
         pmsg = pmsg_array[24];
         emsg = emsg_array[24];
         cmsg = cmsg_array[24];
         /* corner at domain logical coord (1, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy - 1 ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/  ptr_fi[fi][idx] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   #pragma omp section
   {
      if (rowMax && colMax && planeMax && doSend && not_planeOnly) {
         pmsg = pmsg_array[25];
         emsg = emsg_array[25];
         cmsg = cmsg_array[25];
         /* corner at domain logical coord (1, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*dz - 1 ;
         #pragma omp target teams distribute  parallel for  if(target:USE_DEVICE ) is_device_ptr(comBuf) thread_limit(32)
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = /*(domain.*fieldData[fi])*/  ptr_fi[fi][idx] ;
         }
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         //++cmsg ;
      }
   }
   //}
   } //end of parallel sections

   #pragma omp target exit data map(delete:ptr_fi[0:xferFields]) if(USE_DEVICE)

   MPI_Waitall(26, domain.sendRequest, status) ;

   #ifdef USE_NVTX
   nvtxRangeEnd(nvtx_CommSend);
   #endif


}

/******************************************/

void CommSBN(Domain& domain, int xferFields, Domain_member *fieldData) {

   #ifdef USE_NVTX
   nvtxRangeId_t nvtx_CommSBN = nvtxRangeStartA("CommSBN");
   #endif


   if (domain.numRanks() == 1)
      return ;

   /* summation order should be from smallest value to largest */
   /* or we could try out kahan summation! */

   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   Index_t dx = domain.sizeX() + 1 ;
   Index_t dy = domain.sizeY() + 1 ;
   Index_t dz = domain.sizeZ() + 1 ;
   MPI_Status status ;
   Real_t *srcAddr ;
   Index_t rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = 1 ;
   if (domain.rowLoc() == 0) {
      rowMin = 0 ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = 0 ;
   }
   if (domain.colLoc() == 0) {
      colMin = 0 ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = 0 ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = 0 ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = 0 ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   Real_t *ptr_fi[xferFields];
   for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Domain_member dest = fieldData[fi] ;
      ptr_fi[fi] = &(domain.*dest)(0);
   }
   #pragma omp target enter data map(to:ptr_fi[0:xferFields]) if(USE_DEVICE)


/*** IBM: preparing for OpenMP parallel communication  ***/
   int pmsg_array[26], emsg_array[26], cmsg_array[26];
   for (Index_t i = 0; i < 26; ++i)  pmsg_array[i]=0;
   for (Index_t i = 0; i < 26; ++i)  emsg_array[i]=0;
   for (Index_t i = 0; i < 26; ++i)  cmsg_array[i]=0;

   if (planeMin | planeMax) {
      if (planeMin)                  {pmsg_array[0] = pmsg++;}
      if (planeMax)                  {pmsg_array[1] = pmsg++;}
   }
   if (rowMin | rowMax) {
      if (rowMin)                    {pmsg_array[2] = pmsg++;}
      if (rowMax)                    {pmsg_array[3] = pmsg++;}
   }
   if (colMin | colMax) {
      if (colMin)                    {pmsg_array[4] = pmsg++;}
      if (colMax)                    {pmsg_array[5] = pmsg++;}
   }
   if (rowMin && colMin)             {emsg_array[6] = emsg++; pmsg_array[6] = pmsg;}
   if (rowMin && planeMin)           {emsg_array[7] = emsg++; pmsg_array[7] = pmsg;}
   if (colMin && planeMin)           {emsg_array[8] = emsg++; pmsg_array[8] = pmsg;}
   if (rowMax && colMax)             {emsg_array[9] = emsg++; pmsg_array[9] = pmsg;}
   if (rowMax && planeMax)           {emsg_array[10] = emsg++; pmsg_array[10] = pmsg;}
   if (colMax && planeMax)           {emsg_array[11] = emsg++; pmsg_array[11] = pmsg;}
   if (rowMax && colMin)             {emsg_array[12] = emsg++; pmsg_array[12] = pmsg;}
   if (rowMin && planeMax)           {emsg_array[13] = emsg++; pmsg_array[13] = pmsg;}
   if (colMin && planeMax)           {emsg_array[14] = emsg++; pmsg_array[14] = pmsg;}
   if (rowMin && colMax)             {emsg_array[15] = emsg++; pmsg_array[15] = pmsg;}
   if (rowMax && planeMin)           {emsg_array[16] = emsg++; pmsg_array[16] = pmsg;}
   if (colMax && planeMin)           {emsg_array[17] = emsg++; pmsg_array[17] = pmsg;}
   if (rowMin && colMin && planeMin) {cmsg_array[18] = cmsg++; pmsg_array[18] = pmsg; emsg_array[18] = emsg;}
   if (rowMin && colMin && planeMax) {cmsg_array[19] = cmsg++; pmsg_array[19] = pmsg; emsg_array[19] = emsg;}
   if (rowMin && colMax && planeMin) {cmsg_array[20] = cmsg++; pmsg_array[20] = pmsg; emsg_array[20] = emsg;}
   if (rowMin && colMax && planeMax) {cmsg_array[21] = cmsg++; pmsg_array[21] = pmsg; emsg_array[21] = emsg;}
   if (rowMax && colMin && planeMin) {cmsg_array[22] = cmsg++; pmsg_array[22] = pmsg; emsg_array[22] = emsg;}
   if (rowMax && colMin && planeMax) {cmsg_array[23] = cmsg++; pmsg_array[23] = pmsg; emsg_array[23] = emsg;}
   if (rowMax && colMax && planeMin) {cmsg_array[24] = cmsg++; pmsg_array[24] = pmsg; emsg_array[24] = emsg;}
   if (rowMax && colMax && planeMax) {cmsg_array[25] = cmsg++; pmsg_array[25] = pmsg; emsg_array[25] = emsg;}


   #pragma omp parallel sections private(pmsg,emsg,cmsg,srcAddr,status) num_threads(1)
   {
   #pragma omp section
   { 
   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         pmsg = pmsg_array[0];
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               /*(domain.*dest)*/ ptr_fi[fi][i] += srcAddr[i+fi*opCount] ;
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
      if (planeMax) {
         pmsg = pmsg_array[1];
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               /*(domain.*dest)*/ ptr_fi[fi][dx*dy*(dz - 1) + i] += srcAddr[i + fi*opCount] ;
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
         pmsg = pmsg_array[2];
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  /*(domain.*dest)*/ ptr_fi[fi][i*dx*dy + j] += srcAddr[i*dx + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
//LG1 collapse issue
      if (rowMax) {
         pmsg = pmsg_array[3];
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  /*(domain.*dest)*/ ptr_fi[fi][ dx*(dy - 1) +  i*dx*dy + j] += srcAddr[i*dx + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

//LG1 collapse issue
      if (colMin) {
         pmsg = pmsg_array[4];
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  /*(domain.*dest)*/ ptr_fi[fi][i*dx*dy + j*dx] += srcAddr[i*dy + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
//LG1 collapse issue
      if (colMax) {
         pmsg = pmsg_array[5];
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  /*(domain.*dest)*/ ptr_fi[fi][dx - 1 + i*dx*dy + j*dx] += srcAddr[i*dy + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (rowMin & colMin) {
      pmsg = pmsg_array[6];
      emsg = emsg_array[6];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][i*dx*dy] += srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin & planeMin) {
      pmsg = pmsg_array[7];
      emsg = emsg_array[7];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][i] += srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMin & planeMin) {
      pmsg = pmsg_array[8];
      emsg = emsg_array[8];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][i*dx] += srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & colMax) {
      pmsg = pmsg_array[9];
      emsg = emsg_array[9];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy - 1 + i*dx*dy] += srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & planeMax) {
      pmsg = pmsg_array[10];
      emsg = emsg_array[10];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*(dy-1) + dx*dy*(dz-1) + i] += srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMax & planeMax) {
      pmsg = pmsg_array[11];
      emsg = emsg_array[11];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy*(dz-1) + dx - 1 + i*dx] += srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & colMin) {
      pmsg = pmsg_array[12];
      emsg = emsg_array[12];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*(dy-1) + i*dx*dy] += srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin & planeMax) {
      pmsg = pmsg_array[13];
      emsg = emsg_array[13];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy*(dz-1) + i] += srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMin & planeMax) {
      pmsg = pmsg_array[14];
      emsg = emsg_array[14];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy*(dz-1) + i*dx] += srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin & colMax) {
      pmsg = pmsg_array[15];
      emsg = emsg_array[15];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx - 1 + i*dx*dy] += srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & planeMin) {
      pmsg = pmsg_array[16];
      emsg = emsg_array[16];

      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*(dy - 1) + i] += srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMax & planeMin) {
      pmsg = pmsg_array[17];
      emsg = emsg_array[17];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) if(target:USE_DEVICE) is_device_ptr(srcAddr) thread_limit(64) nowait
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/  ptr_fi[fi][dx - 1 + i*dx] += srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin & colMin & planeMin) {
      pmsg = pmsg_array[18];
      emsg = emsg_array[18];
      cmsg = cmsg_array[18];

      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32) nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][0] += comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin & colMin & planeMax) {
      pmsg = pmsg_array[19];
      emsg = emsg_array[19];
      cmsg = cmsg_array[19];

      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32)  nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] += comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin & colMax & planeMin) {
      pmsg = pmsg_array[20];
      emsg = emsg_array[20];
      cmsg = cmsg_array[20];
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32) nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] += comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin & colMax & planeMax) {
      pmsg = pmsg_array[21];
      emsg = emsg_array[21];
      cmsg = cmsg_array[21];
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32) nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] += comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & colMin & planeMin) {
      pmsg = pmsg_array[22];
      emsg = emsg_array[22];
      cmsg = cmsg_array[22];
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32) nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] += comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & colMin & planeMax) {
      pmsg = pmsg_array[23];
      emsg = emsg_array[23];
      cmsg = cmsg_array[23];
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32) nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ptr_fi[fi][idx] += comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & colMax & planeMin) {
      pmsg = pmsg_array[24];
      emsg = emsg_array[24];
      cmsg = cmsg_array[24];
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32) nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] += comBuf[fi] ;
      }
      ++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax & colMax & planeMax) {
      pmsg = pmsg_array[25];
      emsg = emsg_array[25];
      cmsg = cmsg_array[25];
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*dz - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  if(target:USE_DEVICE) is_device_ptr(comBuf) thread_limit(32) nowait
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] += comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   } //end of omp sections
   #pragma omp taskwait 
   #pragma omp target exit data map(delete:ptr_fi[0:xferFields]) if(USE_DEVICE)

   #ifdef USE_NVTX
   nvtxRangeEnd(nvtx_CommSBN);
   #endif

}

/******************************************/

void CommSyncPosVel(Domain& domain) {

   #ifdef USE_NVTX
   nvtxRangeId_t nvtx_CommSyncPosVel = nvtxRangeStartA("CommSyncPosVel");
   #endif


   if (domain.numRanks() == 1)
      return ;

   int myRank ;
   bool doRecv = false ;
   Index_t xferFields = 6 ; /* x, y, z, xd, yd, zd */
   Domain_member fieldData[6] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   Index_t dx = domain.sizeX() + 1 ;
   Index_t dy = domain.sizeY() + 1 ;
   Index_t dz = domain.sizeZ() + 1 ;
   MPI_Status status ;
   Real_t *srcAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;

   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::y ;
   fieldData[2] = &Domain::z ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::yd ;
   fieldData[5] = &Domain::zd ;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;


   Real_t *ptr_fi[xferFields];
   for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Domain_member dest = fieldData[fi] ;
      ptr_fi[fi] = &(domain.*dest)(0);
   }
   #pragma omp target enter data map(to:ptr_fi[0:xferFields]) if(USE_DEVICE)



/*** IBM: preparing for OpenMP parallel communication  ***/
   int pmsg_array[26], emsg_array[26], cmsg_array[26];
   for (Index_t i = 0; i < 26; ++i)  pmsg_array[i]=0;
   for (Index_t i = 0; i < 26; ++i)  emsg_array[i]=0;
   for (Index_t i = 0; i < 26; ++i)  cmsg_array[i]=0;

   int task_filed_offset[26];
   
   if (planeMin | planeMax) {
      if (planeMin && doRecv) {pmsg_array[0] = pmsg++;}
      if (planeMax)           {pmsg_array[1] = pmsg++;}
   }
   if (rowMin | rowMax) {
      if (rowMin && doRecv)   {pmsg_array[2] = pmsg++;}
      if (rowMax)             {pmsg_array[3] = pmsg++;}
   }
   if (colMin | colMax) {
      if (colMin && doRecv)   {pmsg_array[4] = pmsg++;}
      if (colMax)             {pmsg_array[5] = pmsg++;}
   }
   if (rowMin && colMin && doRecv)   {emsg_array[6] = emsg++; pmsg_array[6] = pmsg;}
   if (rowMin && planeMin && doRecv) {emsg_array[7] = emsg++; pmsg_array[7] = pmsg;}
   if (colMin && planeMin && doRecv) {emsg_array[8] = emsg++; pmsg_array[8] = pmsg;}
   if (rowMax && colMax)             {emsg_array[9] = emsg++; pmsg_array[9] = pmsg;}
   if (rowMax && planeMax)           {emsg_array[10] = emsg++; pmsg_array[10] = pmsg;}
   if (colMax && planeMax)           {emsg_array[11] = emsg++; pmsg_array[11] = pmsg;}
   if (rowMax && colMin)             {emsg_array[12] = emsg++; pmsg_array[12] = pmsg;}
   if (rowMin && planeMax)           {emsg_array[13] = emsg++; pmsg_array[13] = pmsg;}
   if (colMin && planeMax)           {emsg_array[14] = emsg++; pmsg_array[14] = pmsg;}
   if (rowMin && colMax && doRecv)   {emsg_array[15] = emsg++; pmsg_array[15] = pmsg;}
   if (rowMax && planeMin && doRecv) {emsg_array[16] = emsg++; pmsg_array[16] = pmsg;}
   if (colMax && planeMin && doRecv) {emsg_array[17] = emsg++; pmsg_array[17] = pmsg;} 
   if (rowMin && colMin && planeMin && doRecv) {cmsg_array[18] = cmsg++; pmsg_array[18] = pmsg; emsg_array[18] = emsg;} 
   if (rowMin && colMin && planeMax) {cmsg_array[19] = cmsg++; pmsg_array[19] = pmsg; emsg_array[19] = emsg;} 
   if (rowMin && colMax && planeMin && doRecv) {cmsg_array[20] = cmsg++; pmsg_array[20] = pmsg; emsg_array[20] = emsg;}
   if (rowMin && colMax && planeMax) {cmsg_array[21] = cmsg++; pmsg_array[21] = pmsg; emsg_array[21] = emsg;} 
   if (rowMax && colMin && planeMin && doRecv) {cmsg_array[22] = cmsg++; pmsg_array[22] = pmsg; emsg_array[22] = emsg;} 
   if (rowMax && colMin && planeMax) {cmsg_array[23] = cmsg++; pmsg_array[23] = pmsg; emsg_array[23] = emsg;}
   if (rowMax && colMax && planeMin && doRecv) {cmsg_array[24] = cmsg++; pmsg_array[24] = pmsg; emsg_array[24] = emsg;}
   if (rowMax && colMax && planeMax) {cmsg_array[25] = cmsg++; pmsg_array[25] = pmsg; emsg_array[25] = emsg;} 

   pmsg = emsg = cmsg = 0;
/*** IBM: end preparing for OpenMP parallel communication  ***/


   #pragma omp parallel sections private(pmsg,emsg,cmsg,srcAddr,status) num_threads(1) 
   {
   #pragma omp section
   { 
   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin && doRecv) {
         pmsg = pmsg_array[0];

         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE) 
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               /*(domain.*dest)*/ ptr_fi[fi][i] = srcAddr[i + fi*opCount] ;
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
      if (planeMax) {
         pmsg = pmsg_array[1];

         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               /*(domain.*dest)*/ptr_fi[fi][dx*dy*(dz - 1) + i] = srcAddr[i + fi*opCount] ;
            }
            //srcAddr += opCount ;
         }
    //     ++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

//LG1 collapse issue
      if (rowMin && doRecv) {
         pmsg = pmsg_array[2];
         //int kk = 2;
         //printf("pecmsg = [%d %d %d], expected = [%d %d %d]\n", pmsg_array[kk],emsg_array[kk], cmsg_array[kk],pmsg,emsg,cmsg);

         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(srcAddr) thread_limit(64) nowait  if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  /*(domain.*dest)*/ ptr_fi[fi][i*dx*dy + j] = srcAddr[i*dx + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
      if (rowMax) {
         pmsg = pmsg_array[3];
         //int kk = 3;
         //printf("pecmsg = [%d %d %d], expected = [%d %d %d]\n", pmsg_array[kk],emsg_array[kk], cmsg_array[kk],pmsg,emsg,cmsg);

         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  /*(domain.*dest)*/ ptr_fi[fi][dx*(dy - 1) + i*dx*dy + j] = srcAddr[i*dx + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;
//LG1 collapse issue
      if (colMin && doRecv) {
         pmsg = pmsg_array[4];
         //int kk = 4;
         //printf("pecmsg = [%d %d %d], expected = [%d %d %d]\n", pmsg_array[kk],emsg_array[kk], cmsg_array[kk],pmsg,emsg,cmsg);

         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  /*(domain.*dest)*/  ptr_fi[fi][i*dx*dy + j*dx] = srcAddr[i*dy + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
//LG1 collapse issue
      if (colMax) {
         pmsg = pmsg_array[5];
         //int kk = 5;
         //printf("pecmsg = [%d %d %d], expected = [%d %d %d]\n", pmsg_array[kk],emsg_array[kk], cmsg_array[kk],pmsg,emsg,cmsg);

         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  /*(domain.*dest)*/ ptr_fi[fi][dx - 1 + i*dx*dy + j*dx] = srcAddr[i*dy + j + fi*opCount] ;
               }
            }
            //srcAddr += opCount ;
         }
         //++pmsg ;
      }
   }
   }
   #pragma omp section
   {
   if (rowMin && colMin && doRecv) {
      pmsg = pmsg_array[6];
      emsg = emsg_array[6];
      //int kk = 6;
      //printf("pecmsg = [%d %d %d], expected = [%d %d %d]\n", pmsg_array[kk],emsg_array[kk], cmsg_array[kk],pmsg,emsg,cmsg);

      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][i*dx*dy] = srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin && planeMin && doRecv) {
      pmsg = pmsg_array[7];
      emsg = emsg_array[7];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][i] = srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMin && planeMin && doRecv) {
      pmsg = pmsg_array[8];
      emsg = emsg_array[8];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][i*dx] = srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && colMax) {
      pmsg = pmsg_array[9];
      emsg = emsg_array[9];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy - 1 + i*dx*dy] = srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && planeMax) {
      pmsg = pmsg_array[10];
      emsg = emsg_array[10];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*(dy-1) + dx*dy*(dz-1) + i] = srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMax && planeMax) {
      pmsg = pmsg_array[11];
      emsg = emsg_array[11];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy*(dz-1) + dx - 1 + i*dx] = srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && colMin) {
      pmsg = pmsg_array[12];
      emsg = emsg_array[12];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*(dy-1) + i*dx*dy] = srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin && planeMax) {
      pmsg = pmsg_array[13];
      emsg = emsg_array[13];

      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy*(dz-1) + i] = srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMin && planeMax) {
      pmsg = pmsg_array[14];
      emsg = emsg_array[14];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*dy*(dz-1) + i*dx] = srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin && colMax && doRecv) {
      pmsg = pmsg_array[15];
      emsg = emsg_array[15];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx - 1 + i*dx*dy] = srcAddr[i + fi*dz] ;
         }
         //srcAddr += dz ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && planeMin && doRecv) {
      pmsg = pmsg_array[16];
      emsg = emsg_array[16];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         // Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx*(dy - 1) + i] = srcAddr[i + fi*dx] ;
         }
         //srcAddr += dx ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (colMax && planeMin && doRecv) {
      pmsg = pmsg_array[17];
      emsg = emsg_array[17];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) nowait if(target:USE_DEVICE)
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         //Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            /*(domain.*dest)*/ ptr_fi[fi][dx - 1 + i*dx] = srcAddr[i + fi*dy] ;
         }
         //srcAddr += dy ;
      }
      //++emsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin && colMin && planeMin && doRecv) {
      pmsg = pmsg_array[18];
      emsg = emsg_array[18];
      cmsg = cmsg_array[18];
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32) nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][0] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin && colMin && planeMax) {
      pmsg = pmsg_array[19];
      emsg = emsg_array[19];
      cmsg = cmsg_array[19];
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32) nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin && colMax && planeMin && doRecv) {
      pmsg = pmsg_array[20];
      emsg = emsg_array[20];
      cmsg = cmsg_array[20];
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32) nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMin && colMax && planeMax) {
      pmsg = pmsg_array[21];
      emsg = emsg_array[21];
      cmsg = cmsg_array[21];
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32) nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && colMin && planeMin && doRecv) {
      pmsg = pmsg_array[22];
      emsg = emsg_array[22];
      cmsg = cmsg_array[22];
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32) nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && colMin && planeMax) {
      pmsg = pmsg_array[23];
      emsg = emsg_array[23];
      cmsg = cmsg_array[23];
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32) nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && colMax && planeMin && doRecv) {
      pmsg = pmsg_array[24];
      emsg = emsg_array[24];
      cmsg = cmsg_array[24];
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32) nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   #pragma omp section
   {
   if (rowMax && colMax && planeMax) {
      pmsg = pmsg_array[25];
      emsg = emsg_array[25];
      cmsg = cmsg_array[25];
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*dz - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      #pragma omp target teams distribute parallel for  is_device_ptr(comBuf) thread_limit(32)  nowait if(target:USE_DEVICE)
      for (Index_t fi=0; fi<xferFields; ++fi) {
         /*(domain.*fieldData[fi])*/ ptr_fi[fi][idx] = comBuf[fi] ;
      }
      //++cmsg ;
   }
   }
   } //end of parallel omp sections

   #pragma omp taskwait
   #pragma omp target exit data map(delete:ptr_fi[0:xferFields]) if(USE_DEVICE)

   #ifdef USE_NVTX
   nvtxRangeEnd(nvtx_CommSyncPosVel);
   #endif

}

/******************************************/

void CommMonoQ(Domain& domain)
{

   #ifdef USE_NVTX
   nvtxRangeId_t nvtx_CommMonoQ = nvtxRangeStartA("CommMonoQ");
   #endif


   if (domain.numRanks() == 1)
      return ;

   int myRank ;
   Index_t xferFields = 3 ; /* delv_xi, delv_eta, delv_zeta */
   Domain_member fieldData[3] ;
   Index_t fieldOffset[3] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t dx = domain.sizeX() ;
   Index_t dy = domain.sizeY() ;
   Index_t dz = domain.sizeZ() ;
   MPI_Status status ;
   Real_t *srcAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   /* point into ghost data area */
   // fieldData[0] = &(domain.delv_xi(domain.numElem())) ;
   // fieldData[1] = &(domain.delv_eta(domain.numElem())) ;
   // fieldData[2] = &(domain.delv_zeta(domain.numElem())) ;
   fieldData[0] = &Domain::delv_xi ;
   fieldData[1] = &Domain::delv_eta ;
   fieldData[2] = &Domain::delv_zeta ;
   fieldOffset[0] = domain.numElem() ;
   fieldOffset[1] = domain.numElem() ;
   fieldOffset[2] = domain.numElem() ;

   Real_t *ptr_fi[xferFields];
   for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Domain_member dest = fieldData[fi] ;
      ptr_fi[fi] = &(domain.*dest)(0);
   }
   #pragma omp target enter data map(to:ptr_fi[0:xferFields],fieldOffset[0:3]) if(USE_DEVICE)


/*** IBM: preparing for OpenMP parallel communication  ***/
   int index_array[6];
   int task_filed_offset[6];
   int cnt =  domain.numElem();
   if (planeMin | planeMax) {
      Index_t opCount = dx * dy ;
      if (planeMin)  {index_array[0] = pmsg++;  task_filed_offset[0] = cnt; cnt += opCount;} 
      if (planeMax)  {index_array[1] = pmsg++;  task_filed_offset[1] = cnt; cnt += opCount;}
   }
   if (rowMin | rowMax) {
      Index_t opCount = dx * dz ;
      if (rowMin) {index_array[2] = pmsg++; task_filed_offset[2] = cnt; cnt += opCount;}
      if (rowMax) {index_array[3] = pmsg++; task_filed_offset[3] = cnt; cnt += opCount;}
  }
  if (colMin | colMax) {
      Index_t opCount = dy * dz ;
      if (colMin) {index_array[4] = pmsg++; task_filed_offset[4] = cnt; cnt += opCount;}
      if (colMax) {index_array[5] = pmsg++; task_filed_offset[5] = cnt; cnt += opCount;}
  }


   pmsg = 0; 
/*** IBM: end of preparing for OpenMP parallel communication  ***/


   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

  #pragma omp parallel sections private(status,pmsg) num_threads(1)
   {
   #pragma omp section
   {
   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;
      
      if (planeMin) {
        Index_t offset = task_filed_offset[0];
#ifdef COMM_OPT
       pmsg  =index_array[0];
#endif
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               //(domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
               ptr_fi[fi][offset /*fieldOffset[fi] */+ i] = srcAddr[i + fi*opCount] ;

            }
            //srcAddr += opCount ;
            //fieldOffset[fi] += opCount ;
         }
    //     for (Index_t fi=0 ; fi<xferFields; ++fi) fieldOffset[fi] += opCount ;
   //      #pragma omp target update to(fieldOffset[0:3]) if(USE_DEVICE)
#ifndef COMM_OPT
         ++pmsg ;
#endif
      }
      if (planeMax) {
        Index_t offset = task_filed_offset[1];
#ifdef COMM_OPT
      pmsg  =index_array[1];
#endif
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
           // Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               //(domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
               ptr_fi[fi][/*fieldOffset[fi]*/offset + i] = srcAddr[i + fi*opCount] ;

            }
            //srcAddr += opCount ;
            //fieldOffset[fi] += opCount ;
         }
         for (Index_t fi=0 ; fi<xferFields; ++fi) fieldOffset[fi] += opCount ;
     //    #pragma omp target update to(fieldOffset[0:3]) if(USE_DEVICE)
#ifndef COMM_OPT
         ++pmsg ;
#endif
      }
   }
   }
   #pragma omp section
   {
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
        Index_t offset = task_filed_offset[2];
#ifdef COMM_OPT
      pmsg  =index_array[2];
#endif
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               //(domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
               ptr_fi[fi][/*fieldOffset[fi]*/ offset + i] = srcAddr[i + fi*opCount] ;

            }
            //srcAddr += opCount ;
            //fieldOffset[fi] += opCount ;
         }
       //  for (Index_t fi=0 ; fi<xferFields; ++fi) fieldOffset[fi] += opCount ;
       //  #pragma omp target update to(fieldOffset[0:3]) if(USE_DEVICE)
#ifndef COMM_OPT
         ++pmsg ;
#endif
      }
      if (rowMax) {
        Index_t offset = task_filed_offset[3];
#ifdef COMM_OPT
      pmsg  =index_array[3];
#endif
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               //(domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
               ptr_fi[fi][/*fieldOffset[fi]*/ offset + i] = srcAddr[i + fi*opCount] ;
            }
            //srcAddr += opCount ;
            //fieldOffset[fi] += opCount ;
         }
        // for (Index_t fi=0 ; fi<xferFields; ++fi) fieldOffset[fi] += opCount ;
        // #pragma omp target update to(fieldOffset[0:3]) if(USE_DEVICE)
#ifndef COMM_OPT
         ++pmsg ;
#endif
      }
   }
   }
   #pragma omp section
   {
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
        Index_t offset = task_filed_offset[4];
#ifdef COMM_OPT
      pmsg  =index_array[4];
#endif
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               //(domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
               ptr_fi[fi][/*fieldOffset[fi]*/ offset + i] = srcAddr[i+fi*opCount] ;
            }
            //srcAddr += opCount ;
            //fieldOffset[fi] += opCount ;
         }
         // for (Index_t fi=0 ; fi<xferFields; ++fi) fieldOffset[fi] += opCount ;
         //#pragma omp target update to(fieldOffset[0:3]) if(USE_DEVICE)
#ifndef COMM_OPT
         ++pmsg ;
#endif
      }
      if (colMax) {
        Index_t offset = task_filed_offset[5];
#ifdef COMM_OPT
      pmsg  =index_array[5];
#endif
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(srcAddr) thread_limit(64) if(target:USE_DEVICE)
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            //Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               //(domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
               ptr_fi[fi][/*fieldOffset[fi]*/ offset + i] = srcAddr[i + fi*opCount] ;
            }
            //srcAddr += opCount ;
         }
#ifndef COMM_OPT
         ++pmsg ;
#endif
      }
   }
   }
   } //end of sections
 
   #pragma omp target exit data map(delete:ptr_fi[0:xferFields],fieldOffset[0:3]) if(USE_DEVICE)

   #ifdef USE_NVTX
   nvtxRangeEnd(nvtx_CommMonoQ);
   #endif


}

#endif
