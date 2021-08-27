//dim3_sweep_kernel(
//  int *diag_len, int *diag_ic, int *diag_j, int *diag_k, int *diag_count, 
//  int ndiag, int grpidx, int *dogrp, int ich, int id, int oct, 
//  int ichunk, int ndimen, int noct, 
//  int nang, int nx, int ny, int nz, int jst, int kst, 
//  int cmom, int src_opt, int ng, int fixup,
//  int jlo, int klo, int jhi, int khi, int jd, int kd,
//  int firsty, int firstz, int lasty, int lastz, 
//  const double* __restrict__ vdelt, const double* __restrict__ w, double *t_xs,
//  double tolr, double hi, double *hj, double *hk, double *mu,
//  volatile double *psii, volatile double *psij, volatile double *psik, 
//  double *jb_in, double *kb_in, double *jb_out, double *kb_out,
//  double *ptr_in, double *ptr_out,
//  const double* __restrict__ qtot, const double* __restrict__ ec, 
//  double *dinv, double *qim,
//  double *flux, double *fluxm, unsigned int nwarp)

__global__ void ckim_kernel(volatile double* d_buf_y, volatile double* d_buf_z,int NG, int NC,int bSizeY,int bSizeZ,int dNple,
                               double* d_ptrin, double* ptrout, volatile int* ptrin_rdy, volatile int* ptrin_dne,
                               volatile int* seqINy, volatile int* seqINz, volatile int* d_seqOUTy, volatile int* d_seqOUTz,
                               int* angrpBG, int* dogrp      )
{
  extern __shared__ double smem[];
  int tt = blockIdx.x;
  int yC = blockIdx.y;
  int zC = blockIdx.z;

  //int aa = threadIdx.x; //angle
  int jj = threadIdx.y;
  int kk = threadIdx.z;

  int seq = 0;

  //issue syncthreads
  for(int ii=0;ii<jj+kk;ii++) __syncthreads();

  for(int angrp=angrpBG[tt];angrp<angrpBG[tt+1];angrp++)
  {
    //convert angrpBG to grp and angle
    int curGrp = dogrp[angrpBG[tt]/NA];
    int curAng = angrpBG[tt]%NA;

    int aa = curAng*16 + threadIdx.x;

    for(int ic=0;ic<NC;ic++)
    {
      if( yC ==0 ) while(seq > seqINy(tt,zC)) ;
      else while(seq > d_buf_y(seq,tt,yC,zC));

      if( zC ==0 ) while(seq > seqINz(tt,zC)) ;
      else while(seq > d_buf_z(seq,tt,yC,zC)) ;

      double* d_buf_y_pull_base = &d_buf_y(seq,tt,yC,zC) + 1;
      double* d_buf_z_pull_base = &d_buf_z(seq,tt,yC,zC) + 1;
      double* d_buf_y_push_base = &d_buf_y(seq,tt,yC+1,zC) + 1;
      double* d_buf_z_push_base = &d_buf_z(seq,tt,yC,zC+1) + 1;

      #define d_buf_y_pull(a,b,c) d_buf_y_pull_base[c + 16 * ( b + 8 * a )]
      #define d_buf_z_pull(a,b,c) d_buf_z_pull_base[c + 16 * ( b + 8 * a )]
      #define d_buf_y_push(a,b,c) d_buf_y_push_base[c + 16 * ( b + 8 * a )]
      #define d_buf_z_push(a,b,c) d_buf_z_push_base[c + 16 * ( b + 8 * a )]

      for(int ii=0;ii<ichunk;ii++)
      {
        //if jj==0, read from GMEM
        //else read from SMEM
        if(jj==0) psij= d_buf_y_pull(ii,kk,aa);
        else psij=smem_psij(jj-1,kk,aa);

        //if kk==0, read from GMEM
        //else read from SMEM
        if(kk==0) psik=d_buf_z_pull(ii,jj,aa);
        else psik=smem_psik(jj,kk-1,aa);

        double psi = qtot(0,ii,jj,kk);
        if ( src_opt == 3) psi = psi + qim(aa,ii,jj,kk,oct,g);

        for(int ll=1;ll<cmom;ll++)
          psi += ec(aa,ll) * qtot(ll,ii,jj,kk);

        double pc = psi + psii*mu(aa)*hi(aa) + psij*hj(aa) + psik*hk(aa);
        if ( vdelt != 0. ) pc += vdelt * ptr_in(aa,ii,jj,kk);

        if ( fixup == 0 )
        {
          psi = pc * dinv(aa,ii,jj,kk);
          psii = 2. * psi - psii;
          psij = 2. * psi - psij;
          if ( ndimen == 3 ) psik = 2.*psi - psik;
          if ( vdelt != 0. )                                         &
            ptr_out(aa,ii,jj,kk) = 2.*psi - ptr_in(aa,ii,jj,kk);
        }

        if(jj==tbY-1) d_buf_y_push(ii,kk,aa)=psij;
        else smem_psij(jj,kk) = psij;

        if(kk==tbZ-1) d_buf_z_push(ii,jj,aa)=psij;
        else smem_psik(jj,kk) = psik;
        //8x8x16x8Bx2 = 16KB
        //save to SMEM or GMEM

        __syncthreads();
      }

      seq++;
       
      //memsync
      __threadfence();

      //relay the signal
      if( yC == tbY-1 ) d_seqOUTy(tt,zC)++;
      else d_buf_y(seq,tt,yC+1,zC)=seq;

      if( zC == tbZ-1 )
      {
        d_seqOUTz(tt,yC)=seq;
        if(yC==3 && tt==6) printf("notifying tt=%d, d_seq=%d, yC=%d, zC=%d\n",tt,seq,yC,zC);
      }
      else
        d_buf_z(seq,tt,yC,zC+1)=seq;
    }

    //perform sum over angle
    //transposed load
    {

    }

    //check readiness of ptr_in  
    while(ptrin_rdy(tt,yC,zC) < seq ) ;
    ptrin_dne(tt,yC,zC) = seq;


  } //angrp

  //done
  //issue remaining syncthreads
  for(int ii=0;ii<14-jj+kk;ii++) __syncthreads();

}


extern "C"
void dim3_sweep_cuda_(
  int *h_dogrp, int *num_grth,
  int *nc, int *ichunk, int *ndimen, int *noct,
  int *nang, int *nx, int *ny, int *nz, int *jst, int *kst, 
  int *cmom, int *src_opt, int *ng, int *fixup,
  int *jlo, int *klo, int *jhi, int *khi, int *jd, int *kd,
  int *lasty, int *lastz, int *firsty, int *firstz, int *timedep, 
  double *tolr, double *hi, 
  int *mtag, int *yp_rcv, int *yp_snd, int *yproc, 
  int *zp_rcv, int *zp_snd, int *zproc, int *ycomm, int *zcomm, int *iproc)
{ 
  int ndogrp = 0;
  for (int i = 0; i < *num_grth; i++) {
    if (h_dogrp[i] == 0) break;
    ndogrp++;
  }

  if (ndogrp == 0) 
    return;

  int avgAngrp = (NA * ndogrp)/nTBG;
  int remAngrp = (NA * ndogrp)%nTBG;
  printf("avgAngrp=%d remAngrp=%d\n",avgAngrp,remAngrp);

  angrpBG[0]=0;
  for(int ii=0;ii<nTBG;ii++)
  {
    angrpL[ii]=NC*(avgAngrp + (remAngrp>0));
    angrpBG[ii+1] = angrpBG[ii] + angrpL[ii]; 
    remAngrp--;
  }

  CTimer timer, timer2;
  timer2.Start();

  int streamcount, istream;
  MPI_Status status;

  //start======================================================================
  //initial recv post
  for(int tt=0;tt<nTBG; tt++)
  {
    //z-dir
    if (*zp_rcv != *zproc && z_comm != MPI_COMM_NULL) 
    {
      for(int jj=0;jj<tbY;jj++)
      {
        seqRVz(tt,jj)++;
        int tag = getMpiRTagZ(*zproc,seqRVz(tt,jj),tt,jj); //Tag matching may be an issue?
        MPI_Irecv( &h_RBufZ(seqRVz(tt,jj),tt,jj), bSizeZ+1, MPI_DOUBLE_PRECISION,
                      *zp_rcv, tag, z_comm, &zRreq(tt,jj));
        if(tt==6 && jj==3) printf("posting recv tt=%d, y=%d, d_seqRVz=%d tag=%d\n", tt,jj,seqRVz(tt,jj),tag);
      }
    }
    else
    {
      for(int jj=0;jj<tbY;jj++)
        seqINz(tt,jj)= angrpL[tt];
    }

    //y-dir
    if (*yp_rcv != *zproc && y_comm != MPI_COMM_NULL) 
    {
      for(int kk=0;kk<tbZ;kk++)
      {
        seqRVy(tt,kk)++;
        int tag = getMpiRTagY(*yproc,seqRVy(tt,kk),tt,kk);
        MPI_Irecv( &h_RBufY(seqRVy(tt,kk),tt,kk), bSizeY+1, MPI_DOUBLE_PRECISION,
                      *yp_rcv, tag, y_comm, &yRreq(tt,kk));
      }
    }
    else
    {
      for(int kk=0;kk<tbZ;kk++)
        seqINy(tt,kk)=angrpL[tt];
    }
  }
  //inital recv done========================================================


  cudaStream_t kerStream,memStream;
  cudaStreamCreate ( &kerStream) ;
  cudaStreamCreate ( &memStream) ;
  //=================== device start ===================================

  ckim_kernel<<<dim3(nTBG,tbY,tbZ),dim3(achunk,jchunk,kchunk),0,kerStream>>>(
d_buf_y, d_buf_z, NG, NC, bSizeY, bSizeZ, dNple,
                                        d_ptrin, ptrout, ptrin_rdy, ptrin_dne,
                                        seqINy, seqINz, d_seqOUTy, d_seqOUTz,
                                        angrpBG, dogrp);

  //====================================================================
  

  MPI_Status status0,status1;
  bool alldone=false;
  while(!alldone)
  {
     //=================== receive ===================================
     alldone=true;
     int yind,zind,yflag,zflag;
     zflag=0; yflag=0;

     if (*yp_rcv != *yproc && y_comm != MPI_COMM_NULL) MPI_Testany(nTBG*tbZ,yRreq,&yind,&yflag,&status0);
     if (*zp_rcv != *zproc && z_comm != MPI_COMM_NULL) MPI_Testany(nTBG*tbY,zRreq,&zind,&zflag,&status1);

     for(int tt=0;tt<nTBG;tt++)
     {
       for(int jj=0;jj<tbY;jj++)
         if( seqINz(tt,jj)+1 < angrpL[tt] ) {alldone=false; break;}

       for(int kk=0;kk<tbY;kk++)
         if( seqINy(tt,kk)+1 < angrpL[tt] ) {alldone=false; break;}
     }

     if(zflag)
     {
       //get tt,jj from yind
       int tt = zind / tbY;
       int jj = zind % tbY;

       cudaMemcpyAsync(&d_buf_z(seqRVz(tt,jj),tt,jj,0), &h_RBufZ(seqRVz(tt,jj),tt,jj) , bSizeZ+1,  cudaMemcpyHostToDevice, memStream );
       seqRVz(tt,jj)++;
       if(seqRVz(tt,jj)<angrpL[tt])
       {
         int tag = getMpiRTagZ(*zproc,seqRVz(tt,jj),tt,jj); //Tag matching may be an issue?
         MPI_Irecv( &h_RBufZ(seqRVz(tt,jj),tt,jj), bSizeZ+1, MPI_DOUBLE_PRECISION,
                      *zp_rcv, tag, z_comm, &zRreq(tt,jj));
       }
       cudaStreamSynchronize(memStream); 	
       seqINz(tt,jj)++;
       if(tt==6 && jj==3) printf("received updating tt=%d, y=%d, d_seqInz=%d\n", tt,jj,seqINz(tt,jj));
     }

     if(yflag)
     {
       //get tt,jj from yind
       int tt = yind / tbZ;
       int kk = yind % tbZ;

       cudaMemcpyAsync(&d_buf_y(seqINy(tt,kk),tt,0,kk), &h_RBufY(seqINy(tt,kk),tt,kk), bSizeY,  cudaMemcpyHostToDevice, memStream );
       seqRVy(tt,kk)++;
       if(seqRVy(tt,kk)<angrpL[tt])
       {
         int tag = getMpiRTagY(*yproc,seqRVy(tt,kk),tt,kk);
         MPI_Irecv( &h_RBufY(seqRVy(tt,kk),tt,kk), bSizeY, MPI_DOUBLE_PRECISION,
                       *yp_rcv, tag, y_comm, &yRreq(tt,kk));
       }
       cudaStreamSynchronize(memStream); 	
       seqINy(tt,kk)++;
     }

     //=================== send ===================================
     for(int tt=0;tt<nTBG; tt++)
     {
     //y-dir
       if (*yp_snd != *yproc && y_comm != MPI_COMM_NULL) 
       {
         for(int kk=0;kk<tbZ;kk++)
           if(seqOUTy(tt,kk)+1<angrpL[tt])
           {
             alldone=false;
             if( d_seqOUTy(tt,kk) > seqOUTy(tt,kk) )
             {
                //post send
                seqOUTy(tt,kk)++;  //it starts at -1
                {
                  cudaMemcpyAsync(&h_SBufY(seqOUTy(tt,kk),tt,kk), &d_buf_y(seqOUTy(tt,kk),tt,tbY,kk),  bSizeY,  cudaMemcpyDeviceToHost, memStream );
                  cudaStreamSynchronize(memStream); 	
              
                  int tag = getMpiRTagY(*yp_snd,seqOUTy(tt,kk),tt,kk);
                  MPI_Isend( &h_SBufY(seqOUTy(tt,kk),tt,kk), bSizeY, MPI_DOUBLE_PRECISION,
                            *yp_snd, tag, y_comm, &ySreq(seqOUTy(tt,kk),tt,kk) ); //there could be many send outstanding... //how to handle? check if send done?
                 }
              }
            }
       }
       //z-dir
       if (*zp_snd != *zproc && z_comm != MPI_COMM_NULL) 
       {
         for(int jj=0;jj<tbY;jj++)
           if(seqOUTz(tt,jj)+1<angrpL[tt])
           {
             alldone=false;
             if( d_seqOUTz(tt,jj) > seqOUTz(tt,jj) )
             {
                seqOUTz(tt,jj)++;  //it starts at -1
                //if(tt==6 && jj==3) printf("checking rank:%d  tt=%d jj=%d seqOUTz=%d\n",myrank,tt,jj,seqOUTz(tt,jj));
                //post send
                cudaMemcpyAsync(&h_SBufZ(seqOUTz(tt,jj),tt,jj), &d_buf_z(seqOUTz(tt,jj),tt,tbY,jj), bSizeZ+1,  cudaMemcpyDeviceToHost, memStream );
                cudaStreamSynchronize(memStream); 	
             
                int tag = getMpiRTagZ(*zp_snd,seqOUTz(tt,jj),tt,jj);
                if(tt==6 && jj==3) printf("posting send tt=%d, y=%d, seqOUTz=%d and tag=%d\n", tt,jj,seqOUTz(tt,jj),tag);
                MPI_Isend( &h_SBufY(seqOUTz(tt,jj),tt,jj), bSizeZ+1, MPI_DOUBLE_PRECISION,
                          *zp_snd, tag, z_comm, &zSreq(seqOUTz(tt,jj),tt,jj) ); //there could be many send outstanding... //how to handle? check if send done?
             }
           }
       }
     } //tt

     


     //ptr_in buffering
     //ptr_out is done by direct store from GPU
     //hope that OS can handle huge pinned memory
     
     //GPU has tbY x tbZ x NtbYZ contexted
     //ptr_in takes NC * achunk * ichunk * tbY x tbZ x NtbYZ x 8 x 8 
     //with NC*ichunk=512, achunk=16, this would be 448MB
     //each threadblock publish its state
     //host keeps seq number and slack

     const int slackPTR=4;
     for(int tt=0;tt<nTBG;tt++)
     {
       for(int jj=0;jj<tbY;jj++)
       for(int kk=0;kk<tbZ;kk++)
       {
         if(h_ptrin_rdy(tt,jj,kk)+1<angrpL[tt])
         {
           if(h_ptrin_rdy(tt,jj,kk) - slackPTR < ptrin_dne(tt,jj,kk) )
           {
             h_ptrin_rdy(tt,jj,kk)++;
             ptrin_rdy(tt,jj,kk)=h_ptrin_rdy(tt,jj,kk);
           }
         }
       }
     }
  } // while
  
}  //end of dim3_sweep_cuda_

