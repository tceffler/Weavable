#define jchunk 4
#define kchunk 4
#define achunk 32
#define bchunk 16

__global__ void 
dim3_kernel(
  int *dogrp,  
  int ichunk, int jchunk0, int kchunk0, int achunk0, int oct, int ndimen, 
  int nx, int ny, int nz, int nang, int noct,
  int NA, int NC, int NG, int cmom, int src_opt, int fixup,
  int tbY, int tbZ, int nTBG,
  int ptrNple,
  int timedep,
  const double* __restrict__ vdelt, const double* __restrict__ w, double *t_xs,
  double tolr, double hi, double *hj, double *hk, double *mu,
  const double* __restrict__ qtot, const double* __restrict__ ec, 
  double *dinv, double *qim,
  double *psi_save, double *flux, double* fluxm,
  volatile double* d_buf_y, volatile double* d_buf_z, int bSizeY,int bSizeZ,int dNple,
  double* d_ptrin, double* d_ptrout, volatile int* ptrin_rdy, volatile int* ptrin_dne,
  volatile int* seqINy, volatile int* seqINz, volatile int* d_seqOUTy, volatile int* d_seqOUTz,
  int* angrpBG, int maxAngrp, int rank, int yzFlip ,
  int bufNple, double* h_RBufY, double* h_RBufZ, double* h_SBufY, double* h_SBufZ
  )
{
  __shared__ double smem_pool[1024*3]; //32KB

  __shared__ double* smem_psij; smem_psij = &(smem_pool[0]);
  __shared__ double* smem_psik; smem_psik = &(smem_pool[32*4*4]);
  __shared__ double* wpsi; wpsi = &(smem_pool[0]);
  __shared__ double* wpsum; wpsum = &(smem_pool[33*32]);
  __shared__ double* smem_ec; smem_ec = &(smem_pool[65*32]);
  

  #define smem_psij(a,b,c) smem_psij[c + 32*(a + jchunk*(b))]
  #define smem_psik(a,b,c) smem_psik[c + 32*(a + jchunk*(b))]

  int tt = blockIdx.x;
  int yC = blockIdx.y;
  int zC = blockIdx.z;

  int jj = threadIdx.y;
  int kk = threadIdx.z;
  int aa = threadIdx.x;
  int bb = aa % 16; //actual angle
  int xFlip = threadIdx.x >> 4;
  oct = ( 1 - xFlip ) + oct;

  int seq = 0;
  int ptrinSeq=0;

  int yFlip = 0x2 & yzFlip;
  int zFlip = 0x1 & yzFlip;
  int yy = yC*jchunk + jj; yy = yFlip?(ny-yy-1):yy;
  int zz = zC*kchunk + kk; zz = zFlip?(nz-zz-1):zz;

  psi_save += tt * nx*ny*nz*WARP_SIZE;   
  #define psi_save(i,j,k,a) psi_save[a + WARP_SIZE*(i + nx*(j + ny*(k)))]

  //issue syncthreads

  for(int angrp=angrpBG[tt];angrp<angrpBG[tt+1];angrp++)
  {
    //convert angrpBG to grp and angle
    int curGrp = dogrp[angrp/NA]-1; //index copied from fortran array
    int curAng = angrp%NA;

    int ll = jj + jchunk*kk;
    if(ll < cmom-1) smem_ec[aa + 32*ll] = ec_aa(curAng,bb,ll+1,oct);
     __syncthreads();

    double* psi_save_b = &(psi_save(0,yy,zz,aa));
     
    double* t_xs_aa_value = &(t_xs_aa(0,yy,zz,curAng));

    for(int ii=0;ii<jj+kk;ii++) {__syncthreads();  __syncthreads();}
    //check readiness of ptr_in  
    if (aa==0 && jj==0 && kk==0 && yC==0 && zC==0)
      while(ptrin_rdy(tt,0,0) < ptrinSeq );
    __syncthreads();

    double psij,psik,psii=0.;
    for(int ic=0;ic<NC;ic++)
    {
      if(aa==0) {
        if( zC ==0 ) while(seq > seqINz(tt,0)) ;
        else  while(seq > d_buf_z(ic,tt,yC,zC)) ;
      }
      if (aa==0) {
        if( yC == 0 ) while(seq > seqINy(tt,0)) ;
        else while(seq > d_buf_y(ic,tt,yC,zC));
      }
      __syncwarp();

      volatile double* d_buf_y_pull_base = &d_buf_y(ic,tt,yC,zC) + 16 ;
      volatile double* d_buf_z_pull_base = &d_buf_z(ic,tt,yC,zC) + 16 ;
      volatile double* d_buf_y_push_base = &d_buf_y(ic,tt,yC+1,zC) + 16;
      volatile double* d_buf_z_push_base = &d_buf_z(ic,tt,yC,zC+1) + 16;
      double *ptrin_base = &d_ptrin(ptrinSeq,tt,yC,zC); 
      double *ptrout_base = &d_ptrout(ptrinSeq,tt,yC,zC); 

      if (yC == 0 && jj == 0 ) { d_buf_y_pull_base = &h_RBufY(seq,tt,zC,0) + 16 ; }
      if (zC == 0 && kk == 0 ) { d_buf_z_pull_base = &h_RBufZ(seq,tt,yC,0) + 16 ; }

      if (yC == tbY-1 && jj == jchunk-1 ) { d_buf_y_push_base = &h_SBufY(seq,tt,zC,0) + 16; }
      if (zC == tbZ-1 && kk == kchunk-1 ) { d_buf_z_push_base = &h_SBufZ(seq,tt,yC,0) + 16; }

      #define d_buf_y_pull(i,k,a) d_buf_y_pull_base[a + achunk * ( k + kchunk * (i) )]
      #define d_buf_z_pull(i,j,a) d_buf_z_pull_base[a + achunk * ( j + jchunk * (i) )]
      #define d_buf_y_push(i,k,a) d_buf_y_push_base[a + achunk * ( k + kchunk * (i) )]
      #define d_buf_z_push(i,j,a) d_buf_z_push_base[a + achunk * ( j + jchunk * (i) )]
      #define d_ptrin_lcl(a,i,j,k)  ptrin_base[a + achunk*(i + nx*(j + jchunk*(k)))]
      #define d_ptrout_lcl(a,i,j,k) ptrout_base[a + achunk*(i + nx*(j + jchunk*(k)))]

      //volatile double* smem_psij_jj = &(smem_psij(jj-1,kk,aa));

      volatile double* d_buf_y_pull_value = &d_buf_y_pull(0,kk,aa);
      volatile double* smem_psij_value = &smem_psij(jj-1,kk,aa);

      volatile double* d_buf_z_pull_value = &d_buf_z_pull(0,jj,aa);
      volatile double* smem_psik_value = &smem_psik(jj,kk-1,aa);

      const double* qtot_aa_value = &qtot_aa(0,0,yy,zz,curGrp);

      volatile double mu_hi_aa_value = mu_aa(curAng,bb)*hi;
      volatile double hj_aa_value = hj_aa(curAng,bb);
      volatile double hk_aa_value = hk_aa(curAng,bb);

      volatile double* d_ptrin_lcl_value = &(d_ptrin_lcl(aa,0,jj,kk));
      volatile double* dinv_aa_value = &(dinv_aa(curAng,bb,0,yy,zz,curGrp));
      volatile double* d_ptrout_lcl_value = &(d_ptrout_lcl(aa,0,jj,kk));

      volatile double* d_buf_y_push_value = &d_buf_y_push(0,kk,aa);
      volatile double* d_buf_z_push_value = &d_buf_z_push(0,jj,aa);

      for(int ii=0;ii<ichunk;ii++)
      {
//        int tmp_xx=ii + ic * ichunk;
        int xx = ii + ic * ichunk; xx = xFlip?(nx-xx-1):xx;
        if((yzFlip & NOTYRECV ) && (jj==0) && (yC==0)  ) {psij=0;}
        else
        {
          if(jj==0) psij=*d_buf_y_pull_value; //d_buf_y_pull(ii,kk,aa);
          else psij=*smem_psij_value; //smem_psij(jj-1,kk,aa);
          d_buf_y_pull_value += achunk*kchunk;
        }

        if((yzFlip & NOTZRECV) && (kk==0) && (zC==0)) {psik=0;}
        else
        {
          if(kk==0) psik=*d_buf_z_pull_value; //d_buf_z_pull(ii,jj,aa);
          else psik=*smem_psik_value; //smem_psik(jj,kk-1,aa);
          d_buf_z_pull_value += achunk*jchunk;
        }

        __syncthreads();

        double psi = *(qtot_aa_value + xx*cmom) ;//qtot_aa(0,xx,yy,zz,curGrp); //no angle dep
//        if ( xx < 3 && yy < 3 &&  zz<3 && curGrp==0 && bb==0 && curAng==0)
//          printf("xx=%d yy=%d zz=%d angle=%d group=%d oct=%d qtot=%f\n",xx+1,yy+1,zz+1,bb+1,curGrp+1,oct+1,psi);
//        __syncwarp();

        for(int ll=1;ll<cmom;ll++)
          psi += smem_ec[aa + 32*(ll-1)] * *(qtot_aa_value + ll + xx*cmom); //qtot_aa(ll,xx,yy,zz,curGrp);

//        if ( xx >10 && xx < 20 &&  yy<4 && zz<4 && curGrp==0 && bb<4 && curAng==0)
//          printf("xx=%d yy=%d zz=%d angle=%d group=%d oct=%d psi=%f\n",xx+1,yy+1,zz+1,bb+1,curGrp+1,oct+1,psi);
//        __syncwarp();

        double pc = psi + psii*mu_hi_aa_value +  psij*hj_aa_value + psik*hk_aa_value;

        if ( vdelt[curGrp] != 0. ) pc += vdelt[curGrp] * (*d_ptrin_lcl_value); //d_ptrin_lcl(aa,ii,jj,kk);

        if ( fixup == 0 )
        {
          psi = pc * (*(dinv_aa_value+xx*nang)); //dinv_aa(curAng,bb,xx,yy,zz,curGrp);
          psii = 2. * psi - psii;
          psij = 2. * psi - psij;
          if ( ndimen == 3 ) psik = 2.*psi - psik;

          if ( vdelt[curGrp] != 0. )                                         
          {
            *d_ptrout_lcl_value = 2.*psi - (*d_ptrin_lcl_value); //d_ptrin_lcl(aa,ii,jj,kk);
          }
        }
        else
        {
          //double sum_hv = 4;
          //double hv1 = 1, hv2 = 1, hv3 = 1, hv4 = 1;
          unsigned hv_bitV=0xFF;
          double fxhv1, fxhv2, fxhv3, fxhv4;
          pc = pc * (*(dinv_aa_value+xx*nang)); //dinv_aa(curAng,bb,xx,yy,zz,curGrp);

          while(1)
          {
            fxhv1 = 2.0*pc - psii;
            fxhv2 = 2.0*pc - psij;
            if (ndimen == 3) fxhv3 = 2.0*pc - psik;
            if (vdelt[curGrp] != 0.0) fxhv4 = 2.0*pc - (*d_ptrin_lcl_value); //d_ptrin_lcl(aa,ii,jj,kk);
            hv_bitV = hv_bitV & ((fxhv1 < 0)?0xFE:0xFF);
            hv_bitV = hv_bitV & ((fxhv2 < 0)?0xFD:0xFF);
            hv_bitV = hv_bitV & ((fxhv3 < 0)?0xFB:0xFF);
            hv_bitV = hv_bitV & ((fxhv4 < 0)?0xF7:0xFF);
            if ((hv_bitV >> 4) == (hv_bitV & 0xF)) break;
            hv_bitV = ((0xF & hv_bitV) << 4) + (0xF & hv_bitV);

            //pc = psii*mu_aa(curAng,bb)*hi*(1. + hv1) + psij*hj_aa(curAng,bb)*(1. + hv2) + psik*hk_aa(curAng,bb)*(1. + hv3);
            //pc = psii*mu_hi_aa_value*(1. + hv1) + psij*hj_aa_value*(1. + hv2) + psik*hk_aa_value*(1. + hv3);
            pc  = psii*mu_hi_aa_value* ((hv_bitV & 0x1)?2.:1.);
            pc += psij*hj_aa_value*    ((hv_bitV & 0x2)?2.:1.);
            pc += psik*hk_aa_value*    ((hv_bitV & 0x4)?2.:1.); 
            //if (vdelt[curGrp] != 0.0) pc += vdelt[curGrp] * d_ptrin_lcl(aa,ii,jj,kk)*(1.0+hv4);
            if (vdelt[curGrp] != 0.0) pc += vdelt[curGrp] * (*d_ptrin_lcl_value)*(((hv_bitV & 0x8)?2.:1.));
            pc = psi + 0.5*pc;
            //double den = t_xs_aa(xx,yy,zz,curGrp) + mu_aa(curAng,bb)*hi*hv1 + hj_aa(curAng,bb)*hv2 + hk_aa(curAng,bb)*hv3 + vdelt[curGrp]*hv4;
            double den;
            den  = *(t_xs_aa_value+xx);
            den += (hv_bitV&0x1)?mu_hi_aa_value:0;
            den += (hv_bitV&0x2)?hj_aa_value:0;
            den += (hv_bitV&0x4)?hk_aa_value:0;
            den += (hv_bitV&0x8)?vdelt[curGrp]:0;
 
            if (den > tolr) pc = pc/den;
            else pc = 0.0;
          }
          psi = pc;
          psii = (hv_bitV&0x1)?fxhv1:0;
          psij = (hv_bitV&0x2)?fxhv2:0;
          if (ndimen == 3) psik = (hv_bitV&0x4)?fxhv3:0;
          //if (vdelt[curGrp] != 0.)  { d_ptrout_lcl(aa,ii,jj,kk) = fxhv4 * hv4; }
          if (vdelt[curGrp] != 0.)  { *d_ptrout_lcl_value = (hv_bitV&0x8)?fxhv4:0; }
        } 
        __syncwarp();
        d_ptrin_lcl_value += achunk;
        d_ptrout_lcl_value += achunk;
  
          
        //psi_save(ii+ic*ichunk,yy,zz,aa) = psi;
        *psi_save_b = psi; psi_save_b += WARP_SIZE;

        if(jj==jchunk-1) *(d_buf_y_push_value)=psij;
        else *(smem_psij_value+32) = psij;
        d_buf_y_push_value += achunk*kchunk;

        if(kk==kchunk-1) *(d_buf_z_push_value)=psik;
        else *(smem_psik_value+32*jchunk) = psik;
        d_buf_z_push_value += achunk*jchunk;

        __syncthreads();
      }

      //memsync
      if ( jj == jchunk-1 && kk== kchunk-1 && aa==0)
      {
        __threadfence();
    

        d_buf_y(ic,tt,yC+1,zC) = seq;
        d_buf_z(ic,tt,yC,zC+1) = seq;

//#ifdef VALUEPRINT
//      if(curGrp==1 &&  aa==0) printf("push oct=%d curAng=%d tt=%d yC=%d zC=%d ic*ichunk=%d seq=%d buf_y=%f,%p buf_z=%f,%p\n",oct,curAng,tt,yC,zC,ic*ichunk,seq,
      //if( aa==0) printf("push yC=%d zC=%d ic=%d seq=%d buf_y=%f,%p buf_z=%f,%p\n",yC,zC,ic,seq,
//              d_buf_y(ic,tt,yC+1,zC),&d_buf_y(ic,tt,yC+1,zC),d_buf_z(ic,tt,yC,zC+1),&d_buf_z(ic,tt,yC,zC+1));
//#endif

//        printf("d_buf_y(%d,%d,%d,%d)=%d\n",ic,tt,yC+1,zC,seq);
//        printf("d_buf_z(%d,%d,%d,%d)=%d\n",ic,tt,yC,zC+1,seq);

        //relay the signal
        if(( yC == tbY-1 ) && ( zC == tbZ-1))
        { 
          __threadfence_system();
          d_seqOUTy(tt,zC)=seq;
          d_seqOUTz(tt,yC)=seq;
//        if(aa==0 && yC==3) printf("Y:notifying tt=%d, d_seq=%d, yC=%d, zC=%d\n",tt,seq,yC,zC);
        }
      }

      seq++;
    }
    if ( jj==jchunk-1 && kk==kchunk-1 && aa==0 ) ptrin_dne(tt,yC,zC) = ptrinSeq;

    __syncthreads();

    ptrinSeq++;

    //perform sum over angle
    //transposed load
    //    flux(i,j,k) = flux(i,j,k) + SUM( w*psi )
    //    DO l = 1, cmom-1
    //      fluxm(l,i,j,k) = fluxm(l,i,j,k) + SUM( ec(:,l+1)*w*psi )
    //    END DO

    //issue remaining syncthreads
    for(int ii=0;ii<(jchunk+kchunk-2)-(jj+kk);ii++) {__syncthreads(); __syncthreads();}

    //idea
    //4x4 warps and 32 threads
    //
    
     __threadfence_block();

    int tid=aa;
    int wid = jj + jchunk*kk;
    int bb=aa % 16;
    oct = oct & 0x6;
    #define wpsi(i,j) wpsi[i + 33*(j)]
    #define wpsum(l,a) wpsum[a + 32*(l)]
    for(int ic=0;ic<NC;ic++)
    {

      int xxT = ic * ichunk + tid;
      int xxW0 = wid+ic*ichunk; xxW0 = xFlip?(nx-1-xxW0):xxW0;
      int xxW1 = xFlip?xxW0-16:xxW0+16;
      //xFlip=1 16-31. oct=oct &0x6
      //xFlip=0  0-15. oct=oct &0x6 + 1

      for(int jjl=0;jjl<4;jjl++)
      for(int kkl=0;kkl<4;kkl++)
      {
        //shadow yy,zz
        int yy = yC*jchunk + jjl; yy = yFlip?(ny-yy-1):yy;
        int zz = zC*kchunk + kkl; zz = zFlip?(nz-zz-1):zz;

        __syncthreads();

        wpsi(wid,tid) = w_aa(curAng,bb) * psi_save(xxW0,yy,zz,tid);
        wpsi(wid+16,tid) = w_aa(curAng,bb) * psi_save(xxW1,yy,zz,tid);
        //tid  0-15. xxW0 0-15, xxW1 16-31, 
        //tid 16-15. flip : xxW0 0-15, xxW1 16-31, 

        __syncthreads();

        if(wid<cmom)
        {
          if(wid==0)
          { 
            if ( curAng == 0 && oct == 0 ) wpsum(wid,tid) = 0;
            else wpsum(wid,tid) = flux_aa(xxT,yy,zz,curGrp);
          }
          else
          {
            if ( curAng == 0 && oct == 0 ) wpsum(wid,tid) = 0;
            else wpsum(wid,tid) = fluxm_aa(wid-1,xxT,yy,zz,curGrp);
          }

          //thread maps x-dir
          //warp maps mom
          if(wid==0)
          {
            for(int an=0;an<16;an++)
            {
              wpsum(wid,tid) += wpsi(tid,an);
              wpsum(wid,tid) += wpsi(tid,an+16);
            }
          }
          else
          {
            const double* ec_aa_value0 = &ec_aa(curAng,0,wid,oct);
            const double* ec_aa_value1 = &ec_aa(curAng,0,wid,oct+1);

            for(int an=0;an<16;an++)
            {
//              wpsum(wid,tid) +=  ec_aa(curAng,an,wid,oct+1) * wpsi(tid,an);
//              wpsum(wid,tid) +=  ec_aa(curAng,an,wid,oct)   * wpsi(tid,an+16);

//                if ( xxT==0 && yy==0 && zz==0 && wid==1 && curGrp==0)
//                   printf("ang=%d ec=%e wpsi=%e oct=%d ec=%e wpsi=%e oct=%d\n",curAng*16+an,*(ec_aa_value1),wpsi(tid,an),oct+1,*(ec_aa_value0),wpsi(tid,an+16),oct);

              wpsum(wid,tid) +=  *(ec_aa_value1)  * wpsi(tid,an);
              wpsum(wid,tid) +=  *(ec_aa_value0)  * wpsi(tid,an+16);
              ec_aa_value0++;
              ec_aa_value1++;

            }
//            if ( xxT==0 && yy==0 && zz==0 && wid==1 && curGrp==0)
//                 printf("oct=%d fluxm=%e\n",oct,wpsum(wid,tid));
          }
        

          if(wid==0)
          { 
            flux_aa(xxT,yy,zz,curGrp) = wpsum(wid,tid);
          }
          else
          {
            fluxm_aa(wid-1,xxT,yy,zz,curGrp) = wpsum(wid,tid);
          }
        }
      }
    } //end of sum

    //bring it back 
    oct = ( 1 - xFlip ) + oct;

    //buffer consumption checking
    if (yC < tbY-1)
       while ( (seq-1) >  d_buf_y((NC-1),tt,yC+2,zC)) {;}

    if (zC < tbZ-1)
       while ( (seq-1) >  d_buf_z((NC-1),tt,yC,zC+2)) {;}

  } //angrp

  //done

  #undef d_buf_y_pull
  #undef d_buf_z_pull
#undef jchunk
#undef kchunk
#undef achunk
#undef bchunk
  
  
  __threadfence_system();
  
}
