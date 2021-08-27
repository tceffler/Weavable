__global__ void set_cnt(double* d_buf_y, double* d_buf_z, int dNple, int tbY, int tbZ, int bSizeY, int bSizeZ, int ichunk, int jchunk, int kchunk)
{
  int tt = blockIdx.x;
  int zC = blockIdx.z;
  int yC = blockIdx.y;

  int seq = threadIdx.x;

  d_buf_y(seq,tt,yC,zC) = -1;
  d_buf_z(seq,tt,yC,zC) = -1;
}

__global__ void set_d_buf_y(double* d_buf_y, int dNple, int tbY, int tbZ, int bSizeY, int ichunk, int jchunk, int kchunk, int func, int rank, int avgAngrp)
{
  int tt = blockIdx.x;
  int seq = blockIdx.y;
  int zC = blockIdx.z;

  int aa = threadIdx.x; //angle
  //int ii = threadIdx.y;

  for(int ii = threadIdx.y; ii<ichunk; ii+=32)
  {
    volatile double* d_buf_y_pull_base;
    if (func==0) d_buf_y_pull_base = &d_buf_y(seq,tt,0,zC);
    else d_buf_y_pull_base = &d_buf_y(seq,tt,tbY,zC);
 
    #define d_buf_y_pull(a,b,c) d_buf_y_pull_base[c + WARP_SIZE * ( b + kchunk * a )]
    if(aa==0 && ii==0) d_buf_y_pull(0,0,0)=dNple*(avgAngrp+1);
    d_buf_y_pull_base += 1;
    for(int kk=0;kk<kchunk;kk++)
    {
      double tmp=ii + ichunk*(kk + kchunk * aa) ;
      //if( func==0) d_buf_y_pull(ii,kk,aa) = tmp;
      if( func==0) d_buf_y_pull(ii,kk,aa) = 0.;
      else
      {
        if (d_buf_y_pull(ii,kk,aa) != tmp)
         printf("Y:%d:disagree at tt=%d,seq=%d,zC=%d,aa=%d,ii=%d,kk=%d got %f but %f\n",rank, tt,seq,zC,aa,ii,kk, d_buf_y_pull(ii,kk,aa), tmp);
      }
    }
  }
  #undef d_buf_y_pull
}


__global__ void set_d_buf_z(double* d_buf_z, int dNple, int tbY, int tbZ, int bSizeZ, int ichunk, int jchunk, int kchunk,int func, int rank, int avgAngrp)
{
  int tt = blockIdx.x;
  int seq = blockIdx.y;
  int yC = blockIdx.z;

  //int aa = threadIdx.x; //angle
  int aa = threadIdx.x;
  //int ii = threadIdx.y;

  for(int ii = threadIdx.y; ii< ichunk;ii+=32)
  {

    volatile double* d_buf_z_pull_base;
    if( func==0) d_buf_z_pull_base = &d_buf_z(seq,tt,yC,0);
    else  d_buf_z_pull_base = &d_buf_z(seq,tt,yC,tbZ);
 
 
    #define d_buf_z_pull(a,b,c) d_buf_z_pull_base[c + WARP_SIZE * ( b + jchunk * a )]
    if(aa==0 && ii==0) d_buf_z_pull(0,0,0)=dNple*(avgAngrp+1);
    d_buf_z_pull_base += 1;
    for(int jj=0;jj<jchunk;jj++)
    {
      double tmp=ii + ichunk*(jj + jchunk*aa);
      //if (func==0) d_buf_z_pull(ii,jj,aa) = tmp;
      if (func==0) d_buf_z_pull(ii,jj,aa) = 0.;
      else
      {
        if (d_buf_z_pull(ii,jj,aa) != tmp) printf("Z:%d:disagree at tt=%d,seq=%d,yC=%d,aa=%d,ii=%d,jj=%d got %f but %f %p\n",rank, tt,seq,yC,aa,ii,jj,d_buf_z_pull(ii,jj,aa), tmp, &d_buf_z_pull(ii,jj,aa));
      }
    }
  }
  #undef d_duf_z_pull
}

__global__ void chk_d_buf_y(double* d_buf_y, int dNple, int tbY, int tbZ, int bSizeY, int jchunk, int kchunk)
{
  int tt = blockIdx.x;
  int seq = blockIdx.y;
  int zC = blockIdx.z;

  //int aa = threadIdx.x; //angle
  int aa = threadIdx.y;
  int ii = threadIdx.x;

  volatile double* d_buf_y_pull_baseI = &d_buf_y(seq,tt,0,zC) + 1;
  volatile double* d_buf_y_pull_baseO = &d_buf_y(seq,tt,tbY,zC) + 1;
  #define d_buf_y_pullI(a,b,c) d_buf_y_pull_baseO[c + WARP_SIZE * ( b + jchunk * a )]
  #define d_buf_y_pullO(a,b,c) d_buf_y_pull_baseI[c + WARP_SIZE * ( b + jchunk * a )]
  for(int kk=0;kk<kchunk;kk++)
    if(d_buf_y_pullI(ii,kk,aa) != d_buf_y_pullO(ii,kk,aa))  
      printf("disagree at tt=%d,seq=%d,zC=%d,ii=%d,kk=%d,aa=%d,\n",tt,seq,zC,ii,kk,aa);

  #undef d_buf_y_pullI
  #undef d_buf_y_pullO
}
