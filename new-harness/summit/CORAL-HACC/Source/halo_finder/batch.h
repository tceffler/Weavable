#pragma once

#ifdef BASELINE
#define BATCH_SIZE 1
#else
#define BATCH_SIZE 16
#endif
  
class BatchInfo {
  public:
    int size;
    int offset;
    int count_[BATCH_SIZE];
    int count1_[BATCH_SIZE];
    float *xx_[BATCH_SIZE];
    float *yy_[BATCH_SIZE];
    float *zz_[BATCH_SIZE];
    float *xx1_[BATCH_SIZE];
    float *yy1_[BATCH_SIZE];
    float *zz1_[BATCH_SIZE];
    float *mass1_[BATCH_SIZE];
    float *vx_[BATCH_SIZE];
    float *vy_[BATCH_SIZE];
    float *vz_[BATCH_SIZE];
    inline void clear() {
      size=0;
      offset=0;
    }
    inline void add(int count, int count1, float* xx, float* yy, float* zz, 
        float* xx1, float* yy1, float* zz1, float* mass1, 
        float* vx, float* vy, float* vz) {
      //copy device pointers into batch
      count_[size]=count;
      count1_[size]=count1;
      xx_[size]=xx;
      yy_[size]=yy;
      zz_[size]=zz;
      xx1_[size]=xx1;
      yy1_[size]=yy1;
      zz1_[size]=zz1;
      mass1_[size]=mass1;
      vx_[size]=vx;
      vy_[size]=vy;
      vz_[size]=vz;
      size++;
      offset+=count1;
      //printf("Adding batch, count1: %d, new offset: %d\n", count1, offset);
    }
};
