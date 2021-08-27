#include<stdio.h>

/* call cuHostRegister to page lock the data,
 * then call acc_map_data to put it in the OpenACC present table */
int cuMemHostRegister(void *p, size_t bytes, unsigned int flags);
int cuGetErrorString(int, const char**p);
void pgi_acc_map_host_data(void* a, size_t bytes){
  int r;
  r = cuMemHostRegister(a, bytes, 0);
  if(r){
    const char* s;
    cuGetErrorString(r, &s);
    fprintf(stderr,"acc_map_data(%p,%lu) failed with error %d:%s\n", a, bytes, r, s);
    exit(0);
  }
  /* map the address 'a' to itself */
  //acc_map_data(a, a, bytes);
}
void pgi_acc_unmap_host_data(void* a){
  int r;
  r = cuMemHostUnregister(a, 0);
  if(r){
    const char* s;
    cuGetErrorString(r, &s);
    fprintf(stderr,"acc_unmap_data(%p) failed with error %d:%s\n", a, r, s);
    exit(0);
  }
  /* map the address 'a' to itself */
  //acc_map_data(a, a, bytes);
}

