
#ifndef  VECTOR_UVM_H 
#define VECTOR_UVM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


template <class T>
struct UMAllocator {
  typedef T value_type;
  UMAllocator() {}
  template <class U> UMAllocator(const UMAllocator<U>& other);

  T* allocate(std::size_t n)
  {
    T* ptr;
#ifdef USE_UVM
    cudaMallocManaged(&ptr, n*sizeof(T));
#else
    ptr = (T*) malloc(n*sizeof(T));
#endif
    return ptr;
  }

  void deallocate(T* p, std::size_t n)
  {
#ifdef USE_UVM
    cudaFree(p);
#else
    free(p);
#endif
  }
};

template <class T, class U>
bool operator==(const UMAllocator<T>&, const UMAllocator<U>&);
template <class T, class U>
bool operator!=(const UMAllocator<T>&, const UMAllocator<U>&);


#endif

