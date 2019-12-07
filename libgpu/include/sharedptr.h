/*
  sharedptr.h

  Convenience class for shared CPU/GPU allocations.
  Based on the X10 Runtime ideas described in Pai et al. in PACT 2012.
  Also see NVIDIA Hemi's array.h at <https://github.com/harrism/hemi>

  Copyright (C) 2014--2016, The University of Texas at Austin

  Author: Sreepathi Pai  <sreepai@ices.utexas.edu>
*/

#pragma once
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <mpi.h>
#include <climits>
#include "cutil_subset.h"

template <typename T>
class Shared {
  T** ptrs;
  bool* owner;
  bool* isCPU;
  int max_devices;
  size_t nmemb;

public:
  Shared() { nmemb = 0; }

  Shared(size_t nmemb) {
    this->nmemb = nmemb;
    max_devices = 2;
    ptrs        = (T**)calloc(max_devices, sizeof(T*));
    owner       = (bool*)calloc(max_devices, sizeof(bool));
    isCPU       = (bool*)calloc(max_devices, sizeof(bool));

    isCPU[0] = true;

    for (int i = 0; i < max_devices; i++)
      owner[i] = true;
  }

  size_t size() const { return this->nmemb; }

  void alloc(size_t nmemb) {
    assert(this->nmemb == 0);

    this->nmemb = nmemb;

    max_devices = 2;
    ptrs        = (T**)calloc(max_devices, sizeof(T*));
    owner       = (bool*)calloc(max_devices, sizeof(bool));
    isCPU       = (bool*)calloc(max_devices, sizeof(bool));

    isCPU[0] = true;

    for (int i = 0; i < max_devices; i++)
      owner[i] = true;
  }

  void free() {
    for (int i = 0; i < max_devices; i++)
      free_device(i);
  }

  bool free_device(int device = 0) {
    assert(device < max_devices);

    if (!ptrs[device])
      return true;

    if (isCPU[device])
      ::free(ptrs[device]);
    else {
      if (cudaFree(ptrs[device]) == cudaSuccess)
        ptrs[device] = NULL;
      else
        return false;
    }

    return true;
  }

  bool find_owner(int& o) {
    int i;
    for (i = 0; i < max_devices; i++)
      if (owner[i]) {
        o = i;
        break;
      }

    return i < max_devices;
  }

  T* cpu_rd_ptr() {
    if (ptrs[0] == NULL)
      ptrs[0] = (T*)calloc(nmemb, sizeof(T));

    if (!owner[0]) {
      int o;
      if (find_owner(o))
        copy(o, 0);

      owner[0] = true;
    }

    return ptrs[0];
  }

  T* cpu_wr_ptr(bool overwrite = false) {
    if (ptrs[0] == NULL)
      ptrs[0] = (T*)calloc(nmemb, sizeof(T));

    if (!owner[0]) {
      if (!overwrite) {
        int o;
        if (find_owner(o))
          copy(o, 0);
      }

      owner[0] = true;
    }

    for (int i = 1; i < max_devices; i++)
      owner[i] = false;

    return ptrs[0];
  }

  T* gpu_rd_ptr(int device = 1) /* device >= 1 */
  {
    assert(device >= 1);

    if (ptrs[device] == NULL)
      CUDA_SAFE_CALL(cudaHostAlloc(&ptrs[device], nmemb * sizeof(T), cudaHostAllocPortable));

    if (!owner[device]) {
      int o;
      if (find_owner(o))
        copy(o, device);

      owner[device] = true;
    }

    return ptrs[device];
  }

  T* gpu_wr_ptr(bool overwrite = false, int device = 1) {
    assert(device >= 1);

    if (ptrs[device] == NULL) {
      CUDA_SAFE_CALL(cudaHostAlloc(&ptrs[device], nmemb * sizeof(T), cudaHostAllocPortable));
    }

    if (!owner[device]) {
      if (!overwrite) {
        int o;
        if (find_owner(o))
          copy(o, device);
      }

      owner[device] = true;
    }

    for (int i = 0; i < max_devices; i++)
      if (i != device)
        owner[i] = false;

    return ptrs[device];
  }

  T* zero_gpu(int device = 1) {
    T* p = gpu_wr_ptr(true, device);
    CUDA_SAFE_CALL(cudaMemset(p, 0, sizeof(T) * nmemb));
    return p;
  }

  void copy(int src, int dst) {
    if (!ptrs[src])
      return;

    assert(ptrs[dst]);

    if (isCPU[dst] && !isCPU[src]) {
      CUDA_SAFE_CALL(cudaMemcpy(ptrs[dst], ptrs[src], nmemb * sizeof(T),
                                cudaMemcpyDefault));
                                //cudaMemcpyDeviceToHost));
    } else if (!isCPU[dst] && !isCPU[src]) {
      CUDA_SAFE_CALL(cudaMemcpy(ptrs[dst], ptrs[src], nmemb * sizeof(T),
                                cudaMemcpyDefault));
                                //cudaMemcpyDeviceToDevice));
    } else if (!isCPU[dst] && isCPU[src]) {
      CUDA_SAFE_CALL(cudaMemcpy(ptrs[dst], ptrs[src], nmemb * sizeof(T),
                                cudaMemcpyDefault));
                                //cudaMemcpyHostToDevice));
    } else
      abort(); // cpu-to-cpu not implemented
  }

  __device__ __host__ T* ptr() {
#ifdef __CUDA_ARCH__
    return ptrs[1]; // TODO: this is invalid beyond one gpu device!
#else
    return ptrs[0];
#endif
  }
};

template <typename T>
class DeviceOnly {
  T* ptr;
  size_t nmemb;
  T max_val;
  MPI_Request send_req, recv_req;

public:
  DeviceOnly() {
    ptr   = NULL;
    nmemb = 0;
    max_val = std::numeric_limits<T>::max();
  }

  DeviceOnly(size_t nmemb) {
    ptr = NULL;
    alloc(nmemb);
    max_val = std::numeric_limits<T>::max();
  }

  size_t size() const { return nmemb; }

  void alloc(size_t nmemb) {
    assert(this->nmemb == 0);
    this->nmemb = nmemb;
    CUDA_SAFE_CALL(cudaHostAlloc(&ptr, nmemb * sizeof(T), cudaHostAllocPortable));
  }

  bool free() {
    if (ptr == NULL)
      return true;
    if (cudaFree(ptr) == cudaSuccess) {
      ptr = NULL;
      return true;
    }
    return false;
  }

  T* zero_gpu() {
    CUDA_SAFE_CALL(cudaMemset(ptr, 0, sizeof(T) * nmemb));
    return ptr;
  }

  void copy_to_gpu(T* cpu_ptr) { copy_to_gpu(cpu_ptr, nmemb); }

  void copy_to_gpu(T* cpu_ptr, size_t nuseb) {
    if (cpu_ptr == NULL)
      return;
    assert(ptr != NULL);
    assert(nuseb <= nmemb);
    CUDA_SAFE_CALL(
        //cudaMemcpy(ptr, cpu_ptr, nuseb * sizeof(T), cudaMemcpyHostToDevice));
        cudaMemcpy(ptr, cpu_ptr, nuseb * sizeof(T), cudaMemcpyDefault));
  }

  void copy_to_cpu(T* cpu_ptr) { copy_to_cpu(cpu_ptr, nmemb); }

  void send_mpi(size_t nuseb, unsigned to_id) {
	  if (ptr == NULL)
		  return;
	  assert(nuseb <= nmemb);
	  MPI_Send(ptr, nuseb * sizeof(T), MPI_BYTE,
             to_id, 10000, MPI_COMM_WORLD);
  }

  void recv_mpi() {
    recv_mpi(ULLONG_MAX);
  }

  void recv_mpi(size_t nuseb) {
    if (ptr == NULL)
      return;
    MPI_Status stat;
    MPI_Recv(ptr, nuseb * sizeof(T), MPI_BYTE,
             MPI_ANY_SOURCE, 10000, MPI_COMM_WORLD,
             &stat);
    int count;
    MPI_Get_count(&stat, MPI_BYTE, &count);
    printf("received size: %d\n", count);
  }

  void send_impi(size_t nuseb, unsigned to_id, unsigned tag) {
    if (ptr == NULL) { return; }
    assert(nuseb <= nmemb);
    MPI_Isend(ptr, nuseb * sizeof(T), MPI_BYTE,
              to_id, tag, MPI_COMM_WORLD, &send_req);
    printf("Dyn Send data: %d\n", nuseb);
    //MPI_Status req_stat;
    //MPI_Wait(&req, &req_stat);
  }

  void recv_impi(unsigned tag) {
    recv_impi(ULLONG_MAX, tag);
  }

  void recv_impi(size_t nuseb, unsigned tag) {
    if (ptr == NULL) { return; }
    assert(nuseb <= nmemb);
    MPI_Irecv(ptr, nuseb * sizeof(T), MPI_BYTE, MPI_ANY_SOURCE,
              tag, MPI_COMM_WORLD, &recv_req);
    int flag = 0;
    MPI_Status req_stat;
    MPI_Wait(&recv_req, &req_stat);
    //MPI_Wait(&send_req, &req_stat);
    /*
    MPI_Test(&recv_req, &flag, &req_stat); 
    while (!flag) {
      MPI_Test(&recv_req, &flag, &req_stat);
    }
    */
    int count;
    MPI_Get_count(&req_stat, MPI_BYTE, &count);
    printf("shared ptr recv data: %d\n", count);
  }

  void copy_to_cpu(T* cpu_ptr, size_t nuseb) {
    if (ptr == NULL)
      return;
    assert(cpu_ptr != NULL);
    assert(nuseb <= nmemb);
    CUDA_SAFE_CALL(
        //cudaMemcpy(cpu_ptr, ptr, nuseb * sizeof(T), cudaMemcpyDeviceToHost));
        cudaMemcpy(cpu_ptr, ptr, nuseb * sizeof(T), cudaMemcpyDefault));
  }

  __device__ __host__ T* device_ptr() {
#ifdef __CUDA_ARCH__
    return ptr; // TODO: this is invalid beyond one gpu device!
#else
    return ptr;
#endif
  }
};
