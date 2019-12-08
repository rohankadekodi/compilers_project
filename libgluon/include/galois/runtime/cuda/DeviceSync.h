/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/*
 */

/**
 * @file DeviceSync.h
 *
 * CUDA header for GPU runtime
 *
 * @todo better file description + document this file
 */
#pragma once
#include "galois/cuda/DynamicBitset.h"
#include "galois/cuda/Context.h"
#include "galois/runtime/DataCommMode.h"
#include "cub/util_allocator.cuh"
#include <cuda_runtime.h>

#ifdef __GALOIS_CUDA_CHECK_ERROR__
#define check_cuda_kernel                                                      \
  check_cuda(cudaDeviceSynchronize());                                         \
  check_cuda(cudaGetLastError());
#else
#define check_cuda_kernel check_cuda(cudaGetLastError());
#endif

//#define GPU_LOG
MPI_Request send_req = MPI_REQUEST_NULL;
MPI_Request recv_req;
MPI_Request req_list[2] = { send_req, recv_req };
MPI_Status send_stat;
MPI_Status recv_stat;
//#define GPUDIRECT_LOG

enum SharedType { sharedMaster, sharedMirror };
enum UpdateOp { setOp, addOp, minOp };

void kernel_sizing(dim3& blocks, dim3& threads) {
  threads.x = 256;
  threads.y = threads.z = 1;
  blocks.x              = ggc_get_nSM() * 8;
  blocks.y = blocks.z = 1;
}

__global__ void deserialize_isupdate(uint8_t* buffer,
                                     DynamicBitset* __restrict__ is_updated,
                                     index_type is_updated_size) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;

  int test;
  for (index_type src = 0+tid; src < is_updated_size; src += nthreads) {
    //is_updated[src] = *reinterpret_cast<uint64_t *>(buffer[src]);
    //test = is_updated[src];
    //((uint64_t*) is_updated)[src] = ((uint64_t *)buffer)[src];
    /*
    is_updated[src]  = (((uint64_t)buffer[src+tid*8+7]) << 56) & 0xFF;
    is_updated[src] |= (((uint64_t)buffer[src+tid*8+6]) << 48) & 0xFF;
    is_updated[src] |= (((uint64_t)buffer[src+tid*8+5]) << 40) & 0xFF;
    is_updated[src] |= (((uint64_t)buffer[src+tid*8+4]) << 32) & 0xFF;                    
    is_updated[src] |= (((uint64_t)buffer[src+tid*8+3]) << 24) & 0xFF;
    is_updated[src] |= (((uint64_t)buffer[src+tid*8+2]) << 16) & 0xFF;
    is_updated[src] |= (((uint64_t)buffer[src+tid*8+1]) << 8) & 0xFF;
    is_updated[src] |= (((uint64_t)buffer[src+tid*8]) & 0xFF);
    */
#ifdef GPU_LOG
    printf("\t\tisupdate[%d]=%d\n", src, ((uint64_t *)is_updated)[src]);
#endif
  }
}

template <typename OffsetIteratorType>
__global__ void deserialize_offsets(uint8_t* buffer,
                                    const OffsetIteratorType offsets,
                                    index_type offset_size) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;

  for (index_type src = 0+tid; src < offset_size; src += nthreads) {
    //offsets[src] = ((uint32_t *)buffer)[src];
    offsets[src] = ((((uint32_t)buffer[src+tid*4+3]) << 24) & 0xFF)
                   | ((((uint32_t)buffer[src+tid*4+2]) << 16) & 0xFF)
                   | ((((uint32_t)buffer[src+tid*4+1]) << 8) & 0xFF)
                   | (((uint32_t)buffer[src+tid*4]) & 0xFF); 
#ifdef GPU_LOG
    printf("\t\toffsets[%d]=%d\n", src, offsets[src]);
#endif
  }
}

template <typename DataType>
__global__ void deserialize_data(uint8_t* buffer,
                                 DataType* __restrict__ data,
                                 index_type data_size) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;

  for (index_type src = 0+tid; src < data_size; src += nthreads) {
    //data[src] = ((DataType *)buffer)[src];
      data[src] = ((((uint32_t)buffer[src+tid*4+3]) << 24) & 0xFF)
                 | ((((uint32_t)buffer[src+tid*4+2]) << 16) & 0xFF)
                 | ((((uint32_t)buffer[src+tid*4+1]) << 8) & 0xFF)
                 | (((uint32_t)buffer[src+tid*4]) & 0xFF);
#ifdef GPU_LOG
    printf("\t\tdata[%d]=%d\n", src, data[src]);
#endif
  }
}

__global__ void serialize_isupdate(uint8_t* buffer,
                                   uint64_t* __restrict__ is_updated,
                                   index_type is_updated_size) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  uint64_t test = 0;
  for (index_type src = 0+tid; src < is_updated_size; src += nthreads){
    buffer[src+7+8*tid] = (uint8_t)((is_updated[src] >> 56) & 0xFF);
    buffer[src+6+8*tid] = (uint8_t)((is_updated[src] >> 48) & 0xFF);
    buffer[src+5+8*tid] = (uint8_t)((is_updated[src] >> 40) & 0xFF);
    buffer[src+4+8*tid] = (uint8_t)((is_updated[src] >> 32) & 0xFF);
    buffer[src+3+8*tid] = (uint8_t)((is_updated[src] >> 24) & 0xFF);
    buffer[src+2+8*tid] = (uint8_t)((is_updated[src] >> 16) & 0xFF);
    buffer[src+1+8*tid] = (uint8_t)((is_updated[src] >> 8) & 0xFF);
    buffer[src+0+8*tid] = (uint8_t)((is_updated[src]) & 0xFF);
    //test = is_updated[src];
    //test = ((uint64_t *)buffer)[src];
    //((uint64_t *)buffer)[src] = ((uint64_t*)is_updated)[src];

#ifdef GPU_LOG
    printf("\t\tSent: isupdate buffer[%d]=%d\n", src, ((uint64_t *)buffer)[src]);
#endif
  }
}

template <typename OffsetIteratorType>
__global__ void serialize_offsets(uint8_t* buffer,
                                  const OffsetIteratorType offsets,
                                  index_type offset_size) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  for (index_type src = 0+tid; src < offset_size; src += nthreads){
    //((uint32_t *)buffer)[src] = offsets[src];
    buffer[src+3+4*tid] = (uint8_t)((offsets[src] >> 24) & 0xFF);
    buffer[src+2+4*tid] = (uint8_t)((offsets[src] >> 16) & 0xFF);
    buffer[src+1+4*tid] = (uint8_t)((offsets[src] >> 8) & 0xFF);
    buffer[src+0+4*tid] = (uint8_t)(offsets[src] & 0xFF);
#ifdef GPU_LOG
    printf("\t\tSent: offsets buffer[%d]=%d\n", src, ((uint32_t *)buffer)[src]);
#endif
  }
}

template <typename DataType>
__global__ void serialize_data(uint8_t* buffer,
                               DataType* __restrict__ data,
                               index_type data_size) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  for (index_type src = 0+tid; src < data_size; src += nthreads){
    //((DataType *)buffer)[src] = data[src];
    buffer[src+3+4*tid] = (uint8_t)((data[src] >> 24) & 0xFF);
    buffer[src+2+4*tid] = (uint8_t)((data[src] >> 16) & 0xFF);
    buffer[src+1+4*tid] = (uint8_t)((data[src] >> 8) & 0xFF);
    buffer[src+0+4*tid] = (uint8_t)(data[src] & 0xFF);
#ifdef GPU_LOG
    printf("\t\tSent: data buffer[%d]=%d\n", src, ((DataType *)buffer)[src]);
#endif
  }
}

template <typename DataType>
__global__ void batch_get_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 DataType* __restrict__ subset,
                                 const DataType* __restrict__ array) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  //printf("\tSent data---\n");
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    subset[src]    = array[index];
#ifdef GPUDIRECT_LOG
    printf("\t\tSrc: %d, Index: %d, Data: %d\n", src, index, subset[src]); 
#endif
  }
}

template <typename DataType, typename OffsetIteratorType>
__global__ void batch_get_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 DataType* __restrict__ subset,
                                 const DataType* __restrict__ array) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  //printf("\tSent data---\n");
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    subset[src]    = array[index];
#ifdef GPUDIRECT_LOG
    printf("\t\tSrc: %d, Index: %d, Data: %d\n", src, index, subset[src]); 
#endif
  }
}

template <typename DataType>
__global__ void batch_get_reset_subset(index_type subset_size,
                                       const unsigned int* __restrict__ indices,
                                       DataType* __restrict__ subset,
                                       DataType* __restrict__ array,
                                       DataType reset_value) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    subset[src]    = array[index];
    array[index]   = reset_value;
  }
}

template <typename DataType, typename OffsetIteratorType>
__global__ void batch_get_reset_subset(index_type subset_size,
                                       const unsigned int* __restrict__ indices,
                                       const OffsetIteratorType offsets,
                                       DataType* __restrict__ subset,
                                       DataType* __restrict__ array,
                                       DataType reset_value) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    subset[src]    = array[index];
    array[index]   = reset_value;
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_set_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    array[index]   = subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_set_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    array[index]   = subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_add_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    array[index] += subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_add_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    array[index] += subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_min_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    if (array[index] > subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_min_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    //if (subset_size == 681493) {
    //printf("Local Id %d\n", src);
    //printf("Local ID %d, Global Id offset: %d\n", src, offsets[src]); 
    //}
    unsigned index = indices[offsets[src]];
    //if (subset_size == 681493) {
    //printf("Global ID %d\n", index);
    //}
    if (array[index] > subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
    //printf("src %d done\n", src);
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_max_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    if (array[index] < subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_max_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    if (array[index] < subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
  }
}

template <typename DataType>
__global__ void batch_reset(DataType* __restrict__ array, index_type begin,
                            index_type end, DataType val) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = end;
  for (index_type src = begin + tid; src < src_end; src += nthreads) {
    array[src] = val;
  }
}

__global__ void
batch_get_subset_bitset(index_type subset_size,
                        const unsigned int* __restrict__ indices,
                        DynamicBitset* __restrict__ is_subset_updated,
                        DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    if (is_array_updated->test(index)) {
      //printf("INDEX %d is updated\n", src);
      is_subset_updated->set(src);
    }
  }
}

// inclusive range
__global__ void bitset_reset_range(DynamicBitset* __restrict__ bitset,
                                   size_t vec_begin, size_t vec_end, bool test1,
                                   size_t bit_index1, uint64_t mask1,
                                   bool test2, size_t bit_index2,
                                   uint64_t mask2) {
  unsigned tid      = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  for (size_t src = vec_begin + tid; src < vec_end; src += nthreads) {
    bitset->batch_reset(src);
  }

  if (tid == 0) {
    if (test1) {
      bitset->batch_bitwise_and(bit_index1, mask1);
    }
    if (test2) {
      bitset->batch_bitwise_and(bit_index2, mask2);
    }
  }
}

template <typename DataType>
void reset_bitset_field(struct CUDA_Context_Field<DataType>* field,
                        size_t begin, size_t end) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);
  const DynamicBitset* bitset_cpu = field->is_updated.cpu_rd_ptr();
  assert(begin <= (bitset_cpu->size() - 1));
  assert(end <= (bitset_cpu->size() - 1));

  size_t vec_begin = (begin + 63) / 64;
  size_t vec_end;

  if (end == (bitset_cpu->size() - 1))
    vec_end = bitset_cpu->vec_size();
  else
    vec_end = (end + 1) / 64; // floor

  size_t begin2 = vec_begin * 64;
  size_t end2   = vec_end * 64;

  bool test1;
  size_t bit_index1;
  uint64_t mask1;

  bool test2;
  size_t bit_index2;
  uint64_t mask2;

  if (begin2 > end2) {
    test2 = false;

    if (begin < begin2) {
      test1       = true;
      bit_index1  = begin / 64;
      size_t diff = begin2 - begin;
      assert(diff < 64);
      mask1 = ((uint64_t)1 << (64 - diff)) - 1;

      // create or mask
      size_t diff2 = end - end2 + 1;
      assert(diff2 < 64);
      mask2 = ~(((uint64_t)1 << diff2) - 1);
      mask1 |= ~mask2;
    } else {
      test1 = false;
    }
  } else {
    if (begin < begin2) {
      test1       = true;
      bit_index1  = begin / 64;
      size_t diff = begin2 - begin;
      assert(diff < 64);
      mask1 = ((uint64_t)1 << (64 - diff)) - 1;
    } else {
      test1 = false;
    }

    if (end >= end2) {
      test2       = true;
      bit_index2  = end / 64;
      size_t diff = end - end2 + 1;
      assert(diff < 64);
      mask2 = ~(((uint64_t)1 << diff) - 1);
    } else {
      test2 = false;
    }
  }

  // Only require for GPU part.
  bitset_reset_range<<<blocks, threads>>>(field->is_updated.gpu_rd_ptr(),
                                          vec_begin, vec_end, test1, bit_index1,
                                          mask1, test2, bit_index2, mask2);
}

template <typename DataType>
void reset_data_field(struct CUDA_Context_Field<DataType>* field, size_t begin,
                      size_t end, DataType val) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  batch_reset<DataType><<<blocks, threads>>>(
      field->data.gpu_wr_ptr(), (index_type)begin, (index_type)end, val);
}

void get_offsets_from_bitset(index_type bitset_size,
                             unsigned int* __restrict__ offsets,
                             DynamicBitset* __restrict__ bitset,
                             size_t* __restrict__ num_set_bits) {
  cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
  DynamicBitsetIterator flag_iterator(bitset);
  IdentityIterator offset_iterator;
  Shared<size_t> num_set_bits_ptr;
  num_set_bits_ptr.alloc(1);
  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                             offset_iterator, flag_iterator, offsets,
                             num_set_bits_ptr.gpu_wr_ptr(true), bitset_size);
  check_cuda_kernel;
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  //CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                             offset_iterator, flag_iterator, offsets,
                             num_set_bits_ptr.gpu_wr_ptr(true), bitset_size);
  check_cuda_kernel;
  //CUDA_SAFE_CALL(cudaFree(d_temp_storage));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  *num_set_bits = *num_set_bits_ptr.cpu_rd_ptr();
}

template <typename DataType, SharedType sharedType, bool reset>
void batch_get_shared_field(struct CUDA_Context_Common* ctx,
                            struct CUDA_Context_Field<DataType>* field,
                            unsigned from_id, uint8_t* send_buffer,
                            DataType i = 0) {
  struct CUDA_Context_Shared* shared;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
  } else { // sharedMirror
    shared = &ctx->mirror;
  }
  DeviceOnly<DataType>* shared_data = &field->shared_data;
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  // ggc::Timer timer("timer"), timer1("timer1"), timer2("timer2");
  // timer.start();
  // timer1.start();
  size_t v_size = shared->num_nodes[from_id];
  if (reset) {
    batch_get_reset_subset<DataType><<<blocks, threads>>>(
        v_size, shared->nodes[from_id].device_ptr(), shared_data->device_ptr(),
        field->data.gpu_wr_ptr(), i);
  } else {
    batch_get_subset<DataType><<<blocks, threads>>>(
        v_size, shared->nodes[from_id].device_ptr(), shared_data->device_ptr(),
        field->data.gpu_rd_ptr());
  }
  check_cuda_kernel;
  // timer1.stop();
  // timer2.start();
  DataCommMode data_mode = onlyData;
  memcpy(send_buffer, &data_mode, sizeof(data_mode));
  memcpy(send_buffer + sizeof(data_mode), &v_size, sizeof(v_size));
  shared_data->copy_to_cpu((DataType*)(send_buffer + sizeof(data_mode) + sizeof(v_size)), v_size);
  // timer2.stop();
  // timer.stop();
  // fprintf(stderr, "Get %u->%u: Time (ms): %llu + %llu = %llu\n",
  //  ctx->id, from_id,
  //  timer1.duration_ms(), timer2.duration_ms(),
  //  timer.duration_ms());
}

template <typename DataType>
void gpuDirectSend(struct CUDA_Context_Common* ctx, size_t bit_set_count,
                   size_t num_shared, DeviceOnly<DataType>* shared_data,
                   uint8_t* send_buffer, unsigned to_id,
                   DataCommMode data_mode) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  uint8_t* gpu_buffer, *gpu_buffer_pointer;
  size_t offset_size = ctx->offsets.size();
  size_t vec_size = ctx->is_updated.cpu_rd_ptr()->vec_size();
  // size info + offsets + size_info + bitset + size_info + data

  size_t buffer_size = sizeof(DataCommMode);

  //!
  //! Calculate bitset size.
  //!
  if (data_mode != onlyData) {
    buffer_size += sizeof(bit_set_count);
  }
  if ((data_mode == gidsData) || (data_mode == offsetsData)) {
    buffer_size += sizeof(bit_set_count);
    buffer_size += bit_set_count * sizeof(unsigned int);
  } else if ((data_mode == bitsetData)) {
    buffer_size += sizeof(num_shared);
    buffer_size += sizeof(vec_size);
    buffer_size += vec_size * sizeof(uint64_t);
  }
  buffer_size += sizeof(bit_set_count);
  //buffer_size += bit_set_count * sizeof(DataType); 
  buffer_size += bit_set_count * sizeof(DataType); 

  CUDA_SAFE_CALL(cudaHostAlloc(&gpu_buffer, buffer_size, 
                               cudaHostAllocPortable)); 

  //!
  //! Serialize Data.
  //!
  size_t offset = 0;
  // serialize data_mode
  cudaMemcpyAsync(gpu_buffer, &data_mode,
             sizeof(DataCommMode), cudaMemcpyDefault);
  offset += sizeof(DataCommMode);

  if (data_mode == noData) {
    check_cuda_kernel;
    MPI_Isend(gpu_buffer, sizeof(DataCommMode), MPI_BYTE, to_id,
        10000, MPI_COMM_WORLD,
        &send_req); 
    std::cout << "\t\t **** gpudirect send tries to send no-data\n";
    return ;
  }

  if (data_mode != onlyData) {
    cudaMemcpyAsync(gpu_buffer+offset, &bit_set_count,
        sizeof(bit_set_count), cudaMemcpyDefault);
    offset += sizeof(bit_set_count);
  }

  if ((data_mode == gidsData) || (data_mode == offsetsData)) {
    cudaMemcpyAsync(gpu_buffer+offset, &bit_set_count,
       sizeof(bit_set_count), cudaMemcpyDefault);
    offset += sizeof(bit_set_count);
    // OFFSET copy
    cudaMemcpyAsync(gpu_buffer + offset, ctx->offsets.device_ptr(), bit_set_count*sizeof(unsigned int),
               cudaMemcpyDefault);
    //serialize_offsets<<<blocks, threads>>>
    //  (gpu_buffer + offset, ctx->offsets.device_ptr(), bit_set_count);
    offset += sizeof(unsigned int)*bit_set_count;
  } else if ((data_mode == bitsetData)) {
    cudaMemcpyAsync(gpu_buffer+offset, &num_shared,
        sizeof(num_shared), cudaMemcpyDefault);
    offset += sizeof(num_shared);
    cudaMemcpyAsync(gpu_buffer+offset, &vec_size,
        sizeof(vec_size), cudaMemcpyDefault);
    offset += sizeof(vec_size);
    //cudaMemcpy(gpu_buffer+offset, ctx->is_updated.gpu_rd_ptr(),
    //           is_updated_size*sizeof(uint64_t), cudaMemcpyDefault);
    cudaMemcpyAsync(gpu_buffer+offset, ctx->is_updated.gpu_rd_ptr()->get_vec(),
               vec_size*sizeof(uint64_t), cudaMemcpyDefault);
    //ctx->is_updated.gpu_rd_ptr().copy_to_buffer(gpu_buffer + offset);

    //ctx->is_updated.gpu_rd_ptr().copy_to_gpu(gpu_buffer+offset);
    //check_cuda_kernel;
    //serialize_isupdate<<<blocks, threads>>>
    //serialize_isupdate<<<blocks, 1>>>
    //  (gpu_buffer+offset, ctx->is_updated.gpu_rd_ptr()->get_vec(),
    //   is_updated_size);
    offset += sizeof(uint64_t)*vec_size;
  }
  // data assign (uint32_t)
  cudaMemcpyAsync(gpu_buffer+offset, &bit_set_count,
             sizeof(bit_set_count), cudaMemcpyDefault);
  offset += sizeof(bit_set_count);
  cudaMemcpyAsync(gpu_buffer+offset, shared_data->device_ptr(),
             sizeof(DataType)*bit_set_count,
             cudaMemcpyDefault);
  //serialize_data<<<blocks, threads>>>
  //    (gpu_buffer + offset, shared_data->device_ptr(), bit_set_count);

  /*
  serialize_data<<<blocks, threads>>>(gpu_buffer,
                                      ctx->is_updated.gpu_rd_ptr(),
                                      is_updated_size, ctx->offsets.device_ptr(),
                                      offset_size,
                                      shared_data->device_ptr(), data_size);
                                      */
  printf("\tSend Request == \n");
  printf("\t\tTo ID: %d\n", to_id);
  printf("\t\tOffset size : %d\n", ctx->offsets.size());
  printf("\t\tIsupdated size: %d\n", ctx->is_updated.cpu_rd_ptr()->vec_size());
  printf("\t\tData size : %d\n", bit_set_count);
  printf("\t\tRequired buffer size: %d\n", buffer_size);
  printf("\n");

  check_cuda_kernel;
  //MPI_Send(gpu_buffer, buffer_size, MPI_BYTE, to_id, 10000, MPI_COMM_WORLD);

  MPI_Isend(gpu_buffer, buffer_size, MPI_BYTE, to_id, 10000, MPI_COMM_WORLD,
            &send_req);   
}

template <typename DataType, SharedType sharedType, bool reset>
void batch_get_shared_field(struct CUDA_Context_Common* ctx,
                            struct CUDA_Context_Field<DataType>* field,
                            unsigned from_id, uint8_t* send_buffer,
                            size_t* v_size, DataCommMode* data_mode,
                            DataType i = 0) {
  // Here.
  struct CUDA_Context_Shared* shared;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
  } else { // sharedMirror
    shared = &ctx->mirror;
  }

  //printf("\n From ID: %d------------------- \n", from_id);
  //printf("Master node: %d, mirror node: %d\n", *(ctx->master.num_nodes), *(ctx->mirror.num_nodes));
  DeviceOnly<DataType>* shared_data = &field->shared_data;
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  // calculate subset of bitset.
  //ctx->is_updated.cpu_rd_ptr()->resize(shared->num_nodes[from_id]);
  //ctx->is_updated.cpu_rd_ptr()->reset();
  batch_get_subset_bitset<<<blocks, threads>>>(
      shared->num_nodes[from_id], shared->nodes[from_id].device_ptr(),
      ctx->is_updated.gpu_rd_ptr(), field->is_updated.gpu_rd_ptr());
  check_cuda_kernel;
  // calculate offset of bitset.
  get_offsets_from_bitset(shared->num_nodes[from_id],
      ctx->offsets.device_ptr(),
      ctx->is_updated.gpu_rd_ptr(), v_size);
  *data_mode = get_data_mode<DataType>(*v_size, shared->num_nodes[from_id]);

  if (reset) {
    batch_get_reset_subset<DataType><<<blocks, threads>>>(
        *v_size, shared->nodes[from_id].device_ptr(),
        ctx->offsets.device_ptr(), shared_data->device_ptr(),
        field->data.gpu_wr_ptr(), i);
  } else {
    batch_get_subset<DataType><<<blocks, threads>>>(
        *v_size, shared->nodes[from_id].device_ptr(),
        ctx->offsets.device_ptr(), shared_data->device_ptr(),
        field->data.gpu_rd_ptr());
  }
  check_cuda_kernel;
  gpuDirectSend(ctx, *v_size, shared->num_nodes[from_id], shared_data, send_buffer, from_id, *data_mode);
}

template <typename DataType>
void gpuDirectRecv(struct CUDA_Context_Common* ctx,
                   DataCommMode &data_mode,
                   uint64_t &bit_set_count,
                   struct CUDA_Context_Shared* shared,
                   DeviceOnly<DataType>* shared_data,
                   unsigned &from_id) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  uint8_t* gpu_buffer;
  int flag = 0;
  uint64_t num_shared = shared->num_nodes[from_id];
  int num_received;

  MPI_Probe(from_id, 10000, MPI_COMM_WORLD, &recv_stat);
  MPI_Get_count(&recv_stat, MPI_BYTE, &num_received);

  CUDA_SAFE_CALL(cudaHostAlloc(&gpu_buffer, num_received, 
                               cudaHostAllocPortable)); 
  //*from_id = recv_stat.MPI_SOURCE;

  //MPI_Iprobe(MPI_ANY_SOURCE, 10000, MPI_COMM_WORLD, 
  //           &flag, &recv_stat);
  MPI_Recv(gpu_buffer, num_received, MPI_BYTE,
           MPI_ANY_SOURCE, 10000, MPI_COMM_WORLD, &recv_stat);

  //MPI_Irecv(gpu_buffer, num_received, MPI_BYTE,
  //          MPI_ANY_SOURCE, 10000, MPI_COMM_WORLD, &recv_req);
  //MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
  if (send_req != MPI_REQUEST_NULL) {
    //printf("id %d Here\n", ctx->id);
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
    send_req = MPI_REQUEST_NULL;
  }

  uint64_t vec_size, offset_size, data_size;
  uint64_t offset = 0;

  cudaMemcpyAsync(&data_mode, gpu_buffer, sizeof(DataCommMode),
             cudaMemcpyDefault);
  offset += sizeof(DataCommMode);

  if (data_mode == noData) {
    check_cuda_kernel;
    //cudaFree(gpu_buffer);
    return ;
  }

	//assert(data_mode != noData);
  if (data_mode != onlyData) {
    cudaMemcpyAsync(&bit_set_count, gpu_buffer + offset,
               sizeof(bit_set_count), cudaMemcpyDefault);
    offset += sizeof(bit_set_count);
  } else {
    bit_set_count = num_shared;
  }

  assert(data_mode != gidsData);
  if (data_mode == offsetsData) {
    offset += sizeof(bit_set_count);
    /*deserialize_offsets<<<blocks, threads>>>(
        gpu_buffer+offset, ctx->offsets.device_ptr(),
        bit_set_count);
        */
    cudaMemcpyAsync(ctx->offsets.device_ptr(),
               gpu_buffer+offset, bit_set_count*sizeof(unsigned int),
               cudaMemcpyDefault);
    offset += bit_set_count * sizeof(unsigned int);
  } else if ((data_mode == bitsetData)) {
    offset += sizeof(num_shared);
    uint64_t vec_size = ctx->is_updated.cpu_rd_ptr()->vec_size();
    offset += sizeof(vec_size);
    //deserialize_isupdate<<<blocks, threads>>>(
    //    gpu_buffer + offset, ctx->is_updated.gpu_wr_ptr()->get_vec(),
    //    vec_size);
    //cudaMemcpy(ctx->is_updated.gpu_wr_ptr(), gpu_buffer + offset,
    //          sizeof(uint64_t) * vec_size, cudaMemcpyDefault);

    cudaMemcpyAsync(ctx->is_updated.gpu_wr_ptr()->get_vec(), gpu_buffer + offset,
               sizeof(uint64_t) * vec_size, cudaMemcpyDefault);
    //ctx->is_updated.gpu_wr_ptr()->copy_from_buffer(gpu_buffer + offset);
    offset += sizeof(uint64_t) * vec_size;
    size_t v_size;
    get_offsets_from_bitset(num_shared,
                            ctx->offsets.device_ptr(),
                            ctx->is_updated.gpu_rd_ptr(), &v_size);
    assert(bit_set_count == v_size);
  }

  offset += sizeof(bit_set_count);
  cudaMemcpyAsync(shared_data->device_ptr(), gpu_buffer+offset,
             bit_set_count*sizeof(DataType), cudaMemcpyDefault);
  /*
  deserialize_data<<<blocks, threads>>>(
      gpu_buffer + offset, shared_data->device_ptr(),
      bit_set_count);
      */

  check_cuda_kernel;

  printf("\tReceive == \n");
  printf("\t\tThis ID: %d\n", ctx->id);
  printf("\t\tTo ID: %d\n", from_id);
  printf("\t\t# of received: %d\n", num_received);
  printf("\t\tOffset size : %d\n", ctx->offsets.size());
  printf("\t\tIsupdated size: %d\n", ctx->is_updated.gpu_rd_ptr()->vec_size());
  printf("\t\tData size : %d\n", bit_set_count);
  printf("\n");
}

template <typename DataType, SharedType sharedType, UpdateOp op>
void batch_set_shared_field(struct CUDA_Context_Common* ctx,
                            struct CUDA_Context_Field<DataType>* field,
                            unsigned from_id, uint8_t* recv_buffer,
                            DataCommMode data_mode) {
  struct CUDA_Context_Shared* shared;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
  } else { // sharedMirror
    shared = &ctx->mirror;
  }
  DeviceOnly<DataType>* shared_data = &field->shared_data;
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  // ggc::Timer timer("timer"), timer1("timer1"), timer2("timer2");
  // timer.start();
  // timer1.start();
  //deserializeMessage(ctx, data_mode, v_size, shared->num_nodes[from_id], shared_data, recv_buffer);
  //size_t v_size = gpuDirectRecv(ctx, shared, shared_data, &from_id);
  
  size_t v_size;
  gpuDirectRecv(ctx, data_mode, v_size, 
                shared, shared_data, from_id);
  if (data_mode == noData) { return ; }
  
  // timer1.stop();
  // timer2.start();
  if (data_mode == onlyData) {
    if (op == setOp) {
      batch_set_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->nodes[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_wr_ptr(),
          field->is_updated.gpu_wr_ptr());
    } else if (op == addOp) {
      batch_add_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->nodes[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_wr_ptr(),
          field->is_updated.gpu_wr_ptr());
    } else if (op == minOp) {
      batch_min_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->nodes[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_wr_ptr(),
          field->is_updated.gpu_wr_ptr());
    }
  } else if (data_mode == gidsData) {
    if (op == setOp) {
      batch_set_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == addOp) {
      batch_add_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == minOp) {
      batch_min_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    }
  } else { // bitsetData || offsetsData */
    if (op == setOp) {
      batch_set_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->nodes[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == addOp) {
      batch_add_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->nodes[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == minOp) {
      batch_min_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->nodes[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    }
  }
  check_cuda_kernel;
  // timer2.stop();
  // timer.stop();
  // fprintf(stderr, "Set %u<-%u: %d mode Time (ms): %llu + %llu = %llu\n",
  //  ctx->id, from_id, data_mode,
  //  timer1.duration_ms(), timer2.duration_ms(),
  //  timer.duration_ms());
}
