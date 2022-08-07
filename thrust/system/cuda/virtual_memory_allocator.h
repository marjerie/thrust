// added here
//extern "C" {
#pragma once

#include <new>
#include <string>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

#include <thrust/detail/config.h>

THRUST_NAMESPACE_BEGIN

namespace system
{
namespace cuda
{

namespace detail
{

// define our own bad_alloc so we can set its .what()
// template<typename T>
class virtual_memory_allocator
{
    public:
    cudaError_t allocate(void ** ptr, std::size_t bytes)
    {
      CUresult status = CUDA_SUCCESS;

      //status = cuInit(0);
      //printf("cuInit ==== %d\n", status);
      //CUdeviceptr d_p;

      CUmemAllocationProp prop = {};
  
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      int device_id;
      cudaGetDevice(&device_id);
      prop.location.id = device_id;
      //prop.location.id = 0;  // update correct device later
      //prop.win32HandleMetaData = NULL;

      CUmemAccessDesc accessDesc;

      accessDesc.location = prop.location;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

      // std::size_t chunk_sz = 0ULL;

      // status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      //printf("cuMemGetAllocationGranularity ==== %d\n", status);

      CUmemGenericAllocationHandle handle;

      // const std::size_t sz = ((bytes + chunk_sz - 1) / chunk_sz) * chunk_sz;
      
      CUdeviceptr addr = 0ULL;

      status = cuMemAddressReserve((CUdeviceptr*)ptr, bytes, 0ULL, addr, 0ULL);
      //printf("cuMemAddressReserve ==== %d\n", status);

      status = cuMemCreate(&handle, bytes, &prop, 0ULL);
      //printf("cuMemCreate ==== %d\n", status);

      status = cuMemMap((CUdeviceptr)*ptr, bytes, 0ULL, handle, 0ULL);
      //printf("cuMemMap ==== %d\n", status);

      status = cuMemSetAccess((CUdeviceptr)*ptr, bytes, &accessDesc, 1ULL);
      //printf("cuMemSetAccess ==== %d\n", status);

      //*ptr = (void *)&d_p;

      std::cout <<"ptr is " << ptr << '\n';

      if (status != CUDA_SUCCESS) printf("failed virtual memory allocation\n");

      return cudaSuccess;
    }

    cudaError_t deallocate(void * ptr, std::size_t bytes)
    {
      CUresult status = CUDA_SUCCESS;
      //status = cuInit(0);
      //printf("cuInit ==== %d\n", status);

      std::cout <<"ptr is " << ptr << '\n';

      status = cuMemUnmap((CUdeviceptr)ptr, bytes);
      printf("cuMemUnmap ==== %d\n", status);

      if (status != CUDA_SUCCESS)
      {
        printf("failed virtual memory free\n");
        return cudaErrorInvalidValue;
      }

      return cudaSuccess;

    }
};

//   private:
//     //CUdeviceptr d_p;
//     //T* a = nullptr;
//     CUmemAllocationProp prop;
//     CUmemAccessDesc accessDesc;
//     struct Range {
//         CUdeviceptr start;
//         size_t sz;
//     };
//     std::vector<Range> va_ranges;
//     std::vector<CUmemGenericAllocationHandle> handles;
//     std::vector<size_t> handle_sizes;
//     size_t alloc_sz;
//     size_t reserve_sz;
//     size_t chunk_sz;
//     size_t d_size = 0;

//   public:
//     //cudaError_t allocate(void ** ptr, std::size_t bytes)
//     __host__
//     cudaError_t allocate(void ** d_ptr, std::size_t new_sz) 
//     {
//       cuInit(0);
//       CUdeviceptr d_p = 0ULL;
//       printf("here at virtual memory allocator!\n");
//       CUresult status = CUDA_SUCCESS;
//       CUdeviceptr new_ptr = 0ULL;

//       ///////////INITIALIZATION/////////////////////////

//       prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
//       prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
//       //prop.location.id = (int)device;
//       prop.location.id = 0;
//       prop.win32HandleMetaData = NULL;

//       accessDesc.location = prop.location;
//       accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

//       status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
//       printf("cuMemGetAllocationGranularity ==== %d\n", status);
//       assert(chunk_sz != 0);

//       //////////////END OF INITIALIZATION////////////////////

//       const size_t aligned_sz = ((new_sz + chunk_sz - 1) / chunk_sz) * chunk_sz;

//       status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, d_p, 0ULL);
//       printf("cuMemAddressReserve ==== %d\n", status);
      
//       // Try to reserve an address just after what we already have reserved
//       if (status != CUDA_SUCCESS || (new_ptr != d_p)) {
//           if (new_ptr != 0ULL) {
//               (void)cuMemAddressFree(new_ptr, (aligned_sz));
//           }
//           // Slow path - try to find a new address reservation big enough for us
//           status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, 0U, 0);
//           printf("cuMemAddressReserve ==== %d\n", status);

//           if (status == CUDA_SUCCESS && d_p != 0ULL) {
//               CUdeviceptr ptr = new_ptr;
//               // Found one, now unmap our previous allocations
//               status = cuMemUnmap(d_p, alloc_sz);
//               printf("cuMemUnmap ==== %d\n", status);
//               assert(status == CUDA_SUCCESS);
//               for (size_t i = 0ULL; i < handles.size(); i++) {
//                   const size_t hdl_sz = handle_sizes[i];
//                   // And remap them, enabling their access
//                   if ((status = cuMemMap(ptr, hdl_sz, 0ULL, handles[i], 0ULL)) != 0)
//                       break;
//                   if ((status = cuMemSetAccess(ptr, hdl_sz, &accessDesc, 1ULL)) != 0)
//                       break;
//                   ptr += hdl_sz;
//               }
//               if (status != CUDA_SUCCESS) {
//                   // Failed the mapping somehow... clean up!
//                   status = cuMemUnmap(new_ptr, aligned_sz);
//                   printf("cuMemUnmap ==== %d\n", status);
//                   assert(status == CUDA_SUCCESS);
//                   status = cuMemAddressFree(new_ptr, aligned_sz);
//                   printf("cuMemAddressFree ==== %d\n", status);
//                   assert(status == CUDA_SUCCESS);
//               }
//               else {
//                   // Clean up our old VA reservations!
//                   for (size_t i = 0ULL; i < va_ranges.size(); i++) {
//                       (void)cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
//                   }
//                   va_ranges.clear();
//               }
//           }
//           // Assuming everything went well, update everything
//           if (status == CUDA_SUCCESS) {
//               printf("reserve success\n");
//               Range r;
//               d_p = new_ptr;
//               *d_ptr = (void *)d_p;
//               reserve_sz = aligned_sz;
//               r.start = new_ptr;
//               r.sz = aligned_sz;
//               va_ranges.push_back(r);
//           }
//       }
//       else {
//       //std::cout << "Addressed reserved so now update values !! " << status << std::endl;
//           printf("reserve success\n");
//           Range r;
//           r.start = new_ptr;
//           r.sz = aligned_sz;
//           va_ranges.push_back(r);
//           if (d_p == 0ULL) {
//               d_p = new_ptr;
//               *d_ptr = (void *)d_p;
//           }
//           //reserve_sz = aligned_sz;
//       }
//       return cudaSuccess;
//     }

// };


//   public:
//     cudaError_t allocate(void ** ptr, std::size_t bytes)
//     {
//       printf("hello from grow\n");
//         return cudaMalloc(ptr, bytes);
//     }
// }; 
  
} // end detail
} // end cuda
} // end system
THRUST_NAMESPACE_END
//}
