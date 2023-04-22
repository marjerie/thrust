/*
 *  Copyright 2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file 
 *  \brief Allocator types usable with \ref Memory Resources.
 */

#pragma once

#include <limits>

#include <thrust/detail/config.h>
#include <thrust/detail/config/exec_check_disable.h>
#include <thrust/detail/config/memory_resource.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <thrust/mr/validator.h>
#include <thrust/mr/polymorphic_adaptor.h>

// #define vm_array_size 10

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! \addtogroup allocators Allocators
 *  \ingroup memory_management
 *  \{
 */

/*! An \p mr::allocator is a template that fulfills the C++ requirements for Allocators,
 *  allowing to use the NPA-based memory resources where an Allocator is required. Unlike
 *  memory resources, but like other allocators, \p mr::allocator is typed and bound to
 *  allocate object of a specific type, however it can be freely rebound to other types.
 *
 *  \tparam T the type that will be allocated by this allocator.
 *  \tparam MR the upstream memory resource to use for memory allocation. Must derive from
 *      \p thrust::mr::memory_resource and must be \p final (in C++11 and beyond).
 */
template<typename T, class MR>
class allocator : private validator<MR>
{
public:
    /*! The pointer to void type of this allocator. */
    typedef typename MR::pointer void_pointer;

    /*! The value type allocated by this allocator. Equivalent to \p T. */
    typedef T value_type;
    /*! The pointer type allocated by this allocator. Equivaled to the pointer type of \p MR rebound to \p T. */
    typedef typename thrust::detail::pointer_traits<void_pointer>::template rebind<T>::other pointer;
    /*! The pointer to const type. Equivalent to a pointer type of \p MR rebound to <tt>const T</tt>. */
    typedef typename thrust::detail::pointer_traits<void_pointer>::template rebind<const T>::other const_pointer;
    /*! The reference to the type allocated by this allocator. Supports smart references. */
    typedef typename thrust::detail::pointer_traits<pointer>::reference reference;
    /*! The const reference to the type allocated by this allocator. Supports smart references. */
    typedef typename thrust::detail::pointer_traits<const_pointer>::reference const_reference;
    /*! The size type of this allocator. Always \p std::size_t. */
    typedef std::size_t size_type;
    /*! The difference type between pointers allocated by this allocator. */
    typedef typename thrust::detail::pointer_traits<pointer>::difference_type difference_type;

    /*! Specifies that the allocator shall be propagated on container copy assignment. */
    typedef detail::true_type propagate_on_container_copy_assignment;
    /*! Specifies that the allocator shall be propagated on container move assignment. */
    typedef detail::true_type propagate_on_container_move_assignment;
    /*! Specifies that the allocator shall be propagated on container swap. */
    typedef detail::true_type propagate_on_container_swap;

    /*! The \p rebind metafunction provides the type of an \p allocator instantiated with another type.
     *
     *  \tparam U the other type to use for instantiation.
     */
    template<typename U>
    struct rebind
    {
        /*! The typedef \p other gives the type of the rebound \p allocator.
         */
        typedef allocator<U, MR> other;
    };

    /*! Calculates the maximum number of elements allocated by this allocator.
     *
     *  \return the maximum value of \p std::size_t, divided by the size of \p T.
     */
    __thrust_exec_check_disable__
    __host__ __device__
    size_type max_size() const
    {
        return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    /*! Constructor.
     *
     *  \param resource the resource to be used to allocate raw memory.
     */
    __host__ __device__
    allocator(MR * resource) : mem_res(resource)
    {
    }

    /*! Copy constructor. Copies the resource pointer. */
    template<typename U>
    __host__ __device__
    allocator(const allocator<U, MR> & other) : mem_res(other.resource())
    {
    }

    /*! Allocates objects of type \p T.
     *
     *  \param n number of elements to allocate
     *  \return a pointer to the newly allocated storage.
     */
    THRUST_NODISCARD
    __host__
    pointer allocate(size_type n)
    {
        return static_cast<pointer>(mem_res->do_allocate(n * sizeof(T), THRUST_ALIGNOF(T)));
    }

    /*! Deallocates objects of type \p T.
     *
     *  \param p pointer returned by a previous call to \p allocate
     *  \param n number of elements, passed as an argument to the \p allocate call that produced \p p
     */
    __host__
    void deallocate(pointer p, size_type n)
    {
        return mem_res->do_deallocate(p, n * sizeof(T), THRUST_ALIGNOF(T));
    }

    /*! Extracts the memory resource used by this allocator.
     *
     *  \return the memory resource used by this allocator.
     */
    __host__ __device__
    MR * resource() const
    {
        return mem_res;
    }

private:
    MR * mem_res;
};

/*! Compares the allocators for equality by comparing the underlying memory resources. */
template<typename T, typename MR>
__host__ __device__
bool operator==(const allocator<T, MR> & lhs, const allocator<T, MR> & rhs) noexcept
{
    return *lhs.resource() == *rhs.resource();
}

/*! Compares the allocators for inequality by comparing the underlying memory resources. */
template<typename T, typename MR>
__host__ __device__
bool operator!=(const allocator<T, MR> & lhs, const allocator<T, MR> & rhs) noexcept
{
    return !(lhs == rhs);
}

#if THRUST_CPP_DIALECT >= 2011

template<typename T, typename Pointer>
using polymorphic_allocator = allocator<T, polymorphic_adaptor_resource<Pointer> >;

#else // C++11

template<typename T, typename Pointer>
class polymorphic_allocator : public allocator<T, polymorphic_adaptor_resource<Pointer> >
{
    typedef allocator<T, polymorphic_adaptor_resource<Pointer> > base;

public:
    /*! Initializes the base class with the parameter \p resource.
     */
    polymorphic_allocator(polymorphic_adaptor_resource<Pointer>  * resource) : base(resource)
    {
    }
};

#endif // C++11

/*! A helper allocator class that uses global instances of a given upstream memory resource. Requires the memory resource
 *      to be default constructible.
 *
 *  \tparam T the type that will be allocated by this allocator.
 *  \tparam Upstream the upstream memory resource to use for memory allocation. Must derive from
 *      \p thrust::mr::memory_resource and must be \p final (in C++11 and beyond).
 */
template<typename T, typename Upstream>
class stateless_resource_allocator : public thrust::mr::allocator<T, Upstream>
{
    typedef thrust::mr::allocator<T, Upstream> base;

public:
    /*! The \p rebind metafunction provides the type of an \p stateless_resource_allocator instantiated with another type.
     *
     *  \tparam U the other type to use for instantiation.
     */
    template<typename U>
    struct rebind
    {
        /*! The typedef \p other gives the type of the rebound \p stateless_resource_allocator.
         */
        typedef stateless_resource_allocator<U, Upstream> other;
    };

    /*! Default constructor. Uses \p get_global_resource to get the global instance of \p Upstream and initializes the
     *      \p allocator base subobject with that resource.
     */
    __thrust_exec_check_disable__
    __host__ __device__
    stateless_resource_allocator() : base(get_global_resource<Upstream>())
    {
    }

    /*! Copy constructor. Copies the memory resource pointer. */
    __host__ __device__
    stateless_resource_allocator(const stateless_resource_allocator & other)
        : base(other) {}

    /*! Conversion constructor from an allocator of a different type. Copies the memory resource pointer. */
    template<typename U>
    __host__ __device__
    stateless_resource_allocator(const stateless_resource_allocator<U, Upstream> & other)
        : base(other) {}

#if THRUST_CPP_DIALECT >= 2011
    stateless_resource_allocator & operator=(const stateless_resource_allocator &) = default;
#endif

    /*! Destructor. */
    __host__ __device__
    ~stateless_resource_allocator() {}
};

/*! \} // allocators
 */

template<typename T, typename Upstream>
class virtual_memory_resource_allocator : public thrust::mr::allocator<T, Upstream>
{
    typedef thrust::mr::allocator<T, Upstream> base;

    public:
    //typedef typename virtual_memory_resource_allocator::pointer void_pointer;
    //typedef typename thrust::detail::pointer_traits<void_pointer>::template rebind<T>::other pointer;

    //typedef typename base::pointer pointer;

    typedef typename thrust::cuda::pointer<void> Pointer;


        struct Range {
            CUdeviceptr start;
            size_t sz;
        };

        CUmemAllocationProp prop;
        CUmemAccessDesc accessDesc;
        std::vector<Range> va_ranges;
        std::vector<CUmemGenericAllocationHandle> handles;
        std::vector<size_t> handle_sizes;

        // Range va_ranges[vm_array_size];
        // CUmemGenericAllocationHandle handles[vm_array_size];
        // size_t handle_sizes[vm_array_size];
        // thrust::host_vector<Range> va_ranges;
        // thrust::host_vector<CUmemGenericAllocationHandle> handles;
        // thrust::host_vector<size_t> handle_sizes;
        CUdeviceptr d_p;
        size_t count;
        size_t chunk_sz;
        size_t index_va_ranges, index_handles;
        size_t alloc_sz;
        size_t reserve_sz;

        template<typename U>
        struct rebind
        {
            typedef virtual_memory_resource_allocator<U, Upstream> other;
        };

        __host__
        // virtual_memory_resource_allocator() : base(get_global_resource<Upstream>())
        virtual_memory_resource_allocator() : base(new Upstream)
        {
            //base = get_global_resource<Upstream>;
            // std::cout << "mem_res: " << this->mem_res << '\n';
            d_p = 0ULL;
            count = 0;
            chunk_sz = 0;
            // index_va_ranges = 0;
            // index_handles = 0;
            va_ranges.clear();
            handles.clear();
            handle_sizes.clear();
            alloc_sz = 0;
            reserve_sz = 0;
            // CUresult status = cuInit(0);
            cuInit(0);
            // printf("cuInit ==== %d\n", status);

            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            int device_id;
            cudaError_t status_new = cudaGetDevice(&device_id);
            cudaGetDevice(&device_id);
            // printf("cudaGetDevice ==== %d device_id = %d\n", status_new, device_id);
            prop.location.id = device_id;

            accessDesc.location = prop.location;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            // chunk_sz = chunk_sz * 2;
            // status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            // std::cout << "chunk size: " << chunk_sz << '\n';

        }

        /*! Copy constructor. Copies the memory resource pointer. */
        __host__ __device__
        virtual_memory_resource_allocator(const virtual_memory_resource_allocator & other)
            : base(other) {
                d_p = 0ULL;
                count = 0;
                // printf( " d_p is %llu and other.d_p is %llu \n", d_p, other.d_p);
                //chunk_sz = 0;
                index_va_ranges = 0;
                index_handles = 0;

                va_ranges.clear();
                handles.clear();
                handle_sizes.clear();

                //alloc_sz = 0;
                //reserve_sz = 0;
                //d_p = other.d_p;
                count = other.count;
                chunk_sz = other.chunk_sz;
                //index_va_ranges = other.index_va_ranges;
                //index_handles = other.index_handles;
                alloc_sz = other.alloc_sz;
                reserve_sz = other.reserve_sz;
                //Pointer ptr = allocate(0);
                // handles = other.handles;
                // handle_sizes = other.handle_sizes;
                // va_ranges = other.va_ranges;
            }

        /*! Conversion constructor from an allocator of a different type. Copies the memory resource pointer. */
        template<typename U>
        __host__ __device__
        virtual_memory_resource_allocator(const virtual_memory_resource_allocator<U, Upstream> & other)
            : base(other) {}

    #if THRUST_CPP_DIALECT >= 2011
        virtual_memory_resource_allocator & operator=(const virtual_memory_resource_allocator &) = default;
    #endif


        THRUST_NODISCARD
        __host__
        typename allocator<T,Upstream>::pointer allocate(size_t n)
        // pointer allocate(size_t n)
        {            
            CUmemGenericAllocationHandle handle;
            size_t size_diff = n * sizeof(T);
            size_t sz = ((size_diff + chunk_sz - 1) / chunk_sz) * chunk_sz;

            CUresult status = reserve(alloc_sz + sz);

            CUmemAllocationProp prop_new = {};
            CUmemAccessDesc accessDesc_new = {};

            prop_new.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop_new.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            int device_id;
            cudaError_t status_new = cudaGetDevice(&device_id);
            // printf("cudaGetDevice ==== %d device_id = %d\n", status_new, device_id);
            prop_new.location.id = device_id;

            accessDesc_new.location = prop_new.location;
            accessDesc_new.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            if (status == CUDA_SUCCESS) {
                // cuMemCreate(&handle, sz, &prop, 0);
                // cuMemMap(d_p + alloc_sz, sz, 0ULL, handle, 0ULL);
                // cuMemSetAccess(d_p + alloc_sz, sz, &accessDesc, 1ULL);
                if ((status = cuMemCreate(&handle, sz, &prop_new, 0ULL)) == CUDA_SUCCESS) {
                    // std::cout << "cuMemCreate success " << status << '\n';
                    if ((status = cuMemMap(d_p + alloc_sz, sz, 0ULL, handle, 0ULL)) == CUDA_SUCCESS) {
                        // std::cout << "cuMemMap success " << status << '\n';
                        if ((status = cuMemSetAccess(d_p + alloc_sz, sz, &accessDesc_new, 1ULL)) == CUDA_SUCCESS) {
                            // std::cout << "cumemsetaccess success" << '\n';
                            // update_handles(handle, sz);
                            handles.push_back(handle);
                            handle_sizes.push_back(sz);
                            alloc_sz += sz;
                        }
                        if (status != CUDA_SUCCESS) {
                            // std::cout << "cumemsetaccess failed..." << '\n';
                            (void)cuMemUnmap(d_p + alloc_sz, sz);
                            // std::cout << "cuMemUnmap " << cuMemUnmap(d_p + alloc_sz, sz) << '\n';
                        }
                    }
                    if (status != CUDA_SUCCESS) {
                        // std::cout << "cuMemMap failed... " << status << '\n';
                        (void)cuMemRelease(handle);
                        // std::cout << "cuMemRelease " << cuMemRelease(handle) << '\n';
                    }    
                }
            //     else
            //     {
            //         std::cout << "cuMemCreate failed... " << status << '\n';
            //     }
            }
            // else
            // {
            //     std::cout << "reserve failed... " << status << '\n';
            // }

            count = alloc_sz/sizeof(T);

            // std::cout << "d_p is " << std::hex << d_p << std::dec << '\n';
            // std::cout << "pointer is d_p: " << d_p << '\n';
            // std::cout << "vm allocated " << n << " elements" << '\n';

            return static_cast<typename allocator<T,Upstream>::pointer>(Pointer((void *)d_p));
            //return static_cast<pointer>(mem_res->do_allocate(n * sizeof(T), THRUST_ALIGNOF(T)));
        }

        __host__
        CUresult reserve(size_t new_sz)
        {
            CUresult status = CUDA_SUCCESS;
            CUdeviceptr new_ptr = 0ULL;

            CUmemAllocationProp prop_new = {};
            CUmemAccessDesc accessDesc_new = {};

            prop_new.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop_new.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            int device_id;
            cudaError_t status_new = cudaGetDevice(&device_id);
            // printf("cudaGetDevice ==== %d device_id = %d\n", status_new, device_id);
            prop_new.location.id = device_id;

            accessDesc_new.location = prop_new.location;
            accessDesc_new.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            size_t aligned_sz = ((new_sz + chunk_sz - 1) / chunk_sz) * chunk_sz;

            status = cuMemAddressReserve(&new_ptr, (aligned_sz - reserve_sz), 0ULL, d_p + reserve_sz, 0ULL);
            // std::cout << "cuMemAddressReserve ===== " << status << '\n';

            if (status != CUDA_SUCCESS || (new_ptr != d_p + reserve_sz))
            {
                if (new_ptr != 0ULL)
                {
                    status = cuMemAddressFree(new_ptr, (aligned_sz - reserve_sz));
                    // std::cout << "cuMemAddressFree ===== " << status << '\n';
                }

                status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, 0U, 0);
                // std::cout << "cuMemAddressReserve ===== " << status << '\n';

                if (status == CUDA_SUCCESS && d_p != 0ULL)
                {
                    CUdeviceptr ptr = new_ptr;
                    status = cuMemUnmap(d_p, alloc_sz);
                    // std::cout << "cuMemUnmap ===== " << status << '\n';

                    for (size_t i = 0ULL; i < handles.size(); i++) {
                        const size_t hdl_sz = handle_sizes[i];
                        if ((status = cuMemMap(ptr, hdl_sz, 0ULL, handles[i], 0ULL)) != CUDA_SUCCESS)
                            break;
                        // else std::cout << "cuMemMap ===== " << status << '\n';
                        if ((status = cuMemSetAccess(ptr, hdl_sz, &accessDesc_new, 1ULL)) != CUDA_SUCCESS)
                            break;
                        // else std::cout << "cuMemSetAccess ===== " << status << '\n';
                        ptr += hdl_sz;
                    }
                    if (status != CUDA_SUCCESS) {
                        status = cuMemUnmap(new_ptr, aligned_sz);
                        // std::cout << "cuMemUnmap ===== " << status << '\n';
                        assert(status == CUDA_SUCCESS);
                        status = cuMemAddressFree(new_ptr, aligned_sz);
                        // std::cout << "cuMemAddressFree ===== " << status << '\n';
                        assert(status == CUDA_SUCCESS);
                    }
                    else {
                        for (size_t i = 0ULL; i < va_ranges.size(); i++) {
                            status = cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
                            // std::cout << "cuMemAddressFree ===== " << status << '\n';
                        }
                        // index_va_ranges = 0;
                        va_ranges.clear();
                    }
                }
                if (status == CUDA_SUCCESS) {
                    Range r;
                    d_p = new_ptr;
                    reserve_sz = aligned_sz;
                    r.start = new_ptr;
                    r.sz = aligned_sz;
                    va_ranges.push_back(r);
                    // update_va_ranges(new_ptr, aligned_sz);
                }
            }
            else
            {
                Range r;
                r.start = new_ptr;
                r.sz = aligned_sz - reserve_sz;
                va_ranges.push_back(r);
                // update_va_ranges(new_ptr, aligned_sz - reserve_sz);
                if (d_p == 0ULL) {
                    d_p = new_ptr;
                }
                reserve_sz = aligned_sz;
            }
            return status;
        }

        __host__ __device__
        void update_size (size_t n)
        {
            count+=n;
        }

        __host__
        void update_handles(CUmemGenericAllocationHandle handle, size_t sz)
        {
            handles[index_handles] = handle;
            handle_sizes[index_handles++] = sz;
            // std::cout << "handle: " << std::hex << handles[index_handles - 1] << std::dec << "handle size: " << handle_sizes[index_handles - 1] << '\n'; 
        }

        __host__
        void update_va_ranges(CUdeviceptr ptr, size_t range_sz)
        {
            Range r;
            //r.start = (CUdeviceptr) ptr;
            r.start = ptr;
            r.sz = range_sz;
            // std::cout << "va range " << '\n'; 
            // std::cout << "ptr start: " << std::hex << ptr << std::dec << "size: " << range_sz << '\n'; 
            va_ranges[index_va_ranges++] = r;
        }

        __host__
        void deallocate(typename allocator<T,Upstream>::pointer p, size_t n)
        {
            // std::cout << "hello from deallocate from allocator.h" << '\n';
            // std::cout << "d_p is " << std::hex << d_p << std::dec << '\n';
            // std::cout << "p.get() is " << p.get() << '\n';
            // std::cout << "size (n) is " << n << '\n';
            // std::cout << "count is " << count << '\n';
            // for (size_t i = 0ULL; i < index_va_ranges; i++) {
            //     std::cout << "va range: "<< std::hex << va_ranges[i].start << std::dec << " size: " << va_ranges[i].sz << '\n';
            // }
            // for (size_t i = 0ULL; i < index_handles; i++) {
            //     std::cout << "handles: "<< std::hex << handles[i] << std::dec << '\n';
            // }
            CUresult status = CUDA_SUCCESS;
            (void)status;
            if (d_p != 0ULL) {
                // status = cuMemUnmap(d_p, alloc_sz);
                status = cuMemUnmap((CUdeviceptr) p.get(), n*sizeof(T));
                // std::cout << "cuMemUnmap ===== " << status << '\n';
                assert(status == CUDA_SUCCESS);
                for (size_t i = 0ULL; i < va_ranges.size(); i++) {
                    status = cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
                    // std::cout << "cuMemAddressFree ===== " << status << '\n';
                    assert(status == CUDA_SUCCESS);
                }
                va_ranges.clear();
                for (size_t i = 0ULL; i < handles.size(); i++) {
                    status = cuMemRelease(handles[i]);
                    // std::cout << "cuMemRelease ===== " << status << '\n';
                    assert(status == CUDA_SUCCESS);
                }
                handles.clear();
                handle_sizes.clear();
            }
            // return mem_res->do_deallocate(p, n * sizeof(T), THRUST_ALIGNOF(T));
        }

        /*! Destructor. */
        // removed __device__
        __host__ __device__
        ~virtual_memory_resource_allocator() {
            // delete this->mem_res;
            // this->mem_res = nullptr;
            // printf("destructor called\n");
        }
};

} // end mr
THRUST_NAMESPACE_END

