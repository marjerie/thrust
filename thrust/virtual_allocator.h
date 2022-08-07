// added here

#pragma once

#include <thrust/detail/config.h>

// #include the device system's vector header
#define __THRUST_DEVICE_SYSTEM_MEMORY_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/memory.h>
#include __THRUST_DEVICE_SYSTEM_MEMORY_HEADER
#undef __THRUST_DEVICE_SYSTEM_MEMORY_HEADER

THRUST_NAMESPACE_BEGIN

/** \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management_classes
 *  \{
 */

/*! \brief An allocator which creates new elements in memory accessible by
 *         both hosts and devices.
 *
 *  \see https://en.cppreference.com/w/cpp/named_req/Allocator
 */
using thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::virtual_allocator;

/*! \p universal_ptr stores a pointer to an object allocated in memory accessible
 *  to both hosts and devices.
 *
 *  Algorithms dispatched with this type of pointer will be dispatched to
 *  either host or device, depending on which backend you are using. Explicit
 *  policies (\p thrust::device, etc) can be used to specify where an algorithm
 *  should be run.
 *
 *  \p universal_ptr has pointer semantics: it may be dereferenced safely from
 *  both hosts and devices and may be manipulated with pointer arithmetic.
 *
 *  \p universal_ptr can be created with \p universal_allocator or by explicitly
 *  calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p universal_ptr may be obtained by
 *  either its <tt>get</tt> method or the \p raw_pointer_cast free function.
 *
 *  \note \p universal_ptr is not a smart pointer; it is the programmer's
 *  responsibility to deallocate memory pointed to by \p universal_ptr.
 *
 *  \see host_ptr For the documentation of the complete interface which is
 *                shared by \p universal_ptr.
 *  \see raw_pointer_cast
 */
template <typename T>
using virtual_ptr =
  thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::virtual_pointer<T>;

/*! \}
 */

THRUST_NAMESPACE_END
