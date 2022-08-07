// added here

#pragma once

#include <thrust/detail/config.h>
#include <thrust/virtual_allocator.h>

// #include the device system's vector header
#define __THRUST_DEVICE_SYSTEM_VECTOR_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/vector.h>
#include __THRUST_DEVICE_SYSTEM_VECTOR_HEADER
#undef __THRUST_DEVICE_SYSTEM_VECTOR_HEADER

THRUST_NAMESPACE_BEGIN

/** \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management_classes
 *  \{
 */

/*! A \p universal_vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p universal_vector may vary dynamically; memory management is
 *  automatic. The memory associated with a \p universal_vector resides in memory
 *  accessible to hosts and devices.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p universal_vector.
 *  \see device_vector
 */
using thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::virtual_vector;

/*! \}
 */

THRUST_NAMESPACE_END
