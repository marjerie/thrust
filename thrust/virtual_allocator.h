/*
 *  Copyright 2008-2020 NVIDIA Corporation
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


/*! \file universal_allocator.h
 *  \brief An allocator which creates new elements in memory accessible to both
 *         hosts and devices.
 */

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

/*! \}
 */

THRUST_NAMESPACE_END
