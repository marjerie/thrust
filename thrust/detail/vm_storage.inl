/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/vm_storage.h>
#include <thrust/detail/swap.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/allocator/copy_construct_range.h>
#include <thrust/detail/allocator/default_construct_range.h>
#include <thrust/detail/allocator/destroy_range.h>
#include <thrust/detail/allocator/fill_construct_range.h>

#include <nv/target>

// #include <thrust/system/cuda/memory.h>

#include <stdexcept> // for std::runtime_error
#include <utility> // for use of std::swap in the WAR below

THRUST_NAMESPACE_BEGIN
// class virutal_allocator;
namespace detail
{

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  vm_storage<T,Alloc>
    ::vm_storage(const Alloc &alloc)
      :m_allocator(alloc),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
  ;
} // end vm_storage::vm_storage()

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  vm_storage<T,Alloc>
    ::vm_storage(size_type n, const Alloc &alloc)
      :m_allocator(alloc),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
  allocate(n);
} // end vm_storage::vm_storage()

template<typename T, typename Alloc>
__host__ __device__
  vm_storage<T,Alloc>
    ::vm_storage(vm_copy_allocator_t,
        const vm_storage &other)
      :m_allocator(other.m_allocator),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
} // end vm_storage::vm_storage()

template<typename T, typename Alloc>
__host__ __device__
  vm_storage<T,Alloc>
    ::vm_storage(vm_copy_allocator_t,
        const vm_storage &other, size_type n)
      :m_allocator(other.m_allocator),
       m_begin(pointer(static_cast<T*>(0))),
       m_size(0)
{
  allocate(n);
} // end vm_storage::vm_storage()

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  vm_storage<T,Alloc>
    ::~vm_storage()
{
  deallocate();
} // end vm_storage::~vm_storage()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::size_type
    vm_storage<T,Alloc>
      ::size() const
{
  return m_size;
} // end vm_storage::size()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::size_type
    vm_storage<T,Alloc>
      ::max_size() const
{
  return alloc_traits::max_size(m_allocator);
} // end vm_storage::max_size()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::iterator
    vm_storage<T,Alloc>
      ::begin()
{
  return m_begin;
} // end vm_storage::begin()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::const_iterator
    vm_storage<T,Alloc>
      ::begin() const
{
  return m_begin;
} // end vm_storage::begin()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::iterator
    vm_storage<T,Alloc>
      ::end()
{
  return m_begin + size();
} // end vm_storage::end()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::const_iterator
    vm_storage<T,Alloc>
      ::end() const
{
  return m_begin + size();
} // end vm_storage::end()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::pointer
    vm_storage<T,Alloc>
      ::data()
{
  return &*m_begin;
} // end vm_storage::data()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::const_pointer
    vm_storage<T,Alloc>
      ::data() const
{
  return &*m_begin;
} // end vm_storage::data()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::reference
    vm_storage<T,Alloc>
      ::operator[](size_type n)
{
  return m_begin[n];
} // end vm_storage::operator[]()

template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::const_reference
    vm_storage<T,Alloc>
      ::operator[](size_type n) const
{
  return m_begin[n];
} // end vm_storage::operator[]()

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  typename vm_storage<T,Alloc>::allocator_type
    vm_storage<T,Alloc>
      ::get_allocator() const
{
  return m_allocator;
} // end vm_storage::get_allocator()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::allocate(size_type n)
{
  if(n > 0)
  {
    pointer ptr = alloc_traits::allocate(m_allocator,n);
    m_begin = iterator(ptr);
    m_size = m_allocator.count;
  } // end if
  else
  {
    m_begin = iterator(pointer(static_cast<T*>(0)));
    m_size = 0;
  } // end else
} // end vm_storage::allocate()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::deallocate()
{
  if(size() > 0)
  {
    alloc_traits::deallocate(m_allocator,m_begin.base(), size());
    m_begin = iterator(pointer(static_cast<T*>(0)));
    m_size = 0;
  } // end if
} // end vm_storage::deallocate()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::swap(vm_storage &x)
{
  thrust::swap(m_begin, x.m_begin);
  thrust::swap(m_size, x.m_size);

  swap_allocators(
    integral_constant<
      bool,
      allocator_traits<Alloc>::propagate_on_container_swap::value
    >(),
    x.m_allocator);

  thrust::swap(m_allocator, x.m_allocator);
} // end vm_storage::swap()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::default_construct_n(iterator first, size_type n)
{
  default_construct_range(m_allocator, first.base(), n);
} // end vm_storage::default_construct_n()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::uninitialized_fill_n(iterator first, size_type n, const value_type &x)
{
  fill_construct_range(m_allocator, first.base(), n, x);
} // end vm_storage::uninitialized_fill()

template<typename T, typename Alloc>
  template<typename System, typename InputIterator>
  __host__ __device__
    typename vm_storage<T,Alloc>::iterator
      vm_storage<T,Alloc>
        ::uninitialized_copy(thrust::execution_policy<System> &from_system, InputIterator first, InputIterator last, iterator result)
{
  return iterator(copy_construct_range(from_system, m_allocator, first, last, result.base()));
} // end vm_storage::uninitialized_copy()

template<typename T, typename Alloc>
  template<typename InputIterator>
  __host__ __device__
    typename vm_storage<T,Alloc>::iterator
      vm_storage<T,Alloc>
        ::uninitialized_copy(InputIterator first, InputIterator last, iterator result)
{
  // XXX assumes InputIterator's associated System is default-constructible
  typename thrust::iterator_system<InputIterator>::type from_system;

  return iterator(copy_construct_range(from_system, m_allocator, first, last, result.base()));
} // end vm_storage::uninitialized_copy()

template<typename T, typename Alloc>
  template<typename System, typename InputIterator, typename Size>
  __host__ __device__
    typename vm_storage<T,Alloc>::iterator
      vm_storage<T,Alloc>
        ::uninitialized_copy_n(thrust::execution_policy<System> &from_system, InputIterator first, Size n, iterator result)
{
  return iterator(copy_construct_range_n(from_system, m_allocator, first, n, result.base()));
} // end vm_storage::uninitialized_copy_n()

template<typename T, typename Alloc>
  template<typename InputIterator, typename Size>
  __host__ __device__
    typename vm_storage<T,Alloc>::iterator
      vm_storage<T,Alloc>
        ::uninitialized_copy_n(InputIterator first, Size n, iterator result)
{
  // XXX assumes InputIterator's associated System is default-constructible
  typename thrust::iterator_system<InputIterator>::type from_system;

  return iterator(copy_construct_range_n(from_system, m_allocator, first, n, result.base()));
} // end vm_storage::uninitialized_copy_n()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::destroy(iterator first, iterator last)
{
  destroy_range(m_allocator, first.base(), last - first);
} // end vm_storage::destroy()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::deallocate_on_allocator_mismatch(const vm_storage &other)
{
  integral_constant<
    bool,
    allocator_traits<Alloc>::propagate_on_container_copy_assignment::value
  > c;

  deallocate_on_allocator_mismatch_dispatch(c, other);
} // end vm_storage::deallocate_on_allocator_mismatch

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::destroy_on_allocator_mismatch(const vm_storage &other,
        iterator first, iterator last)
{
  integral_constant<
    bool,
    allocator_traits<Alloc>::propagate_on_container_copy_assignment::value
  > c;

  destroy_on_allocator_mismatch_dispatch(c, other, first, last);
} // end vm_storage::destroy_on_allocator_mismatch

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::set_allocator(const Alloc &alloc)
{
  m_allocator = alloc;
} // end vm_storage::set_allocator()

template<typename T, typename Alloc>
__host__ __device__
  bool vm_storage<T,Alloc>
    ::is_allocator_not_equal(const Alloc &alloc) const
{
  return is_allocator_not_equal_dispatch(
    integral_constant<
      bool,
      allocator_traits<Alloc>::is_always_equal::value
    >(),
    alloc);
} // end vm_storage::is_allocator_not_equal()

template<typename T, typename Alloc>
__host__ __device__
  bool vm_storage<T,Alloc>
    ::is_allocator_not_equal(const vm_storage<T,Alloc> &other) const
{
  return is_allocator_not_equal(m_allocator, other.m_allocator);
} // end vm_storage::is_allocator_not_equal()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::propagate_allocator(const vm_storage &other)
{
  integral_constant<
    bool,
    allocator_traits<Alloc>::propagate_on_container_copy_assignment::value
  > c;

  propagate_allocator_dispatch(c, other);
} // end vm_storage::propagate_allocator()

#if THRUST_CPP_DIALECT >= 2011
template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::propagate_allocator(vm_storage &other)
{
  integral_constant<
    bool,
    allocator_traits<Alloc>::propagate_on_container_move_assignment::value
  > c;

  propagate_allocator_dispatch(c, other);
} // end vm_storage::propagate_allocator()

template<typename T, typename Alloc>
__host__ __device__
  vm_storage<T,Alloc> &vm_storage<T,Alloc>
    ::operator=(vm_storage &&other)
{
  if (size() > 0)
  {
    deallocate();
  }
  propagate_allocator(other);
  m_begin = std::move(other.m_begin);
  m_size = std::move(other.m_size);

  other.m_begin = pointer(static_cast<T*>(0));
  other.m_size = 0;

  return *this;
} // end vm_storage::propagate_allocator()
#endif

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::swap_allocators(true_type, const Alloc &)
{
} // end vm_storage::swap_allocators()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::swap_allocators(false_type, Alloc &other)
{
  NV_IF_TARGET(NV_IS_DEVICE, (
    // allocators must be equal when swapping containers with allocators that propagate on swap
    assert(!is_allocator_not_equal(other));
  ), (
    if (is_allocator_not_equal(other))
    {
      throw allocator_mismatch_on_swap();
    }
  ));

  thrust::swap(m_allocator, other);
} // end vm_storage::swap_allocators()

template<typename T, typename Alloc>
__host__ __device__
  bool vm_storage<T,Alloc>
    ::is_allocator_not_equal_dispatch(true_type /*is_always_equal*/, const Alloc &) const
{
  return false;
} // end vm_storage::is_allocator_not_equal_dispatch()

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  bool vm_storage<T,Alloc>
    ::is_allocator_not_equal_dispatch(false_type /*!is_always_equal*/, const Alloc& other) const
{
  return m_allocator != other;
} // end vm_storage::is_allocator_not_equal_dispatch()

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::deallocate_on_allocator_mismatch_dispatch(true_type, const vm_storage &other)
{
  if (m_allocator != other.m_allocator)
  {
    deallocate();
  }
} // end vm_storage::deallocate_on_allocator_mismatch()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::deallocate_on_allocator_mismatch_dispatch(false_type, const vm_storage &)
{
} // end vm_storage::deallocate_on_allocator_mismatch()

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::destroy_on_allocator_mismatch_dispatch(true_type, const vm_storage &other,
        iterator first, iterator last)
{
  if (m_allocator != other.m_allocator)
  {
    destroy(first, last);
  }
} // end vm_storage::destroy_on_allocator_mismatch()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::destroy_on_allocator_mismatch_dispatch(false_type, const vm_storage &,
        iterator, iterator)
{
} // end vm_storage::destroy_on_allocator_mismatch()

__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::propagate_allocator_dispatch(true_type, const vm_storage &other)
{
  m_allocator = other.m_allocator;
} // end vm_storage::propagate_allocator()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::propagate_allocator_dispatch(false_type, const vm_storage &)
{
} // end vm_storage::propagate_allocator()

#if THRUST_CPP_DIALECT >= 2011
__thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::propagate_allocator_dispatch(true_type, vm_storage &other)
{
  m_allocator = std::move(other.m_allocator);
} // end vm_storage::propagate_allocator()

template<typename T, typename Alloc>
__host__ __device__
  void vm_storage<T,Alloc>
    ::propagate_allocator_dispatch(false_type, vm_storage &)
{
} // end vm_storage::propagate_allocator()
#endif

} // end detail

template<typename T, typename Alloc>
__host__ __device__
  void swap(detail::vm_storage<T,Alloc> &lhs, detail::vm_storage<T,Alloc> &rhs)
{
  lhs.swap(rhs);
} // end swap()

THRUST_NAMESPACE_END
