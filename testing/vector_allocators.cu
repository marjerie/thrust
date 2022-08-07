#include <unittest/unittest.h>

#include <thrust/detail/config.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename BaseAlloc, bool PropagateOnSwap>
class stateful_allocator : public BaseAlloc
{
  typedef thrust::detail::allocator_traits<BaseAlloc> base_traits;

public:
    stateful_allocator(int i) : state(i)
    {
    }

    ~stateful_allocator() {}

    stateful_allocator(const stateful_allocator &other)
        : BaseAlloc(other), state(other.state)
    {
    }

    stateful_allocator & operator=(const stateful_allocator & other)
    {
        state = other.state;
        return *this;
    }

#if THRUST_CPP_DIALECT >= 2011
    stateful_allocator(stateful_allocator && other)
        : BaseAlloc(std::move(other)), state(other.state)
    {
        other.state = 0;
    }

    stateful_allocator & operator=(stateful_allocator && other)
    {
        state = other.state;
        other.state = 0;
        return *this;
    }
#endif

    static int last_allocated;
    static int last_deallocated;

    typedef typename base_traits::pointer pointer;
    typedef typename base_traits::const_pointer const_pointer;
    typedef typename base_traits::reference reference;
    typedef typename base_traits::const_reference const_reference;

    pointer allocate(std::size_t size)
    {
        BaseAlloc alloc;
        last_allocated = state;
        return base_traits::allocate(alloc, size);
    }

    void deallocate(pointer ptr, std::size_t size)
    {
        BaseAlloc alloc;
        last_deallocated = state;
        return base_traits::deallocate(alloc, ptr, size);
    }

    static void construct(pointer ptr)
    {
      BaseAlloc alloc;
      return base_traits::construct(alloc, ptr);
    }

    static void destroy(pointer ptr)
    {
      BaseAlloc alloc;
      return base_traits::destroy(alloc, ptr);
    }

    bool operator==(const stateful_allocator &rhs) const
    {
        return state == rhs.state;
    }

    bool operator!=(const stateful_allocator &rhs) const
    {
        return state != rhs.state;
    }

    friend std::ostream & operator<<(std::ostream &os,
        const stateful_allocator & alloc)
    {
        os << "stateful_alloc(" << alloc.state << ")";
        return os;
    }

    typedef thrust::detail::false_type is_always_equal;
    typedef thrust::detail::true_type propagate_on_container_copy_assignment;
    typedef thrust::detail::true_type propagate_on_container_move_assignment;
    typedef thrust::detail::integral_constant<bool, PropagateOnSwap> propagate_on_container_swap;

private:
    int state;
};

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_allocated = 0;

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_deallocated = 0;

typedef stateful_allocator<std::allocator<int>, true> host_alloc;
typedef stateful_allocator<thrust::device_allocator<int>, true> device_alloc;

typedef thrust::host_vector<int, host_alloc> host_vector;
typedef thrust::device_vector<int, device_alloc> device_vector;

typedef stateful_allocator<std::allocator<int>, false> host_alloc_nsp;
typedef stateful_allocator<thrust::device_allocator<int>, false> device_alloc_nsp;

typedef thrust::host_vector<int, host_alloc_nsp> host_vector_nsp;
typedef thrust::device_vector<int, device_alloc_nsp> device_vector_nsp;

template<typename Vector>
void TestVectorAllocatorConstructors()
{
    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(alloc1);
    ASSERT_EQUAL(v1.get_allocator(), alloc1);

    Vector v2(10, alloc1);
    ASSERT_EQUAL(v2.size(), 10u);
    ASSERT_EQUAL(v2.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v3(10, 17, alloc1);
    ASSERT_EQUAL((v3 == std::vector<int>(10, 17)), true);
    ASSERT_EQUAL(v3.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v4(v3, alloc2);
    ASSERT_EQUAL((v3 == v4), true);
    ASSERT_EQUAL(v4.get_allocator(), alloc2);
    ASSERT_EQUAL(Alloc::last_allocated, 2);
    Alloc::last_allocated = 0;

#if THRUST_CPP_DIALECT >= 2011
    // FIXME: uncomment this after the vector_base(vector_base&&, const Alloc&)
    // is fixed and implemented
    // Vector v5(std::move(v3), alloc2);
    // ASSERT_EQUAL((v4 == v5), true);
    // ASSERT_EQUAL(v5.get_allocator(), alloc2);
    // ASSERT_EQUAL(Alloc::last_allocated, 1);
    // Alloc::last_allocated = 0;
#endif

    Vector v6(v4.begin(), v4.end(), alloc2);
    ASSERT_EQUAL((v4 == v6), true);
    ASSERT_EQUAL(v6.get_allocator(), alloc2);
    ASSERT_EQUAL(Alloc::last_allocated, 2);
}

void TestVectorAllocatorConstructorsHost()
{
    TestVectorAllocatorConstructors<host_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorConstructorsHost);

void TestVectorAllocatorConstructorsDevice()
{
    TestVectorAllocatorConstructors<device_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorConstructorsDevice);

template<typename Vector>
void TestVectorAllocatorPropagateOnCopyAssignment()
{
    ASSERT_EQUAL(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = v1;
    ASSERT_EQUAL((v1 == v2), true);
    ASSERT_EQUAL(v2.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 1);
    ASSERT_EQUAL(Alloc::last_deallocated, 2);
}

void TestVectorAllocatorPropagateOnCopyAssignmentHost()
{
    TestVectorAllocatorPropagateOnCopyAssignment<host_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnCopyAssignmentHost);

void TestVectorAllocatorPropagateOnCopyAssignmentDevice()
{
    TestVectorAllocatorPropagateOnCopyAssignment<device_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnCopyAssignmentDevice);

#if THRUST_CPP_DIALECT >= 2011
template<typename Vector>
void TestVectorAllocatorPropagateOnMoveAssignment()
{
    typedef typename Vector::allocator_type Alloc;
    ASSERT_EQUAL(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    {
    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = std::move(v1);
    ASSERT_EQUAL(v2.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 2);
    ASSERT_EQUAL(Alloc::last_deallocated, 2);
    }

    ASSERT_EQUAL(Alloc::last_deallocated, 1);
}

void TestVectorAllocatorPropagateOnMoveAssignmentHost()
{
    TestVectorAllocatorPropagateOnMoveAssignment<host_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnMoveAssignmentHost);

void TestVectorAllocatorPropagateOnMoveAssignmentDevice()
{
    TestVectorAllocatorPropagateOnMoveAssignment<device_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnMoveAssignmentDevice);
#endif

template<typename Vector>
void TestVectorAllocatorPropagateOnSwap()
{
    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(17, alloc1);
    thrust::swap(v1, v2);

    ASSERT_EQUAL(v1.size(), 17u);
    ASSERT_EQUAL(v2.size(), 10u);

    Vector v3(15, alloc1);
    Vector v4(31, alloc2);
    ASSERT_THROWS(thrust::swap(v3, v4), thrust::detail::allocator_mismatch_on_swap);
}

void TestVectorAllocatorPropagateOnSwapHost()
{
    TestVectorAllocatorPropagateOnSwap<host_vector_nsp>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnSwapHost);

void TestVectorAllocatorPropagateOnSwapDevice()
{
    TestVectorAllocatorPropagateOnSwap<device_vector_nsp>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnSwapDevice);

void TestVirtualVectorAllocator() {
    
    typedef thrust::device_vector<int,thrust::system::cuda::virtual_allocator<int>> dev_vec;

    size_t sz = 10;

    std::cout << "-------------------------------- CREATING V11 ---------------------------------" << '\n';

    dev_vec v11(sz,15);

    std::cout << "v11 capacity: " << v11.capacity() << '\n';
    std::cout << "v11 size: " << v11.size() << '\n';

    for (size_t i=0; i<v11.size(); i++) {
        std::cout << "v11[" <<i<<"] = "  << v11[i] << '\n';
    }

    std::cout << "-------------------------------- CREATING V1 ---------------------------------" << '\n';

    dev_vec v1(sz,68);

    std::cout << "v1 capacity: " << v1.capacity() << '\n';
    std::cout << "v1 size: " << v1.size() << '\n';

    for (size_t i=0; i<v1.size(); i++) {
        std::cout << "v1[" <<i<<"] = "  << v1[i] << '\n';
    }

    // std::cout << "-------------------------------- CREATING V1 = V11 ---------------------------------" << '\n';
    // FIXME: uncomment this after operator= is fixed
    // v1 = v11;

    // std::cout << "v1 capacity: " << v1.capacity() << '\n';
    // std::cout << "v1 size: " << v1.size() << '\n';

    // for (size_t i=0; i<v1.size(); i++) {
    //     std::cout << "v1[" <<i<<"] = "  << v1[i] << '\n';
    // }

    std::cout << "-------------------------------- ASSIGNING VALUES TO V11[2,5,8] ---------------------------------" << '\n';

    v11[2] = 100;
    v11[5] = 200;
    v11[8] = 300;

    std::cout << "v11 capacity: " << v11.capacity() << '\n';
    std::cout << "v11 size: " << v11.size() << '\n';

    for (size_t i=0; i<v11.size(); i++) {
        std::cout << "v11[" <<i<<"] = "  << v11[i] << '\n';
    }

    std::cout << "-------------------------------- PUSHING BACK V1 ---------------------------------" << '\n';

    v1.push_back(20);
    v1.push_back(21);
    v1.push_back(20);
    v1.push_back(21);
    v1.push_back(20);
    v1.push_back(21);

    std::cout << "v1 capacity: " << v1.capacity() << '\n';
    std::cout << "v1 size: " << v1.size() << '\n';

    for (size_t i=0; i<v1.size(); i++) {
        std::cout << "v1[" <<i<<"] = "  << v1[i] << '\n';
    }

    // std::cout << "-------------------------------- CREATING V11 = V1 ---------------------------------" << '\n';
    // FIXME: uncomment this after operator= is fixed
    // v11 = v1;

    // std::cout << "v11 capacity: " << v11.capacity() << '\n';
    // std::cout << "v11 size: " << v11.size() << '\n';

    // for (size_t i=0; i<v11.size(); i++) {
    //     std::cout << "v11[" <<i<<"] = "  << v11[i] << '\n';
    // }

    sz = 524286;

    std::cout << "-------------------------------- CREATING V3 ---------------------------------" << '\n';

    dev_vec v3(sz,1);

    std::cout << "v3 capacity: " << v3.capacity() << '\n';
    std::cout << "v3 size: " << v3.size() << '\n';

    for (size_t i=0; i<v3.size(); i++) {
        if (v3[i] != 1) std::cout << "error element: v3[" <<i<<"] = "  << v3[i] << '\n';
    }

    std::cout << "-------------------------------- CREATING V4 ---------------------------------" << '\n';

    dev_vec v4(sz,2);

    std::cout << "v4 capacity: " << v4.capacity() << '\n';
    std::cout << "v4 size: " << v4.size() << '\n';

    for (size_t i=0; i<v4.size(); i++) {
        if (v4[i] != 2) std::cout << "error element: v4[" <<i<<"] = "  << v4[i] << '\n';
    }
    
    std::cout << "-------------------------------- PUSHING BACK V3 ---------------------------------" << '\n';

    v3.push_back(20);
    v3.push_back(21);
    v3.push_back(20);
    v3.push_back(21);
    v3.push_back(20);
    v3.push_back(21);

    std::cout << "v3 capacity: " << v3.capacity() << '\n';
    std::cout << "v3 size: " << v3.size() << '\n';

    for (size_t i=sz-2; i<v3.size(); i++) {
        std::cout << "v3[" <<i<<"] = "  << v3[i] << '\n';
    }

    size_t rem = v3.capacity() - v3.size();

    std::cout << "-------------------------------- PUSHING BACK V3 WITH " << rem << " ELEMENTS ---------------------------------" << '\n';

    for (size_t i=0; i<rem; i++) {
        v3.push_back(20);
    }

    std::cout << "v3 capacity: " << v3.capacity() << '\n';
    std::cout << "v3 size: " << v3.size() << '\n';

    for (size_t i=v3.capacity()-15; i<v3.size(); i++) {
        if (v3[i] != 20) std::cout << "error element: v3[" <<i<<"] = "  << v3[i] << '\n';
    }

    std::cout << "-------------------------------- CREATING V5 ---------------------------------" << '\n';

    dev_vec v5(sz,2);

    std::cout << "v5 capacity: " << v5.capacity() << '\n';
    std::cout << "v5 size: " << v5.size() << '\n';

    for (size_t i=0; i<v5.size(); i++) {
        if (v5[i] != 2) std::cout << "error element: v5[" <<i<<"] = "  << v5[i] << '\n';
    }

    std::cout << "-------------------------------- PUSHING BACK V3 ---------------------------------" << '\n';

    v3.push_back(200);
    v3.push_back(201);
    v3.push_back(200);
    v3.push_back(210);
    v3.push_back(200);
    v3.push_back(210);

    std::cout << "v3 capacity: " << v3.capacity() << '\n';
    std::cout << "v3 size: " << v3.size() << '\n';

    for (size_t i=v3.size()-10; i<v3.size(); i++) {
        std::cout << "v3[" <<i<<"] = "  << v3[i] << '\n';
    }

    std::cout << "-------------------------------- POPPING BACK V3 ---------------------------------" << '\n';

    v3.pop_back();
    v3.pop_back();

    std::cout << "v3 capacity: " << v3.capacity() << '\n';
    std::cout << "v3 size: " << v3.size() << '\n';

    for (size_t i=v3.size()-10; i<v3.size(); i++) {
        std::cout << "v3[" <<i<<"] = "  << v3[i] << '\n';
    }

    std::cout << "-------------------------------- POPPING BACK V5 ---------------------------------" << '\n';

    v5.pop_back();
    v5.pop_back();

    std::cout << "v5 capacity: " << v5.capacity() << '\n';
    std::cout << "v5 size: " << v5.size() << '\n';

    std::cout << "-------------------------------- DEALLOCATING ---------------------------------" << '\n';

    // v4 = v3;

    // std::cout << "v4 capacity: " << v4.capacity() << '\n';
    // std::cout << "v4 size: " << v4.size() << '\n';

    // for (size_t i=sz-2; i<v4.size(); i++) {
    //     if (v4[i] != 1) std::cout << "v4[" <<i<<"] = "  << v4[i] << '\n';
    // }

}

DECLARE_UNITTEST(TestVirtualVectorAllocator);