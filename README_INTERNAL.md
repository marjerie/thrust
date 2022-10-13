
```bash

# Create build directory
mkdir build
cd build

# Configure
cmake -DTHRUST_CPP_DIALECT=17 ..

# To configure for debugging
cmake -DCMAKE_BUILD_TYPE=Debug -DTHRUST_CPP_DIALECT=17 ..

# Add -lcuda to the command in file /thrust/build/testing/CMakeFiles/thrust.test.vector_allocators.dir/link.txt
# After adding, the command will look like this:
# /usr/bin/g++ CMakeFiles/thrust.test.vector_allocators.dir/vector_allocators.cu.o -o ../bin/thrust.test.vector_allocators  ../lib/libthrust.test.framework.a -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl  -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib"

# Build test
make -j 12 thrust.all.test.vector_allocators

# Run test:
./bin/thrust.test.vector_allocators
```