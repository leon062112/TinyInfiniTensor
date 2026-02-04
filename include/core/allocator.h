#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>
#include <iostream>

namespace infini
{
class Allocator
{
private:
    Runtime runtime;

    size_t used;
    size_t peak;
    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    // free block list:
    // key   : start offset
    // value : block size
    std::map<size_t, size_t> freeBlocks;

public:
    Allocator(Runtime runtime);
    virtual ~Allocator();

    // simulate allocation, return offset
    size_t alloc(size_t size);

    // simulate free
    void free(size_t addr, size_t size);

    // do real allocation
    void *getPtr();

    void info();

private:
    size_t getAlignedSize(size_t size);
};
}
