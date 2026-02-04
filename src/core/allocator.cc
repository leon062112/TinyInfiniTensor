#include "core/allocator.h"
#include <utility>

namespace infini
{
Allocator::Allocator(Runtime runtime) : runtime(runtime)
{
    used = 0;
    peak = 0;
    ptr = nullptr;

    // align to 64-bit
    alignment = sizeof(uint64_t);
}

Allocator::~Allocator()
{
    if (this->ptr != nullptr)
    {
        runtime->dealloc(this->ptr);
    }
}

size_t Allocator::alloc(size_t size)
{
    // planning phase only
    IT_ASSERT(this->ptr == nullptr);

    size = this->getAlignedSize(size);

    // ------------------------------------------------
    // 1. try reuse free blocks (first-fit)
    // ------------------------------------------------
    for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
    {
        if (it->second >= size)
        {
            size_t addr = it->first;
            size_t remain = it->second - size;

            freeBlocks.erase(it);
            if (remain > 0)
            {
                freeBlocks[addr + size] = remain;
            }

            used += size;
            peak = std::max(peak, used);
            return addr;
        }
    }

    // ------------------------------------------------
    // 2. bump allocation
    // ------------------------------------------------
    size_t addr = used;
    used += size;
    peak = std::max(peak, used);
    return addr;
}

void Allocator::free(size_t addr, size_t size)
{
    // planning phase only
    IT_ASSERT(this->ptr == nullptr);

    size = getAlignedSize(size);
    used -= size;

    // find first block with start >= addr
    auto it = freeBlocks.lower_bound(addr);

    // ------------------------------------------------
    // merge with previous block if adjacent
    // ------------------------------------------------
    if (it != freeBlocks.begin())
    {
        auto prev = std::prev(it);
        if (prev->first + prev->second == addr)
        {
            addr = prev->first;
            size += prev->second;
            freeBlocks.erase(prev);
        }
    }

    // ------------------------------------------------
    // merge with next block if adjacent
    // ------------------------------------------------
    if (it != freeBlocks.end() && addr + size == it->first)
    {
        size += it->second;
        freeBlocks.erase(it);
    }

    freeBlocks[addr] = size;
}

void *Allocator::getPtr()
{
    if (this->ptr == nullptr)
    {
        this->ptr = runtime->alloc(this->peak);
        printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
    }
    return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size)
{
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info()
{
    std::cout << "Used memory: " << this->used
              << ", peak memory: " << this->peak
              << ", free blocks: " << freeBlocks.size()
              << std::endl;
}
}
