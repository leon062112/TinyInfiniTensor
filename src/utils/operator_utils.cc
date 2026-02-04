#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {
    Shape a = A;
    Shape b = B;

    // 1. 对齐 rank（前面补 1）
    if (a.size() < b.size()) {
        a.insert(a.begin(), b.size() - a.size(), 1);
    } else if (b.size() < a.size()) {
        b.insert(b.begin(), a.size() - b.size(), 1);
    }

    // 2. 逐维计算 broadcast 后的 shape
    Shape out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) {
            out[i] = a[i];
        } else if (a[i] == 1) {
            out[i] = b[i];
        } else if (b[i] == 1) {
            out[i] = a[i];
        } else {
            IT_ASSERT(false && "Broadcast shape mismatch");
        }
    }

    return out;
}


int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
