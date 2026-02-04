#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        const auto A = inputs[0];
        const auto B = inputs[1];
        auto aDims = A->getDims();
        auto bDims = B->getDims();
        auto aRank = aDims.size();
        auto bRank = bDims.size();

        // 计算矩阵乘法的维度 m, n, k
        // 如果 transA=true，则最后两个维度是 (k, m)，否则是 (m, k)
        // 如果 transB=true，则最后两个维度是 (n, k)，否则是 (k, n)
        if (transA) {
            k = aDims[aRank - 2];
            m = aDims[aRank - 1];
        } else {
            m = aDims[aRank - 2];
            k = aDims[aRank - 1];
        }

        if (transB) {
            n = bDims[bRank - 2];
            // k 应该与 B 的维度匹配，已在上面的 k 计算
        } else {
            k = bDims[bRank - 2];
            n = bDims[bRank - 1];
        }

        Shape outputShape;
        size_t maxRank = std::max(aRank, bRank);

        // 处理 batch 维度的广播
        if (maxRank > 2) {
            // 提取 batch 维度
            Shape aBatchDims, bBatchDims;
            for (size_t i = 0; i < aRank - 2; ++i) {
                aBatchDims.push_back(aDims[i]);
            }
            for (size_t i = 0; i < bRank - 2; ++i) {
                bBatchDims.push_back(bDims[i]);
            }

            // 广播 batch 维度
            Shape batchOutput = infer_broadcast(aBatchDims, bBatchDims);
            outputShape = batchOutput;
        }

        // 添加最后两个矩阵维度
        outputShape.push_back(m);
        outputShape.push_back(n);

        return {{outputShape}};
    }

} // namespace infini