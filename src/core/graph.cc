#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        // 规则1：去除相邻的做相反操作的 transpose 算子
        // 规则2：将 transpose 融入到 matmul 的 transA/transB 属性中

        std::unordered_set<Operator> opsToRemove;
        std::unordered_map<Tensor, Tensor> tensorReplacement; // old tensor -> new tensor

        // 规则1: 检查相邻的 transpose 是否做相反操作
        for (auto &op : ops)
        {
            if (opsToRemove.find(op) != opsToRemove.end())
                continue;

            if (op->getOpType() == OpType::Transpose)
            {
                auto outputs = op->getOutputs();
                if (!outputs.empty() && outputs[0]->getTargets().size() == 1)
                {
                    auto succ = outputs[0]->getTargets()[0];
                    if (succ->getOpType() == OpType::Transpose && opsToRemove.find(succ) == opsToRemove.end())
                    {
                        auto transpose1 = as<TransposeObj>(op);
                        auto transpose2 = as<TransposeObj>(succ);

                        auto permute1 = transpose1->getPermute();
                        auto permute2 = transpose2->getPermute();
                        bool isInverse = true;

                        if (permute1.size() == permute2.size())
                        {
                            for (size_t j = 0; j < permute1.size(); ++j)
                            {
                                if (permute2[permute1[j]] != (int)j)
                                {
                                    isInverse = false;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            isInverse = false;
                        }

                        if (isInverse)
                        {
                            // 标记两个 transpose 为删除
                            opsToRemove.insert(op);
                            opsToRemove.insert(succ);

                            // 记录 tensor 替换：succ 的输出应该替换为 op 的输入
                            auto opInput = op->getInputs()[0];
                            auto succOutput = succ->getOutputs()[0];
                            tensorReplacement[succOutput] = opInput;
                        }
                    }
                }
            }
        }

        // 规则2: 检查 matmul 的输入是否有 transpose（只针对没有被规则1处理的 transpose）
        for (auto &op : ops)
        {
            if (opsToRemove.find(op) != opsToRemove.end())
                continue;

            if (op->getOpType() == OpType::MatMul)
            {
                auto matmulOp = as<MatmulObj>(op);
                auto inputs = op->getInputs();

                // 检查第一个输入是否来自 transpose
                if (inputs[0]->getSource())
                {
                    auto pred = inputs[0]->getSource();
                    if (pred->getOpType() == OpType::Transpose &&
                        opsToRemove.find(pred) == opsToRemove.end())
                    {
                        auto transposeOp = as<TransposeObj>(pred);
                        auto permute = transposeOp->getPermute();
                        int rank = permute.size();

                        // 检查是否交换最后两个维度
                        if (rank >= 2 && permute[rank - 2] == rank - 1 && permute[rank - 1] == rank - 2)
                        {
                            bool otherDimsUnchanged = true;
                            for (int j = 0; j < rank - 2; ++j)
                            {
                                if (permute[j] != j)
                                {
                                    otherDimsUnchanged = false;
                                    break;
                                }
                            }

                            if (otherDimsUnchanged)
                            {
                                matmulOp->setTransA(!matmulOp->getTransA());
                                opsToRemove.insert(pred);
                                tensorReplacement[inputs[0]] = pred->getInputs()[0];
                            }
                        }
                    }
                }

                // 检查第二个输入是否来自 transpose
                if (inputs[1]->getSource())
                {
                    auto pred = inputs[1]->getSource();
                    if (pred->getOpType() == OpType::Transpose &&
                        opsToRemove.find(pred) == opsToRemove.end())
                    {
                        auto transposeOp = as<TransposeObj>(pred);
                        auto permute = transposeOp->getPermute();
                        int rank = permute.size();

                        if (rank >= 2 && permute[rank - 2] == rank - 1 && permute[rank - 1] == rank - 2)
                        {
                            bool otherDimsUnchanged = true;
                            for (int j = 0; j < rank - 2; ++j)
                            {
                                if (permute[j] != j)
                                {
                                    otherDimsUnchanged = false;
                                    break;
                                }
                            }

                            if (otherDimsUnchanged)
                            {
                                matmulOp->setTransB(!matmulOp->getTransB());
                                opsToRemove.insert(pred);
                                tensorReplacement[inputs[1]] = pred->getInputs()[0];
                            }
                        }
                    }
                }
            }
        }

        // 应用 tensor 替换：更新所有算子的输入
        for (auto &op : ops)
        {
            if (opsToRemove.find(op) == opsToRemove.end())
            {
                for (auto &input : op->getInputs())
                {
                    auto it = tensorReplacement.find(input);
                    if (it != tensorReplacement.end())
                    {
                        op->replaceInput(input, it->second);
                    }
                }
            }
        }

        // 更新 tensor 的 targets 列表：被替换的 tensor 应该把目标算子添加到新 tensor
        for (auto &p : tensorReplacement)
        {
            auto oldTensor = p.first;
            auto newTensor = p.second;
            // 找到所有使用 oldTensor 的后续算子
            for (auto &targetOp : oldTensor->getTargets())
            {
                if (opsToRemove.find(targetOp) == opsToRemove.end())
                {
                    // 将 targetOp 添加到 newTensor 的 targets 中
                    newTensor->addTarget(targetOp);
                }
            }
        }

        // 清理将被删除的 operator 关联的 tensor 的 source 指针
        for (auto &opToRemove : opsToRemove)
        {
            // 清理输出 tensor 的 source 指针
            for (auto &output : opToRemove->getOutputs())
            {
                if (output && output->getSource() == opToRemove)
                {
                    output->setSource(nullptr);
                }
            }
        }

        // 从剩余 operator 的 predecessor/successor 列表中移除被删除的 operator
        for (auto &op : ops)
        {
            if (opsToRemove.find(op) == opsToRemove.end())
            {
                for (auto &opToRemove : opsToRemove)
                {
                    op->removePredecessors(opToRemove);
                    op->removeSuccessors(opToRemove);
                }
            }
        }

        // 创建一个新的算子列表，只包含未被删除的算子
        OpVec newOps;
        for (auto &op : ops)
        {
            if (opsToRemove.find(op) == opsToRemove.end())
            {
                newOps.push_back(op);
            }
        }
        ops = std::move(newOps);

        // 清理不再使用的 tensor（只保留作为剩余算子的输入/输出的tensor）
        std::unordered_set<Tensor> usedTensors;
        for (auto &op : ops)
        {
            for (auto &input : op->getInputs())
                if (input) usedTensors.insert(input);
            for (auto &output : op->getOutputs())
                if (output) usedTensors.insert(output);
        }

        TensorVec newTensors;
        for (auto &t : tensors)
        {
            // 保留被使用的tensor，或者没有source的输入tensor
            if (usedTensors.find(t) != usedTensors.end())
            {
                newTensors.push_back(t);
            }
        }
        tensors = std::move(newTensors);

        // 更新 sorted 标志
        sorted = false;
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        // Track reference counts for each tensor
        std::unordered_map<Tensor, int> refCounts;
        for (auto &tensor : tensors)
        {
            // 图输入tensor（没有source）至少有1个引用，确保它们被分配内存
            refCounts[tensor] = tensor->getSource() ? 0 : 1;
        }

        // Count how many operators use each tensor as input
        for (auto &op : ops)
        {
            for (auto &input : op->getInputs())
            {
                if (input)
                {
                    refCounts[input]++;
                }
            }
        }

        // Track allocated memory offsets and sizes for each tensor
        std::unordered_map<Tensor, std::pair<size_t, size_t>> tensorAlloc; // tensor -> (offset, size)

        // Process operators in topological order
        for (auto &op : ops)
        {
            // Allocate memory for input tensors if needed (first time used)
            for (auto &input : op->getInputs())
            {
                if (input)
                {
                    // If reference count > 0 and not yet allocated, allocate now
                    if (refCounts[input] > 0 && tensorAlloc.find(input) == tensorAlloc.end())
                    {
                        size_t size = input->getBytes();
                        size_t offset = allocator.alloc(size);
                        tensorAlloc[input] = std::make_pair(offset, size);
                    }
                }
            }

            // Free input tensors when their reference count becomes zero
            for (auto &input : op->getInputs())
            {
                if (input)
                {
                    refCounts[input]--;
                    if (refCounts[input] == 0)
                    {
                        auto it = tensorAlloc.find(input);
                        if (it != tensorAlloc.end())
                        {
                            allocator.free(it->second.first, it->second.second);
                        }
                    }
                }
            }

            // Allocate memory for output tensors
            for (auto &output : op->getOutputs())
            {
                if (output)
                {
                    size_t size = output->getBytes();
                    size_t offset = allocator.alloc(size);
                    tensorAlloc[output] = std::make_pair(offset, size);
                }
            }
        }

        // Get the actual memory pointer and create Blob objects for each tensor
        void *basePtr = allocator.getPtr();
        for (auto &p : tensorAlloc)
        {
            Tensor tensor = p.first;
            void *tensorPtr = static_cast<char *>(basePtr) + p.second.first;
            Blob blob = make_ref<BlobObj>(runtime, tensorPtr);
            tensor->setDataBlob(blob);
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini