#include "optimizations.h"

namespace opt {

std::vector<std::string> getNaturalOrder(ir::Format format) {
    switch (format) {
        case ir::Format::CSR:
            return {"row", "col"};
        case ir::Format::CSC:
            return {"col", "row"};
        case ir::Format::Dense:
        default:
            return {"row", "col"};
    }
}

bool isOuterIndexDense(ir::Format format) {
    return format == ir::Format::CSR ||
           format == ir::Format::CSC ||
           format == ir::Format::Dense;
}

bool isInnerIndexSparse(ir::Format format) {
    return format == ir::Format::CSR || format == ir::Format::CSC;
}

bool needsReordering(const ir::Tensor& tensor) {
    if (tensor.format == ir::Format::Dense || tensor.indices.size() < 2) {
        return false;
    }

    const std::string& firstIdx = tensor.indices[0];
    const std::string& secondIdx = tensor.indices[1];

    if (tensor.format == ir::Format::CSR) {
        return secondIdx < firstIdx;
    }
    if (tensor.format == ir::Format::CSC) {
        return firstIdx < secondIdx;
    }
    return false;
}

} // namespace opt
