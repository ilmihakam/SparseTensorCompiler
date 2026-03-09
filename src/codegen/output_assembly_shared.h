#pragma once

#include <functional>
#include <ostream>
#include <string>

#include "ir.h"
#include "semantic_ir.h"

namespace codegen {

namespace detail {

using EmitIndentedLine = std::function<void(int, const std::string&)>;

void emitMergeAssemblyBody(const EmitIndentedLine& emitLine,
                           const std::string& output,
                           const std::string& left,
                           const std::string& right,
                           bool isCSC,
                           bool isUnion,
                           const std::string& outputPositionVar);

void emitDynamicRowAssemblyBody(const EmitIndentedLine& emitLine,
                                const std::string& output,
                                const std::string& left,
                                const std::string& right,
                                bool isCSC,
                                const std::string& outputPositionVar);

void emitSampledAssemblyBody(const EmitIndentedLine& emitLine,
                             const std::string& output,
                             const std::string& sampled,
                             bool isCSC);

}  // namespace detail

void emitScheduledOutputAssembly(std::ostream& out,
                                 const sparseir::scheduled::Compute& compute,
                                 int indent);

}  // namespace codegen
