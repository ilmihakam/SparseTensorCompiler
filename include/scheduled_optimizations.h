#ifndef SCHEDULED_OPTIMIZATIONS_H
#define SCHEDULED_OPTIMIZATIONS_H

#include "optimizations.h"
#include "semantic_ir.h"

namespace opt {

void applyReordering(sparseir::scheduled::Compute& compute);
void applyBlocking(sparseir::scheduled::Compute& compute, const OptConfig& config);
void applyPositionBlocking(sparseir::scheduled::Compute& compute, const OptConfig& config);
void applyLoopInterchange(sparseir::scheduled::Compute& compute, const OptConfig& config);
void applyOptimizations(sparseir::scheduled::Compute& compute, const OptConfig& config);
void applyOptimizations(sparseir::scheduled::Program& program, const OptConfig& config);

} // namespace opt

#endif // SCHEDULED_OPTIMIZATIONS_H
