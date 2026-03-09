#include <gtest/gtest.h>
#include <fstream>
#include <sstream>

#include "ast.h"
#include "codegen.h"
#include "ir.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"

using namespace SparseTensorCompiler;

namespace {

std::unique_ptr<Program> makeGeneralContractionProgram(bool wrapInFor = false) {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"8"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"6"}));
    program->addStatement(std::make_unique<Declaration>(
        "z", "Dense", std::vector<std::string>{"6"}));

    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto zAccess = std::make_unique<TensorAccess>("z", std::vector<std::string>{"j"});
    auto mul1 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess));
    auto mul2 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul1), std::move(zAccess));
    auto compute = std::make_unique<Computation>(std::move(lhs), std::move(mul2));

    if (!wrapInFor) {
        program->addStatement(std::move(compute));
        return program;
    }

    auto region = std::make_unique<ForStatement>(
        std::vector<std::string>{"A", "y"},
        std::vector<std::string>{"i"});
    region->addStatement(std::move(compute));
    program->addStatement(std::move(region));
    return program;
}

std::unique_ptr<Program> makeSpMVProgram(const std::string& format = "CSR",
                                         bool wrapInFor = false) {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"8"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", format, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"6"}));

    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto compute = std::make_unique<Computation>(
        std::move(lhs),
        std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess)));

    if (!wrapInFor) {
        program->addStatement(std::move(compute));
        return program;
    }

    auto region = std::make_unique<ForStatement>(
        std::vector<std::string>{"A", "y"},
        std::vector<std::string>{"i"});
    region->addStatement(std::move(compute));
    program->addStatement(std::move(region));
    return program;
}

std::unique_ptr<Program> makeSpMMProgram(const std::string& format = "CSR") {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"8", "4"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", format, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "B", "Dense", std::vector<std::string>{"6", "4"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
    auto bAccess = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
    program->addStatement(std::make_unique<Computation>(
        std::move(lhs),
        std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(bAccess))));
    return program;
}

std::unique_ptr<Program> makeSpAddProgram(const std::string& format = "CSR",
                                          const std::string& outputFormat = "Dense") {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "C", outputFormat, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", format, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "B", format, std::vector<std::string>{"8", "6"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto bAccess = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i", "j"});
    program->addStatement(std::make_unique<Computation>(
        std::move(lhs),
        std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(aAccess), std::move(bAccess))));
    return program;
}

std::unique_ptr<Program> makeSpElMulProgram(const std::string& format = "CSR",
                                            const std::string& outputFormat = "Dense") {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "C", outputFormat, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", format, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "B", format, std::vector<std::string>{"8", "6"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto bAccess = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i", "j"});
    program->addStatement(std::make_unique<Computation>(
        std::move(lhs),
        std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(bAccess))));
    return program;
}

std::unique_ptr<Program> makeSDDMMProgram(const std::string& format = "CSR",
                                          const std::string& outputFormat = "Dense") {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "C", outputFormat, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "S", format, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "D", "Dense", std::vector<std::string>{"8", "4"}));
    program->addStatement(std::make_unique<Declaration>(
        "E", "Dense", std::vector<std::string>{"4", "6"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto sAccess = std::make_unique<TensorAccess>("S", std::vector<std::string>{"i", "j"});
    auto dAccess = std::make_unique<TensorAccess>("D", std::vector<std::string>{"i", "k"});
    auto eAccess = std::make_unique<TensorAccess>("E", std::vector<std::string>{"k", "j"});
    auto denseMul = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(dAccess), std::move(eAccess));
    program->addStatement(std::make_unique<Computation>(
        std::move(lhs),
        std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(sAccess), std::move(denseMul))));
    return program;
}

std::unique_ptr<Program> makeSpGEMMProgram(const std::string& format = "CSR",
                                           const std::string& outputFormat = "Dense") {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "C", outputFormat, std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", format, std::vector<std::string>{"8", "4"}));
    program->addStatement(std::make_unique<Declaration>(
        "B", format, std::vector<std::string>{"4", "6"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
    auto bAccess = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
    program->addStatement(std::make_unique<Computation>(
        std::move(lhs),
        std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(bAccess))));
    return program;
}

const sparseir::semantic::Compute* findSemanticCompute(
    const std::vector<std::unique_ptr<sparseir::semantic::Stmt>>& statements) {
    for (const auto& stmt : statements) {
        if (auto* compute = dynamic_cast<const sparseir::semantic::Compute*>(stmt.get())) {
            return compute;
        }
        if (auto* region = dynamic_cast<const sparseir::semantic::Region*>(stmt.get())) {
            if (auto* nested = findSemanticCompute(region->body)) {
                return nested;
            }
        }
    }
    return nullptr;
}

const sparseir::scheduled::Compute* findScheduledCompute(
    const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>& statements) {
    for (const auto& stmt : statements) {
        if (auto* compute = dynamic_cast<const sparseir::scheduled::Compute*>(stmt.get())) {
            return compute;
        }
        if (auto* region = dynamic_cast<const sparseir::scheduled::Region*>(stmt.get())) {
            if (auto* nested = findScheduledCompute(region->body)) {
                return nested;
            }
        }
    }
    return nullptr;
}

bool containsRawStmt(const ir::IRStmt& stmt) {
    if (dynamic_cast<const ir::IRRawStmt*>(&stmt)) {
        return true;
    }
    if (auto* ifStmt = dynamic_cast<const ir::IRIfStmt*>(&stmt)) {
        for (const auto& bodyStmt : ifStmt->thenBody) {
            if (containsRawStmt(*bodyStmt)) {
                return true;
            }
        }
    }
    if (auto* forStmt = dynamic_cast<const ir::IRForStmt*>(&stmt)) {
        for (const auto& bodyStmt : forStmt->body) {
            if (containsRawStmt(*bodyStmt)) {
                return true;
            }
        }
    }
    return false;
}

bool loopTreeContainsRawStmt(const sparseir::scheduled::Loop& loop) {
    for (const auto& stmt : loop.preStmts) {
        if (containsRawStmt(*stmt)) {
            return true;
        }
    }
    for (const auto& stmt : loop.postStmts) {
        if (containsRawStmt(*stmt)) {
            return true;
        }
    }
    for (const auto& child : loop.children) {
        if (loopTreeContainsRawStmt(*child)) {
            return true;
        }
    }
    return false;
}

std::string readTextFile(const std::string& path) {
    std::ifstream input(path);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

} // namespace

TEST(SemanticIRTest, LowerToSemanticProgramBuildsIteratorGraphForGeneralContraction) {
    auto ast = makeGeneralContractionProgram();
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);

    ASSERT_NE(semanticProgram, nullptr);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    ASSERT_EQ(compute->iteratorGraph.iterators.size(), 2u);
    EXPECT_EQ(compute->iteratorGraph.iterators[0].indexName, "i");
    EXPECT_EQ(compute->iteratorGraph.iterators[0].kind, sparseir::IteratorKind::Dense);
    EXPECT_EQ(compute->iteratorGraph.iterators[1].indexName, "j");
    EXPECT_EQ(compute->iteratorGraph.iterators[1].kind, sparseir::IteratorKind::Sparse);
}

TEST(SemanticIRTest, ScheduleProgramMarksExternallyBoundRegionIndices) {
    auto ast = makeGeneralContractionProgram(true);
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto scheduledProgram = sparseir::scheduleProgram(*semanticProgram);

    ASSERT_NE(scheduledProgram, nullptr);
    auto* compute = findScheduledCompute(scheduledProgram->statements);
    ASSERT_NE(compute, nullptr);
    ASSERT_TRUE(compute->fullyLowered);
    ASSERT_NE(compute->rootLoop, nullptr);
    EXPECT_TRUE(compute->rootLoop->isExternallyBound);
    ASSERT_EQ(compute->rootLoop->children.size(), 1u);
    EXPECT_EQ(compute->rootLoop->children[0]->indexName, "j");
    EXPECT_FALSE(compute->rootLoop->children[0]->isExternallyBound);
}

TEST(SemanticIRTest, ScheduleComputeLowersSpMVThroughNewPath) {
    auto ast = makeSpMVProgram();
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 1);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->indexName, "j");
}

TEST(SemanticIRTest, ScheduleComputeLowersCSCSpMMWithParentOverride) {
    auto ast = makeSpMMProgram("CSC");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 1);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "k");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    auto* sparseLoop = scheduled->rootLoop->children[0].get();
    ASSERT_NE(sparseLoop, nullptr);
    EXPECT_EQ(sparseLoop->indexName, "i");
    EXPECT_EQ(sparseLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(sparseLoop->iterator.beginExpr, "A->col_ptr[k]");
    EXPECT_EQ(sparseLoop->iterator.endExpr, "A->col_ptr[k + 1]");
    ASSERT_EQ(sparseLoop->children.size(), 1u);
    EXPECT_EQ(sparseLoop->children[0]->indexName, "j");
}

TEST(SemanticIRTest, ScheduleComputePopulatesExplicitEmissionMetadataForCSCSpMM) {
    auto ast = makeSpMMProgram("CSC");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_NE(scheduled->rootLoop, nullptr);

    const auto* kLoop = scheduled->rootLoop.get();
    EXPECT_EQ(kLoop->headerKind, sparseir::scheduled::LoopHeaderKind::DenseFor);
    EXPECT_EQ(kLoop->lowerExpr, "0");
    EXPECT_EQ(kLoop->upperExpr, "A->cols");

    ASSERT_EQ(kLoop->children.size(), 1u);
    const auto* sparseLoop = kLoop->children[0].get();
    ASSERT_NE(sparseLoop, nullptr);
    EXPECT_EQ(sparseLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(sparseLoop->iterator.pointerVar, "pA");
    EXPECT_EQ(sparseLoop->iterator.beginExpr, "A->col_ptr[k]");
    EXPECT_EQ(sparseLoop->iterator.endExpr, "A->col_ptr[k + 1]");
    EXPECT_EQ(sparseLoop->bindingVarName, "i");
    EXPECT_EQ(sparseLoop->bindingExpr, "A->row_idx[pA]");
}

TEST(SemanticIRTest, ScheduledProgramCodegenEmitsCallsRegionsAndDirectComputeLoops) {
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"8"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"6"}));

    auto call = std::make_unique<CallStatement>("initialize");
    call->addArgument(std::make_unique<Identifier>("x"));
    program->addStatement(std::move(call));

    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto compute = std::make_unique<Computation>(
        std::move(lhs),
        std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess)));
    auto region = std::make_unique<ForStatement>(
        std::vector<std::string>{"A", "y"},
        std::vector<std::string>{"i"});
    region->addStatement(std::move(compute));
    program->addStatement(std::move(region));

    auto semanticProgram = sparseir::lowerToSemanticProgram(*program);
    auto scheduledProgram = sparseir::scheduleProgram(*semanticProgram);
    ASSERT_NE(scheduledProgram, nullptr);

    opt::OptConfig config;
    std::string outputFile = "/tmp/scheduled_program_codegen_test.c";
    ASSERT_TRUE(codegen::generateProgramToFile(*scheduledProgram, config, outputFile));

    std::string code = readTextFile(outputFile);
    EXPECT_NE(code.find("extern void initialize(double*);"), std::string::npos);
    EXPECT_NE(code.find("initialize(x);"), std::string::npos);
    EXPECT_NE(code.find("for (int i = 0; i < 8; i++) {"), std::string::npos);
    EXPECT_NE(code.find("for (int pA = A->row_ptr[i];"), std::string::npos);
    EXPECT_EQ(code.find("lowerToIRProgram"), std::string::npos);
}

TEST(SemanticIRTest, ScheduledProgramCodegenBuildsSparseOutputAssemblyInline) {
    auto ast = makeSpAddProgram("CSR", "CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto scheduledProgram = sparseir::scheduleProgram(*semanticProgram);
    ASSERT_NE(scheduledProgram, nullptr);

    opt::OptConfig config;
    std::string outputFile = "/tmp/scheduled_program_sparse_output_test.c";
    ASSERT_TRUE(codegen::generateProgramToFile(*scheduledProgram, config, outputFile));

    std::string code = readTextFile(outputFile);
    EXPECT_NE(code.find("C->row_ptr = (int*)calloc((size_t)C->rows + 1, sizeof(int));"),
              std::string::npos);
    EXPECT_NE(code.find("while (pA < endA && pB < endB) {"), std::string::npos);
    EXPECT_NE(code.find("C->vals = (double*)calloc((size_t)C->nnz, sizeof(double));"),
              std::string::npos);
    EXPECT_NE(code.find("sp_csr_get(A, i, j)"), std::string::npos);
}

TEST(SemanticIRTest, ScheduleComputeLowersSpAddWithUnionMergeLoop) {
    auto ast = makeSpAddProgram("CSR", "Dense");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseMerge);
    EXPECT_EQ(scheduled->rootLoop->children[0]->merge.strategy, ir::MergeStrategy::Union);
    EXPECT_EQ(scheduled->rootLoop->children[0]->merge.terms.size(), 2u);
}

TEST(SemanticIRTest, ScheduleComputeLowersSpElMulWithMergeLoop) {
    auto ast = makeSpElMulProgram("CSC", "Dense");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "j");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseMerge);
    EXPECT_EQ(scheduled->rootLoop->children[0]->merge.strategy, ir::MergeStrategy::Intersection);
    ASSERT_EQ(scheduled->rootLoop->children[0]->merge.terms.size(), 2u);
}

TEST(SemanticIRTest, ScheduleComputeLowersSparseOutputSpAddThroughOutputPattern) {
    auto ast = makeSpAddProgram("CSR", "CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::SparseFixedPattern);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    EXPECT_EQ(scheduled->outputPattern, sparseir::OutputPatternKind::Union);
    ASSERT_EQ(scheduled->patternSources.size(), 2u);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    auto* outputLoop = scheduled->rootLoop->children[0].get();
    ASSERT_NE(outputLoop, nullptr);
    EXPECT_EQ(outputLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(outputLoop->iterator.beginExpr, "C->row_ptr[i]");
    EXPECT_EQ(outputLoop->iterator.endExpr, "C->row_ptr[i + 1]");
    EXPECT_FALSE(outputLoop->postStmts.empty());
}

TEST(SemanticIRTest, ScheduleComputeLowersSparseOutputSpElMulThroughOutputPattern) {
    auto ast = makeSpElMulProgram("CSR", "CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::SparseFixedPattern);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    EXPECT_EQ(scheduled->outputPattern, sparseir::OutputPatternKind::Intersection);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(scheduled->rootLoop->children[0]->iterator.beginExpr, "C->row_ptr[i]");
    EXPECT_EQ(scheduled->rootLoop->children[0]->iterator.endExpr, "C->row_ptr[i + 1]");
}

TEST(SemanticIRTest, ScheduleComputeLowersSparseOutputSDDMMThroughOutputPattern) {
    auto ast = makeSDDMMProgram("CSR", "CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::SparseFixedPattern);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    EXPECT_EQ(scheduled->outputPattern, sparseir::OutputPatternKind::Sampled);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    auto* samplingLoop = scheduled->rootLoop->children[0].get();
    ASSERT_NE(samplingLoop, nullptr);
    EXPECT_EQ(samplingLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(samplingLoop->iterator.beginExpr, "C->row_ptr[i]");
    EXPECT_EQ(samplingLoop->iterator.endExpr, "C->row_ptr[i + 1]");
    EXPECT_FALSE(samplingLoop->preStmts.empty());
    EXPECT_FALSE(samplingLoop->postStmts.empty());
}

TEST(SemanticIRTest, ScheduleComputeLowersSparseOutputSpGEMMThroughOutputPattern) {
    auto ast = makeSpGEMMProgram("CSR", "CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::HashPerRow);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    EXPECT_EQ(scheduled->outputPattern, sparseir::OutputPatternKind::DynamicRowAccumulator);
    EXPECT_FALSE(scheduled->prologueStmts.empty());
    EXPECT_FALSE(scheduled->epilogueStmts.empty());
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    EXPECT_FALSE(scheduled->rootLoop->preStmts.empty());
    EXPECT_FALSE(scheduled->rootLoop->postStmts.empty());
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    ASSERT_EQ(scheduled->rootLoop->children[0]->children.size(), 1u);
}

TEST(SemanticIRTest, ScheduleComputeLowersCSCSpGEMMThroughOutputPattern) {
    auto ast = makeSpGEMMProgram("CSC", "CSC");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    EXPECT_EQ(scheduled->outputPattern, sparseir::OutputPatternKind::DynamicRowAccumulator);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "j");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(scheduled->rootLoop->children[0]->iterator.beginExpr, "B->col_ptr[j]");
    EXPECT_EQ(scheduled->rootLoop->children[0]->iterator.endExpr, "B->col_ptr[j + 1]");
    ASSERT_EQ(scheduled->rootLoop->children[0]->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->iterator.beginExpr, "A->col_ptr[k]");
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->iterator.endExpr, "A->col_ptr[k + 1]");
}

TEST(SemanticIRTest, ScheduleComputeLowersDenseSpGEMMWithoutKernelTags) {
    auto ast = makeSpGEMMProgram("CSR", "Dense");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(scheduled->rootLoop->children[0]->iterator.beginExpr, "A->row_ptr[i]");
    EXPECT_EQ(scheduled->rootLoop->children[0]->iterator.endExpr, "A->row_ptr[i + 1]");
    ASSERT_EQ(scheduled->rootLoop->children[0]->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->iterator.beginExpr, "B->row_ptr[k]");
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->iterator.endExpr, "B->row_ptr[k + 1]");
}

TEST(SemanticIRTest, ScheduleComputeLowersDenseSDDMMWithoutKernelTags) {
    auto ast = makeSDDMMProgram("CSR", "Dense");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 1);
    EXPECT_EQ(compute->exprInfo.numDenseInputs, 2);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    auto* sparseLoop = scheduled->rootLoop->children[0].get();
    ASSERT_NE(sparseLoop, nullptr);
    EXPECT_EQ(sparseLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(sparseLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(sparseLoop->iterator.beginExpr, "S->row_ptr[i]");
    EXPECT_EQ(sparseLoop->iterator.endExpr, "S->row_ptr[i + 1]");
    ASSERT_EQ(sparseLoop->children.size(), 1u);
    EXPECT_EQ(sparseLoop->children[0]->indexName, "k");
    EXPECT_FALSE(sparseLoop->preStmts.empty());
    EXPECT_FALSE(sparseLoop->postStmts.empty());
}

TEST(SemanticIRTest, LowerFirstComputationToScheduledOptimizedReturnsOptimizedCompute) {
    auto ast = makeSpMMProgram("CSR");
    auto scheduled = sparseir::lowerFirstComputationToScheduledOptimized(
        *ast, opt::OptConfig::blockingOnly(8));

    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_TRUE(scheduled->optimizations.blockingApplied);
    EXPECT_EQ(scheduled->optimizations.blockSize, 8);
}

TEST(SemanticIRTest, ScheduledBlockingAppliesDirectlyToCompute) {
    auto ast = makeSpAddProgram("CSR", "Dense");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);

    opt::applyOptimizations(*scheduled, opt::OptConfig::blockingOnly(16));

    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i_block");
    EXPECT_TRUE(scheduled->optimizations.blockingApplied);
    EXPECT_EQ(scheduled->optimizations.tiledIndex, "i");
}

TEST(SemanticIRTest, ScheduledInterchangeAppliesDirectlyToDenseSDDMM) {
    auto ast = makeSDDMMProgram("CSR", "Dense");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);

    opt::applyOptimizations(*scheduled, opt::OptConfig::interchangeOnly());

    ASSERT_NE(scheduled->rootLoop, nullptr);
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->indexName, "k");
    ASSERT_EQ(scheduled->rootLoop->children[0]->children.size(), 1u);
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->iterator.beginExpr, "S->row_ptr[i]");
    EXPECT_EQ(scheduled->rootLoop->children[0]->children[0]->iterator.endExpr, "S->row_ptr[i + 1]");
    EXPECT_TRUE(scheduled->optimizations.interchangeApplied);
}

TEST(SemanticIRTest, ScheduledProgramOptimizationsRecurseIntoComputes) {
    auto ast = makeSpMVProgram();
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto scheduledProgram = sparseir::scheduleProgram(*semanticProgram);
    ASSERT_NE(scheduledProgram, nullptr);

    opt::applyOptimizations(*scheduledProgram, opt::OptConfig::blockingOnly(8));

    auto* compute = findScheduledCompute(scheduledProgram->statements);
    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    EXPECT_EQ(compute->rootLoop->indexName, "i_block");
    EXPECT_TRUE(compute->optimizations.blockingApplied);
}

// --- WI-5: Structural regression tests ---
// These tests assert on loop tree structure, not on which code path produced it,
// so they remain valid regardless of internal refactoring.

TEST(SemanticIRTest, StructuralRegressionSpMMNesting) {
    // SpMM must produce i(dense) → k(sparse) → j(dense) for CSR.
    auto ast = makeSpMMProgram("CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);

    // i(dense, root)
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    EXPECT_EQ(scheduled->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);

    // k(sparse, child of i)
    auto* kLoop = scheduled->rootLoop->children[0].get();
    EXPECT_EQ(kLoop->indexName, "k");
    EXPECT_EQ(kLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(kLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(kLoop->iterator.beginExpr, "A->row_ptr[i]");
    EXPECT_EQ(kLoop->iterator.endExpr, "A->row_ptr[i + 1]");
    ASSERT_EQ(kLoop->children.size(), 1u);

    // j(dense, child of k) with accumulation body
    auto* jLoop = kLoop->children[0].get();
    EXPECT_EQ(jLoop->indexName, "j");
    EXPECT_EQ(jLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_FALSE(jLoop->postStmts.empty());
}

TEST(SemanticIRTest, StructuralRegressionSDDMMScalarAccumulator) {
    // Sparse-output SDDMM: sparse loop has sum decl (preStmts) + final write
    // (postStmts), and its reduction child has sum accumulation (postStmts).
    auto ast = makeSDDMMProgram("CSR", "CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);
    ASSERT_NE(scheduled->rootLoop, nullptr);

    // i(dense, root)
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);

    // j(sparse, output-driven) with sum decl + final write
    auto* sparseLoop = scheduled->rootLoop->children[0].get();
    EXPECT_EQ(sparseLoop->indexName, "j");
    EXPECT_EQ(sparseLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_FALSE(sparseLoop->preStmts.empty());   // sum = 0.0
    EXPECT_FALSE(sparseLoop->postStmts.empty());  // output = sparse_vals * sum
    ASSERT_EQ(sparseLoop->children.size(), 1u);

    // k(dense, reduction child) with sum accumulation
    auto* reductionLoop = sparseLoop->children[0].get();
    EXPECT_EQ(reductionLoop->indexName, "k");
    EXPECT_FALSE(reductionLoop->postStmts.empty());  // sum += ...
}

TEST(SemanticIRTest, StructuralRegressionSpGEMMWorkspaceAccumulator) {
    // SpGEMM: workspace path is fully structured in scheduled IR.
    auto ast = makeSpGEMMProgram("CSR", "CSR");
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    auto* compute = findSemanticCompute(semanticProgram->statements);
    ASSERT_NE(compute, nullptr);

    auto scheduled = sparseir::scheduleCompute(*compute);
    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);

    // Prologue/epilogue for workspace
    EXPECT_FALSE(scheduled->prologueStmts.empty());
    EXPECT_FALSE(scheduled->epilogueStmts.empty());
    EXPECT_NE(dynamic_cast<ir::IRVarDecl*>(scheduled->prologueStmts.front().get()), nullptr);
    EXPECT_NE(dynamic_cast<ir::IRFreeStmt*>(scheduled->epilogueStmts.front().get()), nullptr);

    // i(dense, outer, workspace accumulator) with touched_count + gather/clear
    ASSERT_NE(scheduled->rootLoop, nullptr);
    EXPECT_EQ(scheduled->rootLoop->indexName, "i");
    EXPECT_FALSE(scheduled->rootLoop->preStmts.empty());   // touched_count
    EXPECT_FALSE(scheduled->rootLoop->postStmts.empty());  // gather-and-clear
    EXPECT_NE(dynamic_cast<ir::IRVarDecl*>(scheduled->rootLoop->preStmts.front().get()), nullptr);
    EXPECT_NE(dynamic_cast<ir::IRForStmt*>(scheduled->rootLoop->postStmts[0].get()), nullptr);
    EXPECT_NE(dynamic_cast<ir::IRForStmt*>(scheduled->rootLoop->postStmts[1].get()), nullptr);

    // k(sparse:A) → j(sparse:B) with structured hash-update body
    ASSERT_EQ(scheduled->rootLoop->children.size(), 1u);
    auto* kLoop = scheduled->rootLoop->children[0].get();
    EXPECT_EQ(kLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(kLoop->iterator.beginExpr, "A->row_ptr[i]");
    EXPECT_EQ(kLoop->iterator.endExpr, "A->row_ptr[i + 1]");
    ASSERT_EQ(kLoop->children.size(), 1u);
    auto* jLoop = kLoop->children[0].get();
    EXPECT_EQ(jLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(jLoop->iterator.beginExpr, "B->row_ptr[k]");
    EXPECT_EQ(jLoop->iterator.endExpr, "B->row_ptr[k + 1]");
    ASSERT_EQ(jLoop->postStmts.size(), 2u);
    EXPECT_NE(dynamic_cast<ir::IRIfStmt*>(jLoop->postStmts[0].get()), nullptr);
    EXPECT_NE(dynamic_cast<ir::IRAssign*>(jLoop->postStmts[1].get()), nullptr);
    EXPECT_FALSE(loopTreeContainsRawStmt(*scheduled->rootLoop));
}

TEST(SemanticIRTest, MixedProgramWithMultipleComputeShapes) {
    // Program with a region containing SpMV + a bare SpMM.
    // Verify both lower through scheduleCompute and produce valid loop trees.
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"8"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"6"}));
    program->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"8", "4"}));
    program->addStatement(std::make_unique<Declaration>(
        "B", "Dense", std::vector<std::string>{"6", "4"}));

    // SpMV inside a region
    {
        auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
        auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
        auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
        auto compute = std::make_unique<Computation>(
            std::move(lhs),
            std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess)));
        auto region = std::make_unique<ForStatement>(
            std::vector<std::string>{"A", "y"},
            std::vector<std::string>{"i"});
        region->addStatement(std::move(compute));
        program->addStatement(std::move(region));
    }

    // SpMM bare
    {
        auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
        auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
        auto bAccess = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
        program->addStatement(std::make_unique<Computation>(
            std::move(lhs),
            std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(bAccess))));
    }

    auto semanticProgram = sparseir::lowerToSemanticProgram(*program);
    ASSERT_NE(semanticProgram, nullptr);
    auto scheduledProgram = sparseir::scheduleProgram(*semanticProgram);
    ASSERT_NE(scheduledProgram, nullptr);

    // Find both computes
    int computeCount = 0;
    std::function<void(const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>&)> countComputes;
    countComputes = [&](const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>& stmts) {
        for (const auto& stmt : stmts) {
            if (auto* c = dynamic_cast<sparseir::scheduled::Compute*>(stmt.get())) {
                EXPECT_TRUE(c->fullyLowered) << "Compute #" << computeCount << " not fully lowered";
                EXPECT_NE(c->rootLoop, nullptr) << "Compute #" << computeCount << " has no root loop";
                computeCount++;
            }
            if (auto* r = dynamic_cast<sparseir::scheduled::Region*>(stmt.get())) {
                countComputes(r->body);
            }
        }
    };
    countComputes(scheduledProgram->statements);
    EXPECT_EQ(computeCount, 2);
}

TEST(SemanticIRTest, ProgramSchedulingThreeShapesAllFullyLowered) {
    // Three different compute shapes: SpMV, SpMM, SpAdd.
    // Verifies scheduleProgram applies compute scheduling per-statement.
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"8"}));
    program->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"6"}));
    program->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"8", "4"}));
    program->addStatement(std::make_unique<Declaration>(
        "B", "Dense", std::vector<std::string>{"6", "4"}));
    program->addStatement(std::make_unique<Declaration>(
        "D", "Dense", std::vector<std::string>{"8", "6"}));
    program->addStatement(std::make_unique<Declaration>(
        "E", "CSR", std::vector<std::string>{"8", "6"}));

    // SpMV: y[i] = A[i,j] * x[j]
    {
        auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
        auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
        auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
        program->addStatement(std::make_unique<Computation>(
            std::move(lhs),
            std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess))));
    }
    // SpMM: C[i,j] = A[i,k] * B[k,j]
    {
        auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
        auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
        auto bAccess = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
        program->addStatement(std::make_unique<Computation>(
            std::move(lhs),
            std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(bAccess))));
    }
    // SpAdd: D[i,j] = A[i,j] + E[i,j]
    {
        auto lhs = std::make_unique<TensorAccess>("D", std::vector<std::string>{"i", "j"});
        auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
        auto eAccess = std::make_unique<TensorAccess>("E", std::vector<std::string>{"i", "j"});
        program->addStatement(std::make_unique<Computation>(
            std::move(lhs),
            std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(aAccess), std::move(eAccess))));
    }

    auto semanticProgram = sparseir::lowerToSemanticProgram(*program);
    ASSERT_NE(semanticProgram, nullptr);
    auto scheduledProgram = sparseir::scheduleProgram(*semanticProgram);
    ASSERT_NE(scheduledProgram, nullptr);

    int computeCount = 0;
    std::function<void(const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>&)> checkComputes;
    checkComputes = [&](const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>& stmts) {
        for (const auto& stmt : stmts) {
            if (auto* c = dynamic_cast<sparseir::scheduled::Compute*>(stmt.get())) {
                EXPECT_TRUE(c->fullyLowered)
                    << "Compute #" << computeCount << " not fully lowered";
                EXPECT_NE(c->rootLoop, nullptr)
                    << "Compute #" << computeCount << " has no root loop";
                computeCount++;
            }
            if (auto* r = dynamic_cast<sparseir::scheduled::Region*>(stmt.get())) {
                checkComputes(r->body);
            }
        }
    };
    checkComputes(scheduledProgram->statements);
    EXPECT_EQ(computeCount, 3);
}
