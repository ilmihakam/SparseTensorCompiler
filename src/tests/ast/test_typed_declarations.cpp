#include <gtest/gtest.h>
#include "ast.h"
#include <memory>
#include <vector>
#include <string>

using namespace SparseTensorCompiler;

class TypedDeclarationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================
// TYPE-ENFORCED SYNTAX TESTS (Type without Shape)
// ============================================

TEST_F(TypedDeclarationTest, DenseTypeNoShape) {
    auto decl = std::make_unique<Declaration>("A", "Dense");

    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");
    EXPECT_TRUE(decl->shape.empty());  // No shape specified
}

TEST_F(TypedDeclarationTest, CSRTypeNoShape) {
    auto decl = std::make_unique<Declaration>("B", "CSR");

    EXPECT_EQ(decl->tensorName, "B");
    EXPECT_EQ(decl->tensorType, "CSR");
    EXPECT_TRUE(decl->shape.empty());
}

TEST_F(TypedDeclarationTest, COOTypeNoShape) {
    auto decl = std::make_unique<Declaration>("C", "COO");

    EXPECT_EQ(decl->tensorName, "C");
    EXPECT_EQ(decl->tensorType, "COO");
}

TEST_F(TypedDeclarationTest, CSCTypeNoShape) {
    auto decl = std::make_unique<Declaration>("D", "CSC");

    EXPECT_EQ(decl->tensorName, "D");
    EXPECT_EQ(decl->tensorType, "CSC");
}

TEST_F(TypedDeclarationTest, ELLPACKTypeNoShape) {
    auto decl = std::make_unique<Declaration>("E", "ELLPACK");

    EXPECT_EQ(decl->tensorName, "E");
    EXPECT_EQ(decl->tensorType, "ELLPACK");
}

TEST_F(TypedDeclarationTest, DIATypeNoShape) {
    auto decl = std::make_unique<Declaration>("F", "DIA");

    EXPECT_EQ(decl->tensorName, "F");
    EXPECT_EQ(decl->tensorType, "DIA");
}

// ============================================
// TYPE-ENFORCED SYNTAX TESTS (Type with Shape)
// ============================================

TEST_F(TypedDeclarationTest, DenseWithShape1D) {
    auto decl = std::make_unique<Declaration>("A", "Dense", std::vector<std::string>{"10"});

    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");
    EXPECT_EQ(decl->shape.size(), 1);
    EXPECT_EQ(decl->shape[0], "10");
}

TEST_F(TypedDeclarationTest, DenseWithShape2D) {
    auto decl = std::make_unique<Declaration>("A", "Dense", std::vector<std::string>{"10", "20"});

    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");
    EXPECT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "10");
    EXPECT_EQ(decl->shape[1], "20");
}

TEST_F(TypedDeclarationTest, CSRWithShape2D) {
    auto decl = std::make_unique<Declaration>("B", "CSR", std::vector<std::string>{"100", "50"});

    EXPECT_EQ(decl->tensorName, "B");
    EXPECT_EQ(decl->tensorType, "CSR");
    EXPECT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "100");
    EXPECT_EQ(decl->shape[1], "50");
}

TEST_F(TypedDeclarationTest, COOWithShape3D) {
    auto decl = std::make_unique<Declaration>("C", "COO", std::vector<std::string>{"10", "20", "30"});

    EXPECT_EQ(decl->tensorName, "C");
    EXPECT_EQ(decl->tensorType, "COO");
    EXPECT_EQ(decl->shape.size(), 3);
    EXPECT_EQ(decl->shape[0], "10");
    EXPECT_EQ(decl->shape[1], "20");
    EXPECT_EQ(decl->shape[2], "30");
}

TEST_F(TypedDeclarationTest, CSCWithLargeShape) {
    auto decl = std::make_unique<Declaration>("D", "CSC", std::vector<std::string>{"1000", "2000"});

    EXPECT_EQ(decl->tensorName, "D");
    EXPECT_EQ(decl->tensorType, "CSC");
    EXPECT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "1000");
    EXPECT_EQ(decl->shape[1], "2000");
}

TEST_F(TypedDeclarationTest, ELLPACKWithShape) {
    auto decl = std::make_unique<Declaration>("E", "ELLPACK", std::vector<std::string>{"5", "8"});

    EXPECT_EQ(decl->tensorName, "E");
    EXPECT_EQ(decl->tensorType, "ELLPACK");
    EXPECT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "5");
    EXPECT_EQ(decl->shape[1], "8");
}

TEST_F(TypedDeclarationTest, DIAWithShape) {
    auto decl = std::make_unique<Declaration>("F", "DIA", std::vector<std::string>{"7", "7"});

    EXPECT_EQ(decl->tensorName, "F");
    EXPECT_EQ(decl->tensorType, "DIA");
    EXPECT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "7");
    EXPECT_EQ(decl->shape[1], "7");
}

// ============================================
// MIXED PROGRAM TESTS
// ============================================

TEST_F(TypedDeclarationTest, ProgramWithVariousTensorTypes) {
    auto program = std::make_unique<Program>();

    // Different tensor types
    program->addStatement(std::make_unique<Declaration>("A", "Dense", std::vector<std::string>{"10", "20"}));
    program->addStatement(std::make_unique<Declaration>("B", "CSR"));
    program->addStatement(std::make_unique<Declaration>("C", "COO", std::vector<std::string>{"10", "20"}));

    ASSERT_EQ(program->statements.size(), 3);

    // Check first declaration (Dense with shape)
    auto* decl1 = dynamic_cast<Declaration*>(program->statements[0].get());
    ASSERT_NE(decl1, nullptr);
    EXPECT_EQ(decl1->tensorName, "A");
    EXPECT_EQ(decl1->tensorType, "Dense");
    EXPECT_EQ(decl1->shape.size(), 2);

    // Check second declaration (CSR without shape)
    auto* decl2 = dynamic_cast<Declaration*>(program->statements[1].get());
    ASSERT_NE(decl2, nullptr);
    EXPECT_EQ(decl2->tensorName, "B");
    EXPECT_EQ(decl2->tensorType, "CSR");
    EXPECT_TRUE(decl2->shape.empty());

    // Check third declaration (COO with shape)
    auto* decl3 = dynamic_cast<Declaration*>(program->statements[2].get());
    ASSERT_NE(decl3, nullptr);
    EXPECT_EQ(decl3->tensorName, "C");
    EXPECT_EQ(decl3->tensorType, "COO");
    EXPECT_EQ(decl3->shape.size(), 2);
    EXPECT_EQ(decl3->shape[0], "10");
    EXPECT_EQ(decl3->shape[1], "20");
}

TEST_F(TypedDeclarationTest, CompleteTypedProgram) {
    auto program = std::make_unique<Program>();

    // Declarations with types and shapes
    program->addStatement(std::make_unique<Declaration>("A", "CSR", std::vector<std::string>{"100", "50"}));
    program->addStatement(std::make_unique<Declaration>("B", "Dense", std::vector<std::string>{"50", "20"}));
    program->addStatement(std::make_unique<Declaration>("C", "Dense", std::vector<std::string>{"100", "20"}));

    ASSERT_EQ(program->statements.size(), 3);

    // Verify all declarations have correct types and shapes
    for (size_t i = 0; i < program->statements.size(); ++i) {
        auto* decl = dynamic_cast<Declaration*>(program->statements[i].get());
        ASSERT_NE(decl, nullptr);
        EXPECT_FALSE(decl->tensorType.empty());
        EXPECT_EQ(decl->shape.size(), 2);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
