//
// Project: DooT2
// File: TestSequence.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include <gtest/gtest.h>

#include "util/Sequence.hpp"


TEST(TestSequence, TestBasics)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");

    Sequence<float> s1(2, {1,2,3});
    ASSERT_EQ(s1.entryShape()[0], 1);
    ASSERT_EQ(s1.entryShape()[1], 2);
    ASSERT_EQ(s1.entryShape()[2], 3);

    // Test adding batches
    torch::Tensor t1 = torch::zeros({2,1,2,3}, torch::kFloat32);
    torch::Tensor t2 = torch::zeros({2,1,2,3}, torch::kFloat32);
    for (int i=0; i<12; ++i) {
        t1.data_ptr<float>()[i] = (float)i;
        t2.data_ptr<float>()[i] = (float)i+12;
    }

    ASSERT_EQ(s1.length(), 0);
    s1.addBatch(t1);
    ASSERT_EQ(s1.length(), 1);
    s1.addBatch(t2);
    ASSERT_EQ(s1.length(), 2);
    s1.addBatch(-32.0f);
    ASSERT_EQ(s1.length(), 3);

    // Check buffer contents
    for (int i=0; i<24; ++i) {
        ASSERT_NEAR(s1.buffer()[i], (float)i, 1.0e-6f);
    }
    for (int i=24; i<36; ++i) {
        ASSERT_NEAR(s1.buffer()[i], -32.0f, 1.0e-6f);
    }

    // Check tensor mapping
    torch::Tensor map1 = s1.tensor();
    ASSERT_TRUE((map1.sizes() == std::vector<int64_t>{3,2,1,2,3}));

    // Test batch addition failures
    ASSERT_THROW({
        torch::Tensor t3 = torch::zeros({1,1,2,3}, torch::kFloat32); // wrong shape
        s1.addBatch(t3);
    }, std::runtime_error);
    ASSERT_DEATH({
        torch::Tensor t3 = torch::zeros({2,1,2,3}, torch::kFloat64); // wrong type
        s1.addBatch(t3);
    }, "");

    // Test resize
    s1.resize(5);
    ASSERT_EQ(s1.length(), 5);
    for (int i=36; i<60; ++i) {
        ASSERT_NEAR(s1.buffer()[i], 0.0f, 1.0e-6f);
    }

    s1.resize(7, 10.0f);
    ASSERT_EQ(s1.length(), 7);
    for (int i=60; i<84; ++i) {
        ASSERT_NEAR(s1.buffer()[i], 10.0f, 1.0e-6f);
    }

    torch::Tensor t3 = torch::ones({2,1,2,3}, torch::kFloat32);
    s1.resize(9, t3);
    ASSERT_EQ(s1.length(), 9);
    for (int i=84; i<108; ++i) {
        ASSERT_NEAR(s1.buffer()[i], 1.0f, 1.0e-6f);
    }
}

TEST(TestSequence, TestIterators)
{
    Sequence<float> s1(2, {1,2,3});

    for (int j=0; j<10; ++j) {
        torch::Tensor t = torch::zeros({2,1,2,3}, torch::kFloat32);
        for (int i=0; i<12; ++i) {
            t.data_ptr<float>()[i] = (float)(j*12 + i);
        }
        s1.addBatch(t);
    }

    int i = 0;
    for (const auto& d : s1.buffer()) {
        ASSERT_NEAR(d, (float)i++, 1.0e-6);
    }

    torch::Tensor t1 = s1.begin()[0];
    ASSERT_TRUE((t1.sizes() == std::vector<int64_t>{1,2,3}));
    ASSERT_NEAR(t1.data_ptr<float>()[0], 0.0f, 1.0e-6f);
    ASSERT_NEAR(t1.data_ptr<float>()[5], 5.0f, 1.0e-6f);
    torch::Tensor t2 = s1.begin()[1];
    ASSERT_TRUE((t2.sizes() == std::vector<int64_t>{1,2,3}));
    ASSERT_NEAR(t2.data_ptr<float>()[0], 6.0f, 1.0e-6f);
    ASSERT_NEAR(t2.data_ptr<float>()[5], 11.0f, 1.0e-6f);
}

// TODO test batch / entry handles (read and write, all types)
