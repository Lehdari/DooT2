//
// Project: DooT2
// File: TestSequenceStorage.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include <gtest/gtest.h>

#include "util/SequenceStorage.hpp"


TEST(TestSequenceStorage, TestBasics)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");

    SequenceStorage ss(2); // batch size 2
    ASSERT_EQ(ss.getNumSequences(), 0);

    // Test sequence addition
    ss.addSequence<float>("a", 1.0f, {1, 2, 3});
    ASSERT_EQ(ss.getNumSequences(), 1);

    // Test entry addition
    ss.addEntry("a", 2.0f); // batch with all same values
    ASSERT_EQ(ss.getSequence<float>("a")->length(), 1);
    {   // Check contents
        auto* s1 = ss.getSequence<float>("a");
        ASSERT_NE(s1, nullptr);
        auto& ts1 = s1->tensor();
        ASSERT_TRUE((ts1.sizes() == std::vector<int64_t>{1,2,1,2,3}));
        for (int i = 0; i < 12; ++i) {
            ASSERT_NEAR(ts1.data_ptr<float>()[i], 2.0f, 1.0e-6f);
        }
    }
    torch::Tensor t1 = torch::zeros({2,1,2,3}, torch::kFloat32);
    for (int i=0; i<12; ++i)
        t1.data_ptr<float>()[i] = (float)i;
    ss.addEntry("a", t1); // batch from a tensor
    {   // Check contents
        auto* s1 = ss.getSequence<float>("a");
        ASSERT_NE(s1, nullptr);
        auto& ts1 = s1->tensor();
        ASSERT_TRUE((ts1.sizes() == std::vector<int64_t>{2,2,1,2,3}));
        for (int i = 0; i < 12; ++i) {
            ASSERT_NEAR(ts1.data_ptr<float>()[i], 2.0f, 1.0e-6f);
        }
        for (int i = 12; i < 24; ++i) {
            ASSERT_NEAR(ts1.data_ptr<float>()[i], (float)(i-12), 1.0e-6f);
        }
    }

    // Test fetching the sequence with a wrong type (should fail to assert)
    ASSERT_DEATH({
        ss.getSequence<int>("a");
    }, "");

    ss.addSequence<int64_t>("b", 3, {3, 2, 1});
    ASSERT_EQ(ss.getNumSequences(), 2);
    ASSERT_EQ(ss.getSequence<int64_t>("b")->length(), 2);

    ss.addEntry("b", (int64_t)3);

    // TODO test addEntries
    // TODO test resize
    // TODO test batch handle access (getBatch)
}

// TODO test RO5 (new test)