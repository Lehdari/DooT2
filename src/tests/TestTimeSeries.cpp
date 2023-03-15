//
// Project: DooT2
// File: TestTimeSeries.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include <gtest/gtest.h>

#include "util/TimeSeries.hpp"


TEST(TestTimeSeries, TestFundamentals)
{
    ASSERT_EQ(TimeSeries::getNumInstances(), 0);

    TimeSeries timeSeries1;
    ASSERT_EQ(TimeSeries::getNumInstances(), 1);
    ASSERT_EQ(timeSeries1.getNumSeries(), 0);
    timeSeries1.addSeries<double>("loss");
    timeSeries1.addSeries<int64_t>("timestamp", -1); // -1 is default value
    ASSERT_EQ(timeSeries1.getNumSeries(), 2);

    // Test addEntry and getSeries
    timeSeries1.addEntry<double>("loss", 1.1);

    auto* lossVector = timeSeries1.getSeriesVector<double>("loss"); // returns std::vector<double>*
    auto* timestampVector = timeSeries1.getSeriesVector<int64_t>("timestamp"); // returns std::vector<int64_t>*

    ASSERT_NE(lossVector, nullptr);
    ASSERT_NE(timestampVector, nullptr);
    ASSERT_EQ(lossVector->size(), timestampVector->size());
    ASSERT_EQ(timestampVector->at(0), -1);

    // Test addEntries
    timeSeries1.addEntries("loss", 1.2, "timestamp", 8726l, "another_series", std::pair<int, float>(1337, 3.4f));
    timeSeries1.addEntries("loss", 1.3, "another_series", std::pair<int, float>(420, 5.6f));

    auto* anotherSeriesVector = timeSeries1.getSeriesVector<std::pair<int, float>>("another_series");

    ASSERT_NE(anotherSeriesVector, nullptr);
    ASSERT_EQ(lossVector->size(), 3);
    ASSERT_EQ(lossVector->size(), timestampVector->size());
    ASSERT_EQ(lossVector->size(), anotherSeriesVector->size());
    ASSERT_NEAR(lossVector->at(1), 1.2, 1.0e-8);
    ASSERT_EQ(timestampVector->at(1), 8726l);
    ASSERT_EQ(anotherSeriesVector->at(1).first, 1337);
    ASSERT_NEAR(anotherSeriesVector->at(1).second, 3.4f, 1.0e-8);
    ASSERT_NEAR(lossVector->at(2), 1.3, 1.0e-8);
    ASSERT_EQ(timestampVector->at(2), -1);
    ASSERT_EQ(anotherSeriesVector->at(2).first, 420);
    ASSERT_NEAR(anotherSeriesVector->at(2).second, 5.6f, 1.0e-8);



    // Test multiple instances
    TimeSeries timeSeries2;
    ASSERT_EQ(TimeSeries::getNumInstances(), 2);
    ASSERT_EQ(timeSeries2.getNumSeries(), 0);
    timeSeries2.addEntries("money", 123.2, "stonks", 543);
    ASSERT_EQ(timeSeries2.getNumSeries(), 2);

    // Test getSeriesNames
    {
        auto names1 = timeSeries1.getSeriesNames();
        ASSERT_NE(std::find(names1.begin(), names1.end(), "loss"), names1.end());
        ASSERT_NE(std::find(names1.begin(), names1.end(), "timestamp"), names1.end());
        ASSERT_NE(std::find(names1.begin(), names1.end(), "another_series"), names1.end());
        ASSERT_EQ(std::find(names1.begin(), names1.end(), "money"), names1.end());
        ASSERT_EQ(std::find(names1.begin(), names1.end(), "stonks"), names1.end());
        auto names2 = timeSeries2.getSeriesNames();
        ASSERT_EQ(std::find(names2.begin(), names2.end(), "loss"), names2.end());
        ASSERT_EQ(std::find(names2.begin(), names2.end(), "timestamp"), names2.end());
        ASSERT_EQ(std::find(names2.begin(), names2.end(), "another_series"), names2.end());
        ASSERT_NE(std::find(names2.begin(), names2.end(), "money"), names2.end());
        ASSERT_NE(std::find(names2.begin(), names2.end(), "stonks"), names2.end());
    }

    // Test copy constructor and assignment (also destructor)
    {
        TimeSeries timeSeries1b(timeSeries1);
        ASSERT_EQ(TimeSeries::getNumInstances(), 3);
        ASSERT_EQ(timeSeries1b.getNumSeries(), 3);
        auto names1b = timeSeries1b.getSeriesNames();
        ASSERT_NE(std::find(names1b.begin(), names1b.end(), "loss"), names1b.end());
        ASSERT_NE(std::find(names1b.begin(), names1b.end(), "timestamp"), names1b.end());
        ASSERT_NE(std::find(names1b.begin(), names1b.end(), "another_series"), names1b.end());
        ASSERT_EQ(std::find(names1b.begin(), names1b.end(), "money"), names1b.end());
        ASSERT_EQ(std::find(names1b.begin(), names1b.end(), "stonks"), names1b.end());

        TimeSeries timeSeries1c;
        timeSeries1c = timeSeries1;
        ASSERT_EQ(TimeSeries::getNumInstances(), 4);
        ASSERT_EQ(timeSeries1c.getNumSeries(), 3);
        auto names1c = timeSeries1c.getSeriesNames();
        ASSERT_NE(std::find(names1c.begin(), names1c.end(), "loss"), names1c.end());
        ASSERT_NE(std::find(names1c.begin(), names1c.end(), "timestamp"), names1c.end());
        ASSERT_NE(std::find(names1c.begin(), names1c.end(), "another_series"), names1c.end());
        ASSERT_EQ(std::find(names1c.begin(), names1c.end(), "money"), names1c.end());
        ASSERT_EQ(std::find(names1c.begin(), names1c.end(), "stonks"), names1c.end());

        timeSeries1b.addEntry("loss", 123.4);
        ASSERT_EQ(timeSeries1b.length(), 4);
        ASSERT_EQ(timeSeries1.length(), 3);
        ASSERT_EQ(timeSeries1c.length(), 3);
    }
    ASSERT_EQ(TimeSeries::getNumInstances(), 2);

    // Test move constructor and assignment (also destructor)
    {
        TimeSeries timeSeries1b(std::move(timeSeries1));
        ASSERT_EQ(TimeSeries::getNumInstances(), 2);
        ASSERT_EQ(timeSeries1b.getNumSeries(), 3);
        auto names1b = timeSeries1b.getSeriesNames();
        ASSERT_NE(std::find(names1b.begin(), names1b.end(), "loss"), names1b.end());
        ASSERT_NE(std::find(names1b.begin(), names1b.end(), "timestamp"), names1b.end());
        ASSERT_NE(std::find(names1b.begin(), names1b.end(), "another_series"), names1b.end());
        ASSERT_EQ(std::find(names1b.begin(), names1b.end(), "money"), names1b.end());
        ASSERT_EQ(std::find(names1b.begin(), names1b.end(), "stonks"), names1b.end());

        timeSeries1b.addEntry("loss", 123.4);
        ASSERT_EQ(timeSeries1b.length(), 4);

        TimeSeries timeSeries1c;
        ASSERT_EQ(TimeSeries::getNumInstances(), 3);
        timeSeries1c = std::move(timeSeries1b);
        ASSERT_EQ(TimeSeries::getNumInstances(), 2);
        ASSERT_EQ(timeSeries1c.getNumSeries(), 3);
        ASSERT_EQ(timeSeries1c.length(), 4);
        auto names1c = timeSeries1c.getSeriesNames();
        ASSERT_NE(std::find(names1c.begin(), names1c.end(), "loss"), names1c.end());
        ASSERT_NE(std::find(names1c.begin(), names1c.end(), "timestamp"), names1c.end());
        ASSERT_NE(std::find(names1c.begin(), names1c.end(), "another_series"), names1c.end());
        ASSERT_EQ(std::find(names1c.begin(), names1c.end(), "money"), names1c.end());
        ASSERT_EQ(std::find(names1c.begin(), names1c.end(), "stonks"), names1c.end());
    }
    ASSERT_EQ(TimeSeries::getNumInstances(), 1);
}

TEST(TestTimeSeries, TestSerialization)
{
    // TODO
#if 0
    // Test serialization
    auto json = timeSeries1.toJson();
    std::cout << json << std::endl;
    {
        std::ofstream f("TimeSeriesTest.json");
        f << json;
    }
#endif
}