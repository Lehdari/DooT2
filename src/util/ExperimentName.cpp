//
// Project: DooT2
// File: ExperimentName.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "util/ExperimentName.hpp"
#include "gui/State.hpp"

#include <sstream>
#include <regex>


std::string formatExperimentName(const gui::State& guiState)
{
    std::string experimentName(guiState.experimentName);

    // Replace {time} with GMT timestamp
    std::string timestamp = [](){
        using namespace std::chrono;
        std::stringstream ss;
        auto now = system_clock::to_time_t(system_clock::now());
        ss << std::put_time(std::gmtime(&now), "%Y%m%dT%H%M%S");
        return ss.str();
    }();
    experimentName = std::regex_replace(experimentName, std::regex("\\{time}"), timestamp);

    return experimentName;
}
