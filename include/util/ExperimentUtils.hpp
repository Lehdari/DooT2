//
// Project: DooT2
// File: ExperimentName.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <filesystem>


std::string formatExperimentName(const std::string& name, const nlohmann::json& modelConfig);

// Convert a string to an experiment root path.
// Relative paths are considered to be relative to the experiments directory (doot2::experimentsDirectory)
std::filesystem::path experimentRootFromString(const std::string& experimentRoot);

std::vector<nlohmann::json> flattenGridSearchParameters(const nlohmann::json& json);