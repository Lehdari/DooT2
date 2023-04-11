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

#include <string>


namespace gui {
struct State;
} // namespace gui


std::string formatExperimentName(const gui::State& guiState); // TODO maybe move to utils?