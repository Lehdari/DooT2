//
// Project: DooT2
// File: Utils.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <filesystem>


#ifndef ASSETS_DIR
    #error ASSETS_DIR not defined
#endif


const std::filesystem::path assetsDir(ASSETS_DIR);
