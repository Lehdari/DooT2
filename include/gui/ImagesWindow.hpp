//
// Project: DooT2
// File: ImagesWindow.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Window.hpp"


namespace gui {

class ImagesWindow : public Window {
public:
    virtual void render(Trainer* trainer, Model* model, gui::State* guiState) const override;
};

};
