//
// Project: DooT2
// File: GuiImageRelay.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Image.hpp"

#include "gut_opengl/Texture.hpp"


template <typename T_Data> class SingleBuffer;


// Class for rendering images to ImGui from threaded sources
class GuiImageRelay {
public:
    // The constructor takes direct pointer handle to a buffer, meaning the user takes responsibility
    // that the lifetime of the relay does not exceed the lifetime of the buffer.
    GuiImageRelay(SingleBuffer<Image<float>>* imageBuffer = nullptr);

    void render();

private:
    SingleBuffer<Image<float>>* _imageBuffer;
    Image<float>                _image;         // used for potential conversions
    gut::Texture                _texture;
};