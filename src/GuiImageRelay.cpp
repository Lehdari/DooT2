//
// Project: DooT2
// File: GuiImageRelay.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "GuiImageRelay.hpp"
#include "SingleBuffer.hpp"

#include "imgui.h"


GuiImageRelay::GuiImageRelay(SingleBuffer<Image<float>>* imageBuffer) :
    _imageBuffer    (imageBuffer)
{
}

void GuiImageRelay::render()
{
    if (_imageBuffer == nullptr)
        throw std::runtime_error("BUG: trying to render GuiImageRelay instance with imageBuffer == nullptr");

    auto imageHandle = _imageBuffer->read();
    auto* renderImage = imageHandle.get();

    // Conversion might be required
    if (imageHandle->format() != ImageFormat::BGRA) {
        convertImage(*imageHandle, _image, ImageFormat::BGRA);
        renderImage = &_image;
    }

    // Update texture
    _texture.updateFromBuffer(renderImage->data(), GL_BGRA, renderImage->width(), renderImage->height());

    // Render
    ImGui::Image((void*)(intptr_t)_texture.id(),
        ImVec2(_texture.width(), _texture.height()),
        ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
}
