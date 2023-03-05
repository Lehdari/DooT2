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


// Helper function for deciding which channel format to use for the image relay
static inline ImageFormat inferTargetFormat(ImageFormat imageBufferFormat)
{
    switch (imageBufferFormat) {
        case ImageFormat::GRAY:
            return ImageFormat::BGRA; // Actually just use BGRA since GRAY renders by only using the red channel
        default:
            return ImageFormat::BGRA;
    }

    throw std::runtime_error("BUG: Should not be reached");
}

// Helper function for ImageFormat -> GLEnum mapping
static inline GLenum toGLFormat(ImageFormat format)
{
    switch (format) {
        case ImageFormat::BGRA:
            return GL_BGRA;
        case ImageFormat::GRAY:
            return GL_RED;
        default:
            throw std::runtime_error("Unsupported target format");
    }

    throw std::runtime_error("BUG: Should not be reached");
}


GuiImageRelay::GuiImageRelay(SingleBuffer<Image<float>>* imageBuffer) :
    _imageBuffer    (imageBuffer),
    _targetFormat   (_imageBuffer == nullptr ? ImageFormat::BGRA : inferTargetFormat(_imageBuffer->read()->format())),
    _texture        (GL_TEXTURE_2D, _targetFormat == ImageFormat::GRAY ? GL_RED : GL_RGBA, GL_FLOAT)
{
    auto imageHandle = _imageBuffer->read();
    _texture.create(imageHandle->width(), imageHandle->height());
}

void GuiImageRelay::render()
{
    if (_imageBuffer == nullptr)
        throw std::runtime_error("BUG: trying to render GuiImageRelay instance with imageBuffer == nullptr");

    auto imageHandle = _imageBuffer->read();
    auto* renderImage = imageHandle.get();

    // Conversion might be required
    if (imageHandle->format() != _targetFormat) {
        convertImage(*imageHandle, _image, _targetFormat);
        renderImage = &_image;
    }

    // Update texture
    _texture.updateFromBuffer(renderImage->data(), toGLFormat(renderImage->format()),
        renderImage->width(), renderImage->height());

    // Render
    ImGui::Image((void*)(intptr_t)_texture.id(),
        ImVec2(_texture.width(), _texture.height()),
        ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
}
