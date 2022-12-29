//
// Project: DooT2
// File: Image.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


#include <cstdint>
#include <vector>


enum class ImageFormat {
    BGRA // only BGRA supported for now
};

inline int getImageFormatNChannels(ImageFormat imageFormat);


template <typename T_Data>
class Image {
public:
    Image(int width=0, int height=0, ImageFormat format=ImageFormat::BGRA, const T_Data* data=nullptr);
    Image(const Image& image) = default;
    Image(Image&& image) noexcept = default;
    Image& operator=(const Image& image) = default;
    Image& operator=(Image&& image) noexcept = default;

    int width() const noexcept;
    int height() const noexcept;
    const ImageFormat& format() const noexcept;
    const T_Data* data() const noexcept;

    template <typename T_DataSrc, typename T_DataDest>
    friend void convertImage(const Image<T_DataSrc>&, Image<T_DataDest>&);

private:
    int                 _width;
    int                 _height;
    ImageFormat         _format;
    std::vector<T_Data> _data;
};


template <typename T_DataSrc, typename T_DataDest>
inline void convertImage(const Image<T_DataSrc>& srcImage, Image<T_DataDest>& destImage);


#include "Image.inl"
