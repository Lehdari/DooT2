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


#include "Utils.hpp"

#include <cstdint>
#include <vector>


enum class ImageFormat {
    BGRA // only BGRA supported for now
};

inline int getImageFormatNChannels(ImageFormat imageFormat);


template <typename T_Data>
class Image {
public:
    // If data is nullptr, internal buffer will be used. Otherwise, the buffer pointed by data will
    // be utilized as the pixel data buffer. Ownership will not be transferred.
    Image(int width=0, int height=0, ImageFormat format=ImageFormat::BGRA, T_Data* data=nullptr);
    Image(const Image<T_Data>& other);
    Image(Image&&) noexcept = default;
    Image& operator=(const Image<T_Data>& other);
    Image& operator=(Image&&) noexcept = default;

    int width() const noexcept;
    int height() const noexcept;
    const ImageFormat& format() const noexcept;
    const T_Data* data() const noexcept;

    // Set pixel data (will read width * height * nchannels * sizeof(T_Data) bytes from data)
    void copyFrom(const T_Data* data);

    template <typename T_DataOther>
    friend class Image;
    template <typename T_DataSrc, typename T_DataDest>
    friend void convertImage(const Image<T_DataSrc>&, Image<T_DataDest>&);

private:
    int                 _width;
    int                 _height;
    ImageFormat         _format;

    T_Data*             _data;
    size_t              _nElements;
    std::vector<T_Data> _buffer;

    template <typename T_DataOther>
    void copyParamsFrom(const Image<T_DataOther>& other);
};


template <typename T_DataSrc, typename T_DataDest>
inline void convertImage(const Image<T_DataSrc>& srcImage, Image<T_DataDest>& destImage);


#include "Image.inl"
