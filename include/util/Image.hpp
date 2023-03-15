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
#include "MathTypes.hpp"

#include <cstdint>
#include <vector>
#include <type_traits>


enum class ImageFormat : uint32_t {
    BGRA_GAMMA  = 1,            // sRGB gamma correction
    BGRA        = BGRA_GAMMA,
    YUV         = 2,
    GRAY        = 3,
    UNCHANGED   = 0             // only allowed as an argument to convertImage
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

    // Convert to other format
    // Note: Only conversion to formats with equal amount of channels is permitted when
    // using external buffer
    void convertImageFormat(ImageFormat destFormat);

    template <typename T_DataOther>
    friend class Image;
    template <typename T_DataSrc, typename T_DataDest>
    friend void convertImage(const Image<T_DataSrc>&, Image<T_DataDest>&, ImageFormat);

private:
    int                 _width;
    int                 _height;
    ImageFormat         _format;

    T_Data*             _data;
    size_t              _nElements;
    std::vector<T_Data> _buffer;

    template <typename T_DataOther>
    void copyParamsFrom(const Image<T_DataOther>& other);

    INLINE bool usingExternalBuffer();

    // Format conversion machinery
    static void convertImageFormat(
        ImageFormat srcFormat, ImageFormat destFormat,
        const T_Data* srcBuffer, size_t nSrcBufferElements,
        T_Data* destBuffer, size_t nDestBufferElements);
    template <int T_NChannelsSrc>
    static inline void convertImageFormat_(
        ImageFormat srcFormat, ImageFormat destFormat,
        const T_Data* srcBuffer, size_t nSrcBufferElements,
        T_Data* destBuffer, size_t nDestBufferElements);
    template <int T_NChannelsSrc, int T_NChannelsDest>
    INLINE static void applyFormatConversion(
        const Eigen::Matrix<T_Data, T_NChannelsDest, T_NChannelsSrc>& matrix,
        const T_Data* srcBuffer, size_t nSrcBufferElements,
        T_Data* destBuffer, size_t nDestBufferElements);
};


template <typename T_DataSrc, typename T_DataDest>
inline void convertImage(
    const Image<T_DataSrc>& srcImage,
    Image<T_DataDest>& destImage,
    ImageFormat destFormat = ImageFormat::UNCHANGED);

// Helper function for getting a matrix for image format conversion
template <typename T_Data, int T_NChannelsSrc, int T_NChannelsDest>
inline Eigen::Matrix<T_Data, T_NChannelsDest, T_NChannelsSrc>
    getImageFormatConversionMatrix(ImageFormat srcFormat, ImageFormat destFormat);


#include "Image.inl"
