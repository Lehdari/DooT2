//
// Project: DooT2
// File: image.inl
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

inline int getImageFormatNChannels(ImageFormat imageFormat)
{
    if (imageFormat == ImageFormat::BGRA)
        return 4;

    return -1;
}


template<typename T_Data>
Image<T_Data>::Image(int width, int height, ImageFormat format, const T_Data* data) :
    _width  (width),
    _height (height),
    _format (format),
    _data   (width*height*getImageFormatNChannels(ImageFormat::BGRA))
{
    for (size_t i=0; i<_data.size(); ++i) {
        _data[i] = data[i];
    }
}

template<typename T_Data>
int Image<T_Data>::width() const noexcept
{
    return _width;
}

template<typename T_Data>
int Image<T_Data>::height() const noexcept
{
    return _height;
}

template<typename T_Data>
const ImageFormat& Image<T_Data>::format() const noexcept
{
    return _format;
}

template<typename T_Data>
const T_Data* Image<T_Data>::data() const noexcept
{
    return _data.data();
}


template <typename T_DataSrc, typename T_DataDest>
inline void convertImage(const Image<T_DataSrc>& srcImage, Image<T_DataDest>& destImage)
{
    destImage._data.resize(srcImage._data.size());
    for (int i=0; i<srcImage._data.size(); ++i) {
        destImage._data[i] = static_cast<T_DataDest>(srcImage._data[i]);
    }
}

template <>
inline void convertImage<uint8_t, float>(const Image<uint8_t>& srcImage, Image<float>& destImage)
{
    constexpr float convertRatio = 1.0f / 255.0f;
    destImage._data.resize(srcImage._data.size());
    for (int i=0; i<srcImage._data.size(); ++i) {
        destImage._data[i] = static_cast<float>(srcImage._data[i]) * convertRatio;
    }
}

template <>
inline void convertImage<float, uint8_t>(const Image<float>& srcImage, Image<uint8_t>& destImage)
{
    constexpr float convertRatio = 255.0f;
    destImage._data.resize(srcImage._data.size());
    for (int i=0; i<srcImage._data.size(); ++i) {
        destImage._data[i] = static_cast<uint8_t>(srcImage._data[i] * convertRatio);
    }
}
