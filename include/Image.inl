//
// Project: DooT2
// File: image.inl
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

INLINE int getImageFormatNChannels(ImageFormat imageFormat)
{
    if (imageFormat == ImageFormat::BGRA)
        return 4;

    return -1;
}


template<typename T_Data>
Image<T_Data>::Image(int width, int height, ImageFormat format, T_Data* data) :
    _width      (width),
    _height     (height),
    _format     (format),
    _data       (data),
    _nElements  (_width*_height*getImageFormatNChannels(format))
{
    if (_data == nullptr) {
        _buffer.resize(_nElements);
        _data = _buffer.data();
    }
}

template<typename T_Data>
Image<T_Data>::Image(const Image<T_Data>& other) :
    _width      (other._width),
    _height     (other._height),
    _format     (other._format),
    _data       (nullptr),
    _nElements  (other._nElements), // allocate a new, internal buffer
    _buffer     (_nElements)
{
    _data = _buffer.data();
    memcpy(_data, other._data, _nElements*sizeof(T_Data)); // make a copy of the pixel data
}

template<typename T_Data>
Image<T_Data>& Image<T_Data>::operator=(const Image<T_Data>& other)
{
    if (this == &other)
        return *this;

    copyParamsFrom(other);

    return *this;
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
    return _data;
}

template<typename T_Data>
void Image<T_Data>::copyFrom(const T_Data* data)
{
    memcpy(_buffer.data(), data, _nElements * sizeof(T_Data));
}

template<typename T_Data>
template<typename T_DataOther>
void Image<T_Data>::copyParamsFrom(const Image<T_DataOther>& other)
{
    _width = other._width;
    _height = other._height;
    _format = other._format;
    _nElements = other._nElements;
    _buffer.resize(_nElements);
    _data = _buffer.data();
}

template <typename T_DataSrc, typename T_DataDest>
INLINE void convertImage(const Image<T_DataSrc>& srcImage, Image<T_DataDest>& destImage)
{
    if (destImage._data == destImage._buffer.data()) {
        destImage.copyParamsFrom(srcImage);
    }
    else if (srcImage._nElements != destImage._nElements) {
        throw std::runtime_error("N. of buffer elements is required to match when using external pixel buffer");
    }

    for (int i=0; i<srcImage._nElements; ++i) {
        destImage._data[i] = static_cast<T_DataDest>(srcImage._data[i]);
    }
}

template <>
INLINE void convertImage<uint8_t, float>(const Image<uint8_t>& srcImage, Image<float>& destImage)
{
    if (destImage._data == destImage._buffer.data()) {
        destImage.copyParamsFrom(srcImage);
    }
    else if (srcImage._nElements != destImage._nElements) {
        throw std::runtime_error("N. of buffer elements is required to match when using external pixel buffer");
    }

    constexpr float convertRatio = 1.0f / 255.0f;
    for (int i=0; i<srcImage._nElements; ++i) {
        destImage._data[i] = static_cast<float>(srcImage._data[i]) * convertRatio;
    }
}

template <>
INLINE void convertImage<float, uint8_t>(const Image<float>& srcImage, Image<uint8_t>& destImage)
{
    if (destImage._data == destImage._buffer.data()) {
        destImage.copyParamsFrom(srcImage);
    }
    else if (srcImage._nElements != destImage._nElements) {
        throw std::runtime_error("N. of buffer elements is required to match when using external pixel buffer");
    }

    constexpr float convertRatio = 255.0f;
    for (int i=0; i<srcImage._nElements; ++i) {
        destImage._data[i] = static_cast<uint8_t>(srcImage._data[i] * convertRatio);
    }
}
