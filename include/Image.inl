//
// Project: DooT2
// File: image.inl
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

template<typename T_Data>
Image<T_Data>::Image(int width, int height, ImageFormat format, const T_Data* data) :
    _width  (width),
    _height (height),
    _format (format),
    _data   (width*height)
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
