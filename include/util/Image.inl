//
// Project: DooT2
// File: image.inl
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

constexpr int getImageFormatNChannels(ImageFormat imageFormat)
{
    switch (imageFormat) {
        case ImageFormat::BGRA:
            return 4;
        case ImageFormat::RGB:
            return 3;
        case ImageFormat::YUV:
            return 3;
        case ImageFormat::GRAY:
            return 1;
        case ImageFormat::UNCHANGED:
            return 0;
        default:
            throw std::runtime_error("BUG: Support for the image format not implemented!");
    }
    return -1;
}


template <typename T_Data>
Image<T_Data>::Image(int width, int height, ImageFormat format, T_Data* data) :
    _width      (width),
    _height     (height),
    _format     (format),
    _data       (data),
    _nElements  (_width*_height*getImageFormatNChannels(format))
{
    // Unchanged only allowed in convertImage
    if (_format == ImageFormat::UNCHANGED)
        throw std::runtime_error("Invalid image format");

    // Using internal buffer, allocate it
    if (_data == nullptr) {
        _buffer.resize(_nElements);
        _data = _buffer.data();
    }
}

template <typename T_Data>
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

template <typename T_Data>
Image<T_Data>& Image<T_Data>::operator=(const Image<T_Data>& other)
{
    if (this == &other)
        return *this;

    _width = other._width;
    _height = other._height;
    _format = other._format;
    if (_nElements != other._nElements) { // reuse the current buffer in case it's right size
        _nElements = other._nElements;
        _buffer.resize(_nElements);
        _data = _buffer.data();
    }
    memcpy(_data, other._data, _nElements*sizeof(T_Data)); // make a copy of the pixel data

    return *this;
}

template <typename T_Data>
int Image<T_Data>::width() const noexcept
{
    return _width;
}

template <typename T_Data>
int Image<T_Data>::height() const noexcept
{
    return _height;
}

template <typename T_Data>
const ImageFormat& Image<T_Data>::format() const noexcept
{
    return _format;
}

template <typename T_Data>
const T_Data* Image<T_Data>::data() const noexcept
{
    return _data;
}

template<typename T_Data>
T_Data* Image<T_Data>::operator()(int x, int y)
{
    return _data + (y*_width + x)*getImageFormatNChannels(_format);
}

template<typename T_Data>
const T_Data* Image<T_Data>::operator()(int x, int y) const
{
    return _data + (y*_width + x)*getImageFormatNChannels(_format);
}

template <typename T_Data>
void Image<T_Data>::copyFrom(const T_Data* data)
{
    memcpy(_buffer.data(), data, _nElements * sizeof(T_Data));
}

template <typename T_Data>
void Image<T_Data>::convertImageFormat(ImageFormat destFormat)
{
    if (destFormat == ImageFormat::UNCHANGED)
        return;

    if (usingExternalBuffer() && getImageFormatNChannels(destFormat) != getImageFormatNChannels(_format)) {
        throw std::runtime_error("Number of format channels need to match when using external buffer");
    }

    // New number of elements, new buffer
    size_t newNElements = _width*_height*getImageFormatNChannels(destFormat);
    std::vector<T_Data> destBuffer(newNElements);

    // Convert
    convertImageFormat(_format, destFormat, _data, _nElements, destBuffer.data(), newNElements);

    // Replace the old buffer
    if (usingExternalBuffer()) {
        memcpy(_data, destBuffer.data(), _nElements * sizeof(T_Data));
    }
    else {
        _nElements = newNElements;
        _buffer = std::move(destBuffer);
        _data = _buffer.data();
    }

    _format = destFormat;
}

template <typename T_Data>
template <typename T_DataOther>
void Image<T_Data>::copyParamsFrom(const Image<T_DataOther>& other)
{
    _width = other._width;
    _height = other._height;
    _format = other._format;
    _nElements = other._nElements;
    _buffer.resize(_nElements);
    _data = _buffer.data();
}

template <typename T_Data>
INLINE bool Image<T_Data>::usingExternalBuffer()
{
    return _data != _buffer.data();
}

template <typename T_Data>
void Image<T_Data>::convertImageFormat(
    ImageFormat srcFormat, ImageFormat destFormat,
    const T_Data* srcBuffer, size_t nSrcBufferElements,
    T_Data* destBuffer, size_t nDestBufferElements
) {
    switch (getImageFormatNChannels(srcFormat)) {
        case -1:
        case 0:
            throw std::runtime_error("Invalid image format (bug)");

        case 1:
            convertImageFormat_<1>(srcFormat, destFormat,
                srcBuffer, nSrcBufferElements, destBuffer, nDestBufferElements);
            break;

        case 3:
            convertImageFormat_<3>(srcFormat, destFormat,
                srcBuffer, nSrcBufferElements, destBuffer, nDestBufferElements);
            break;

        case 4:
            convertImageFormat_<4>(srcFormat, destFormat,
                srcBuffer, nSrcBufferElements, destBuffer, nDestBufferElements);
            break;

        default:
            throw std::runtime_error("Requested format conversion not implemented yet");
    }
}

template <typename T_Data>
template <int T_NChannelsSrc>
inline void Image<T_Data>::convertImageFormat_(
    ImageFormat srcFormat, ImageFormat destFormat,
    const T_Data* srcBuffer, size_t nSrcBufferElements,
    T_Data* destBuffer, size_t nDestBufferElements
) {
    switch (getImageFormatNChannels(destFormat)) {
    case -1:
        throw std::runtime_error("Invalid image format (bug)");
    case 0: // UNCHANGED, do nothing
        return;
    case 1: {
        const auto conversionMatrix =
            getImageFormatConversionMatrix<T_Data, T_NChannelsSrc, 1>(srcFormat, destFormat);
        applyFormatConversion<T_NChannelsSrc, 1>(conversionMatrix,
            srcBuffer, nSrcBufferElements, destBuffer, nDestBufferElements);
    }   return;
    case 3: {
        const auto conversionMatrix =
            getImageFormatConversionMatrix<T_Data, T_NChannelsSrc, 3>(srcFormat, destFormat);
        applyFormatConversion<T_NChannelsSrc, 3>(conversionMatrix,
            srcBuffer, nSrcBufferElements, destBuffer, nDestBufferElements);
    }   return;
    case 4: {
        const auto conversionMatrix =
            getImageFormatConversionMatrix<T_Data, T_NChannelsSrc, 4>(srcFormat, destFormat);
        applyFormatConversion<T_NChannelsSrc, 4>(conversionMatrix,
            srcBuffer, nSrcBufferElements, destBuffer, nDestBufferElements);
    }   return;
    default:
        throw std::runtime_error("Requested format conversion not implemented yet");
    }
}

template <typename T_Data>
template<int T_NChannelsSrc, int T_NChannelsDest>
INLINE void Image<T_Data>::applyFormatConversion(
    const Eigen::Matrix<T_Data, T_NChannelsDest, T_NChannelsSrc>& matrix,
    const T_Data* srcBuffer, size_t nSrcBufferElements,
    T_Data* destBuffer, size_t nDestBufferElements
) {
    using SrcPixels = Eigen::Matrix<T_Data, T_NChannelsSrc, Eigen::Dynamic>;
    using SrcPixelsMap = Eigen::Map<const SrcPixels>;
    using DestPixels = Eigen::Matrix<T_Data, T_NChannelsDest, Eigen::Dynamic>;
    using DestPixelsMap = Eigen::Map<DestPixels>;

    SrcPixelsMap srcPixels(srcBuffer, T_NChannelsSrc, nSrcBufferElements / T_NChannelsSrc);
    DestPixelsMap destPixels(destBuffer, T_NChannelsDest, nDestBufferElements / T_NChannelsDest);
    destPixels = matrix * srcPixels;

    // set alpha channel to 1 (this will break if alpha channel is other than the last one)
    if constexpr (T_NChannelsSrc < 4 && T_NChannelsDest == 4) {
        destPixels(Eigen::last, Eigen::all).setOnes();
    }
}


template <typename T_DataSrc, typename T_DataDest>
struct ImageConversionParams {};
template <> struct ImageConversionParams<uint8_t, float> {
    static constexpr bool   prescale    {false};
    static constexpr float  scaleRatio  {1.0f / 255.0f};
};
template <> struct ImageConversionParams<float, uint8_t> {
    static constexpr bool   prescale    {true};
    static constexpr float  scaleRatio  {255.0f};
};


template <typename T_DataSrc, typename T_DataDest>
inline void convertImage(
    const Image<T_DataSrc>& srcImage,
    Image<T_DataDest>& destImage,
    ImageFormat destFormat
) {
    T_DataDest* data = destImage._data;
    std::vector<T_DataDest> tempBuffer;
    if (destImage.usingExternalBuffer()) {
        if (srcImage._width != destImage._width || srcImage._height != destImage._height) {
            throw std::runtime_error("Image width and height need to match when using external buffer");
        }

        if (getImageFormatNChannels(srcImage._format) != getImageFormatNChannels(destImage._format)) {
            // Allocate a temp buffer
            tempBuffer.resize(srcImage._nElements);
            data = tempBuffer.data();
        }
        else
            destImage._format = srcImage.format();
    }
    else {
        destImage.copyParamsFrom(srcImage);
        data = destImage._data;
    }

    using Params = ImageConversionParams<T_DataSrc, T_DataDest>;

    if constexpr (std::is_same_v<T_DataSrc, T_DataDest>) {
        // No conversion required, only copy
        memcpy(data, srcImage._data, srcImage._nElements * sizeof(T_DataSrc));
    }
    else {
        // Perform type conversion if the types differ
        if constexpr (ImageConversionParams<T_DataSrc, T_DataDest>::prescale) {
            // prescale
            for (int i = 0; i < srcImage._nElements; ++i) {
                data[i] = static_cast<T_DataDest>(srcImage._data[i] * Params::scaleRatio);
            }
        } else {
            // postscale
            for (int i = 0; i < srcImage._nElements; ++i) {
                data[i] = static_cast<T_DataDest>(srcImage._data[i]) * Params::scaleRatio;
            }
        }
    }

    // Perform the potential image format conversion
    if (destFormat != ImageFormat::UNCHANGED) {
        if (data == tempBuffer.data()) { // using temp buffer for conversion
            size_t newNElements = destImage._width*destImage._height*getImageFormatNChannels(destFormat);
            destImage.convertImageFormat(srcImage._format, destFormat, data, srcImage._nElements,
                destImage._data, newNElements);
        }
        else {
            destImage.convertImageFormat(destFormat);
        }
    }
}

template <typename T_Data, int T_NChannelsSrc, int T_NChannelsDest>
inline Eigen::Matrix<T_Data, T_NChannelsDest, T_NChannelsSrc> getImageFormatConversionMatrix(
    ImageFormat srcFormat, ImageFormat destFormat
) {
    using ConversionMatrix = Eigen::Matrix<T_Data, T_NChannelsDest, T_NChannelsSrc>;

    // Check that template parameter dimensions match
    if (getImageFormatNChannels(srcFormat) != T_NChannelsSrc ||
        getImageFormatNChannels(destFormat) != T_NChannelsDest) {
        throw std::runtime_error("Image format conversion matrix with invalid dimensions requested");
    }

    // Converting to same format (idk why, do whatever you want), just return the identity matrix
    if (srcFormat == destFormat)
        return ConversionMatrix::Identity();

    if constexpr (T_NChannelsSrc == 4) {
        switch (srcFormat) {
        case ImageFormat::BGRA: {
            switch (destFormat) {
            case ImageFormat::RGB: { // BGRA -> RGB
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  0.0,    0.0,    1.0,    0.0,
                                0.0,    1.0,    0.0,    0.0,
                                1.0,    0.0,    0.0,    0.0;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::YUV: { // BGRA -> YUV
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  0.114,      0.587,      0.299,      0.0,
                                0.436,      -0.28886,   -0.14713,   0.0,
                                -0.10001,   -0.51499,   0.615,      0.0;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::GRAY: { // BGRA -> GRAY
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  0.114,      0.587,      0.299,      0.0;
                    return matrix;
                }();
                return matrix;
            }
            default:
                break;
            }
        }   break;

        default: break;
        }
    }
    else if constexpr (T_NChannelsSrc == 3) {
        switch (srcFormat) {
        case ImageFormat::RGB: {
            switch (destFormat) {
            case ImageFormat::BGRA: { // RGB -> BGRA
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  0.0,    0.0,    1.0,
                                0.0,    1.0,    0.0,
                                1.0,    0.0,    0.0,
                                0.0,    0.0,    0.0;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::YUV: { // RGB -> YUV
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  0.299,      0.587,      0.114,
                                -0.14713,   -0.28886,   0.436,
                                0.615,      -0.51499,   -0.10001;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::GRAY: { // RGB -> GRAY
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  0.299,  0.587,  0.114;
                    return matrix;
                }();
                return matrix;
            }
            default: break;
            }
        }   break;
        case ImageFormat::YUV: {
            switch (destFormat) {
            case ImageFormat::BGRA: { // YUV -> BGRA
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  0.99998,    2.03211,        -1.5082e-05,
                                1.0,        -0.394646,      -0.580594,
                                1.0,        -1.17892e-05,   1.13983,
                                0.0,        0.0,            0.0;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::RGB: { // YUV -> RGB
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  1.0,        -1.17892e-05,   1.13983,
                                1.0,        -0.394646,      -0.580594,
                                0.99998,    2.03211,        -1.5082e-05;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::GRAY: { // YUV -> GRAY
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  1.0,    0.0,    0.0;
                    return matrix;
                }();
                return matrix;
            }
            default: break;
            }
        }   break;

        default: break;
        }
    }
    else if constexpr (T_NChannelsSrc == 1) {
        switch (srcFormat) {
        case ImageFormat::GRAY: {
            switch (destFormat) {
            case ImageFormat::BGRA: { // GRAY -> BGRA
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  1.0,
                                1.0,
                                1.0,
                                0.0;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::RGB: { // GRAY -> RGB
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  1.0,
                                1.0,
                                1.0;
                    return matrix;
                }();
                return matrix;
            }
            case ImageFormat::YUV: { // GRAY -> YUV
                static const auto matrix = [](){
                    ConversionMatrix matrix;
                    matrix  <<  1.0,
                                0.0,
                                0.0;
                    return matrix;
                }();
                return matrix;
            }
            default: break;
            }
        }   break;

        default: break;
        }
    }

    throw std::runtime_error("Requested format conversion not implemented yet");
}

template<typename T_Data>
void writeImageToFile(const Image<T_Data>& image, const std::filesystem::path& filename)
{
    // TODO extend, only 8/8/8 RGB PNG supported for now

    Image<T_Data> img;
    convertImage(image, img, ImageFormat::RGB);
    auto nChannels = getImageFormatNChannels(img.format());
    stbi_write_png(filename.c_str(), img.width(), img.height(), nChannels, img.data(), img.width()*nChannels);
}

template<typename T_Data>
Image<T_Data> readImageFromFile(const std::filesystem::path& filename)
{
    // TODO extend, only 8/8/8 RGB PNG supported for now

    int w, h, c;
    unsigned char* data = stbi_load(filename.c_str(), &w, &h, &c, 3);
    if (data == nullptr)
        throw std::runtime_error("Unable to load image from " + filename.string() + ": " + stbi_failure_reason());
    Image<T_Data> img(w, h, ImageFormat::RGB);
    img.copyFrom(data);
    stbi_image_free(data);

    return img;
}
