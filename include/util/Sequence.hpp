//
// Project: DooT2
// File: Sequence.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "util/TensorUtils.hpp"

#include <vector>
#include <cstdint>


template <typename T_Data>
class Sequence {
public:
    // Interface for single entry access
    struct EntryHandle {
        EntryHandle(T_Data* data, const std::vector<int64_t>* entryShape);

        EntryHandle& operator=(const T_Data& rhs);
        EntryHandle& operator=(const torch::Tensor& rhs);

        operator torch::Tensor();

    private:
        T_Data*                     _ptr;
        const std::vector<int64_t>* _entryShape;
    };

    // Interface for batch access
    struct BatchHandle {
        BatchHandle(
            T_Data* data,
            const std::vector<int64_t>* batchShape,
            const std::vector<int64_t>* entryShape,
            int64_t entryElements);

        operator torch::Tensor();

        // Single entry access (id: sequence index in the batch)
        EntryHandle operator[](size_t id) const;

    private:
        T_Data*                     _ptr;
        const std::vector<int64_t>* _batchShape;
        const std::vector<int64_t>* _entryShape;
        int64_t                     _entryElements;
    };

    // TODO const handles

    // Custom iterator types for circular buffer functionality
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = T_Data;
        using pointer           = T_Data*;
        using reference         = T_Data&;

        Iterator(
            std::vector<T_Data>* parentVector,
            int64_t batchElements,
            const std::vector<int64_t>* entryShape,
            int64_t entryElements,
            size_t pos);

        reference operator*() const;
        pointer operator->() const;

        Iterator& operator++();
        Iterator operator++(int);

        // Single entry access (id: sequence index in a batch)
        EntryHandle operator[](size_t id) const;

        friend bool operator== (const Iterator& a, const Iterator& b);
        friend bool operator!= (const Iterator& a, const Iterator& b);

    private:
        std::vector<T_Data>*        _parentVector;
        int64_t                     _batchElements;
        const std::vector<int64_t>* _entryShape;
        int64_t                     _entryElements;
        size_t                      _pos;           // time step
    };

    // TODO const iterator

    Sequence(int64_t batchSize, std::initializer_list<int64_t> entryShape);
    Sequence(int64_t batchSize, const std::vector<int64_t>& entryShape);

    const std::vector<int64_t>& entryShape() const noexcept;
    const std::vector<T_Data>& buffer() const noexcept;

    // Map the sequence contents to a tensor of shape {size, batchSize, entryShape}
    const torch::Tensor& tensor() const;

    // Add a batch to the sequence
    // Required tensor shape: {batchSize, entryShape}
    template <typename T_Batch>
    void addBatch(const T_Batch& batch);

    void resize(size_t newLength, const T_Data& value = T_Data());

    template <typename T_Batch>
    void resize(size_t newLength, const T_Batch& batch = T_Batch());

    Iterator begin();
    Iterator end();
//    ConstIterator cbegin() const;
//    ConstIterator cend() const;

    // Get batch at step t
    BatchHandle operator[](size_t t) noexcept;

    // Number of timesteps
    size_t length() const noexcept;

private:
    int64_t                 _batchSize;
    size_t                  _length;        // number of timesteps
    std::vector<int64_t>    _entryShape;    // shape of a single sequence entry tensor
    int64_t                 _entryElements; // _entryShape values multiplied, number of elements in an entry
    std::vector<int64_t>    _batchShape;    // shape of a batch: {batchSize, entryShape}
    int64_t                 _batchElements; // _batchShape values multiplied, number of elements in a batch
    std::vector<int64_t>    _bufferShape;   // logical shape of the entire buffer: {size, batchSize, entryShape}
    std::vector<T_Data>     _buffer;        // sequential data buffer
    torch::Tensor           _tensor;        // tensor that maps the data buffer above

    static inline int64_t multiplyVectorValues(const std::vector<int64_t>& vector);

    inline void setBatch(size_t t, const torch::Tensor& tensor);
    inline void setBatch(size_t t, const T_Data& data);
    inline void updateTensor(); // for updating the mapping of _tensor member
};


// TODO move the function definitions to an .inl file

template <typename T_Data>
Sequence<T_Data>::EntryHandle::EntryHandle(T_Data* data, const std::vector<int64_t>* entryShape) :
    _ptr        (data),
    _entryShape (entryShape)
{
}

template <typename T_Data>
Sequence<T_Data>::EntryHandle& Sequence<T_Data>::EntryHandle::operator=(const T_Data& rhs)
{
    assert(_ptr != nullptr); // BUG
    *_ptr = rhs;
    return *this;
}

template<typename T_Data>
Sequence<T_Data>::EntryHandle& Sequence<T_Data>::EntryHandle::operator=(const torch::Tensor& rhs)
{
    assert(_ptr != nullptr); // BUG
    torch::from_blob(
        _ptr, *_entryShape,
        torch::TensorOptions().dtype<T_Data>().device(torch::kCPU)
    ) = rhs;
    return *this;
}

template <typename T_Data>
Sequence<T_Data>::EntryHandle::operator torch::Tensor()
{
    return torch::from_blob(
        _ptr, *_entryShape,
        torch::TensorOptions().dtype<T_Data>().device(torch::kCPU)
    );
}

template <typename T_Data>
Sequence<T_Data>::BatchHandle::BatchHandle(
    T_Data* data,
    const std::vector<int64_t>* batchShape,
    const std::vector<int64_t>* entryShape,
    int64_t entryElements
) :
    _ptr            (data),
    _batchShape     (batchShape),
    _entryShape     (entryShape),
    _entryElements  (entryElements)
{
}

template <typename T_Data>
Sequence<T_Data>::BatchHandle::operator torch::Tensor()
{
    return torch::from_blob(
        _ptr, *_batchShape,
        torch::TensorOptions().dtype<T_Data>().device(torch::kCPU)
    );
}

template <typename T_Data>
Sequence<T_Data>::EntryHandle Sequence<T_Data>::BatchHandle::operator[](size_t id) const
{
    return Sequence<T_Data>::EntryHandle(_ptr+_entryElements*id, _entryShape);
}

template <typename T_Data>
Sequence<T_Data>::Iterator::Iterator(
    std::vector<T_Data>* parentVector,
    int64_t batchElements,
    const std::vector<int64_t>* entryShape,
    int64_t entryElements,
    size_t pos
) :
    _parentVector   (parentVector),
    _batchElements  (batchElements),
    _entryShape     (entryShape),
    _entryElements  (entryElements),
    _pos            (pos)
{
}

template <typename T_Data>
Sequence<T_Data>::Iterator::reference Sequence<T_Data>::Iterator::operator*() const
{
    return *(_parentVector->data() + _pos*_batchElements);
}

template <typename T_Data>
Sequence<T_Data>::Iterator::pointer Sequence<T_Data>::Iterator::operator->() const
{
    return _parentVector->data() + _pos*_batchElements;
}

template <typename T_Data>
Sequence<T_Data>::Iterator& Sequence<T_Data>::Iterator::operator++()
{
    ++_pos;
    return *this;
}

template <typename T_Data>
Sequence<T_Data>::Iterator Sequence<T_Data>::Iterator::operator++(int)
{
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

template <typename T_Data>
Sequence<T_Data>::EntryHandle Sequence<T_Data>::Iterator::operator[](size_t id) const
{
    return Sequence<T_Data>::EntryHandle(this->operator->()+_entryElements*id, _entryShape);
}

template <typename T_Data>
bool operator==(const typename Sequence<T_Data>::Iterator& a, const typename Sequence<T_Data>::Iterator& b)
{
    return a._parentVector == b._parentVector && a._pos == b._pos;
}

template <typename T_Data>
bool operator!=(const typename Sequence<T_Data>::Iterator& a, const typename Sequence<T_Data>::Iterator& b)
{
    return !(a==b);
}


template <typename T_Data>
Sequence<T_Data>::Sequence(
    int64_t batchSize,
    std::initializer_list<int64_t> entryShape
) :
    _batchSize      (batchSize),
    _length         (0),
    _entryShape     (entryShape),
    _entryElements  (multiplyVectorValues(_entryShape))
{
    // construct the _batchShape and _bufferShape
    _batchShape.reserve(_entryShape.size()+1);
    _bufferShape.reserve(_entryShape.size()+2);
    _bufferShape.emplace_back(_length);
    _batchShape.emplace_back(_batchSize);
    _bufferShape.emplace_back(_batchSize);
    for (const auto& d : _entryShape) {
        _batchShape.emplace_back(d);
        _bufferShape.emplace_back(d);
    }

    _batchElements = multiplyVectorValues(_batchShape);
}

template <typename T_Data>
Sequence<T_Data>::Sequence(
    int64_t batchSize,
    const std::vector<int64_t>& entryShape
) :
    _batchSize      (batchSize),
    _length         (0),
    _entryShape     (entryShape),
    _entryElements  (multiplyVectorValues(_entryShape))
{
    // construct the _batchShape and _bufferShape
    _batchShape.reserve(_entryShape.size()+1);
    _bufferShape.reserve(_entryShape.size()+2);
    _bufferShape.emplace_back(_length);
    _batchShape.emplace_back(_batchSize);
    _bufferShape.emplace_back(_batchSize);
    for (const auto& d : _entryShape) {
        _batchShape.emplace_back(d);
        _bufferShape.emplace_back(d);
    }

    _batchElements = multiplyVectorValues(_batchShape);
}

template <typename T_Data>
const std::vector<int64_t>& Sequence<T_Data>::entryShape() const noexcept
{
    return _entryShape;
}

template <typename T_Data>
const std::vector<T_Data>& Sequence<T_Data>::buffer() const noexcept
{
    return _buffer;
}

template <typename T_Data>
const torch::Tensor& Sequence<T_Data>::tensor() const
{
    return _tensor;
}

template <typename T_Data>
template <typename T_Batch>
void Sequence<T_Data>::addBatch(const T_Batch& batch)
{
    auto pos = multiplyVectorValues(_bufferShape);
    _bufferShape[0] = ++_length;
    _buffer.resize(multiplyVectorValues(_bufferShape));
    assert(std::distance(_buffer.begin()+pos, _buffer.end()) == _batchElements); // BUG!
    setBatch(_length-1, batch);
    updateTensor();
}

template <typename T_Data>
void Sequence<T_Data>::resize(size_t newLength, const T_Data& value)
{
    _bufferShape[0] = (int64_t)newLength;
    _buffer.resize(multiplyVectorValues(_bufferShape));

    if (newLength > _length) {
        for (int t=_length; t<newLength; ++t) {
            setBatch(t, value);
        }
    }

    _length = newLength;
    updateTensor();
}

template <typename T_Data>
template <typename T_Batch>
void Sequence<T_Data>::resize(size_t newLength, const T_Batch& batch)
{
    _bufferShape[0] = (int64_t)newLength;
    _buffer.resize(multiplyVectorValues(_bufferShape));

    if (newLength > _length) {
        for (int t=_length; t<newLength; ++t) {
            setBatch(t, batch);
        }
    }

    _length = newLength;
    updateTensor();
}

template <typename T_Data>
Sequence<T_Data>::Iterator Sequence<T_Data>::begin()
{
    return Iterator(&_buffer, _batchElements, &_entryShape, _entryElements, 0);
}

template <typename T_Data>
Sequence<T_Data>::Iterator Sequence<T_Data>::end()
{
    return nullptr;
}
#if 0
template <typename T_Data>
Sequence<T_Data>::ConstIterator Sequence<T_Data>::cbegin() const
{
    return nullptr;
}

template <typename T_Data>
Sequence<T_Data>::ConstIterator Sequence<T_Data>::cend() const
{
    return nullptr;
}
#endif

template <typename T_Data>
Sequence<T_Data>::BatchHandle Sequence<T_Data>::operator[](size_t t) noexcept
{
    return Sequence::BatchHandle(_buffer.data() + t*_batchElements, &_batchShape, &_entryShape, _entryElements);
}

template <typename T_Data>
size_t Sequence<T_Data>::length() const noexcept
{
    return _length;
}

template <typename T_Data>
int64_t Sequence<T_Data>::multiplyVectorValues(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    int64_t m = 1;
    for (const auto& v : vector)
        m *= v;

    return m;
}

template <typename T_Data>
void Sequence<T_Data>::setBatch(size_t t, const torch::Tensor& tensor)
{
    assert(tensor.scalar_type() == ToTorchType<T_Data>::Type);

    // Tensor shape can be that of full batch or individual entry
    if (tensor.sizes() == _batchShape) {
        auto* tensorData = tensor.data_ptr<T_Data>();
        for (int i = 0; i < _batchElements; ++i) {
            _buffer[t * _batchElements + i] = tensorData[i];
        }
    }
    else if (tensor.sizes() == _entryShape) {
        auto* tensorData = tensor.data_ptr<T_Data>();
        for (int j=0; j<_batchSize; ++j) {
            for (int i=0; i<_entryElements; ++i) {
                _buffer[t*_batchElements + j*_entryElements + i] = tensorData[i];
            }
        }
    }
    else
        throw std::runtime_error("Invalid tensor shape");
}

template <typename T_Data>
void Sequence<T_Data>::setBatch(size_t t, const T_Data& data)
{
    std::fill(_buffer.begin()+t*_batchElements, _buffer.end(), data);
}

template <typename T_Data>
void Sequence<T_Data>::updateTensor()
{
    if constexpr (isTensorScalarType<T_Data>()) {
        _tensor = torch::from_blob(
            _buffer.data(), _bufferShape,
            torch::TensorOptions().dtype<T_Data>().device(torch::kCPU)
        );
    }
}
