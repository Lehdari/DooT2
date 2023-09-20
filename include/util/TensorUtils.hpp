//
// Project: DooT2
// File: TensorUtils.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "util/Utils.hpp"
#include "util/MathUtils.hpp"

#include <array>
#include <vector>
#include <type_traits>
#include <torch/torch.h>


// template utility for scalar type -> torch type enum conversion
template <typename T_ScalarType>
struct ToTorchType {};
template <> struct ToTorchType<float> { static constexpr auto Type = torch::kFloat32; };
template <> struct ToTorchType<double> { static constexpr auto Type = torch::kFloat64; };
template <> struct ToTorchType<int8_t> { static constexpr auto Type = torch::kInt8; };
template <> struct ToTorchType<int16_t> { static constexpr auto Type = torch::kInt16; };
template <> struct ToTorchType<int32_t> { static constexpr auto Type = torch::kInt32; };
template <> struct ToTorchType<int64_t> { static constexpr auto Type = torch::kInt64; };
template <> struct ToTorchType<uint8_t> { static constexpr auto Type = torch::kUInt8; };


// template utility for torch type enum -> scalar type conversion
template <c10::ScalarType T_TorchType>
struct ToScalarType {};
template <> struct ToScalarType<torch::kFloat32> { using Type = float; };
template <> struct ToScalarType<torch::kFloat64> { using Type = double; };
template <> struct ToScalarType<torch::kInt8> { using Type = int8_t; };
template <> struct ToScalarType<torch::kInt16> { using Type = int16_t; };
template <> struct ToScalarType<torch::kInt32> { using Type = int32_t; };
template <> struct ToScalarType<torch::kInt64> { using Type = int64_t; };
template <> struct ToScalarType<torch::kUInt8> { using Type = uint8_t; };

namespace detail {
    template<typename T, typename = void>
    struct has_Type : std::false_type { };

    template<typename T>
    struct has_Type<T, decltype(std::declval<T>().Type, void())> : std::true_type { };
} // namespace detail

template <typename Type>
constexpr bool isTensorScalarType()
{
    return detail::has_Type<ToTorchType<Type>>::value;
}

// Utility functions for copying data from torch tensors to data structures in main memory

// Raw buffer
template<typename T_Data>
INLINE void copyFromTensor(const torch::Tensor& tensor, T_Data* data, std::size_t size)
{
    if (!tensor.dtype().Match<T_Data>())
        throw std::runtime_error("Tensor and buffer data types do not match");

    if (tensor.numel() != size) // sizes need to exactly match to avoid confusion
        throw std::runtime_error("Unequal number of elements in buffer and tensor");

    const auto device = tensor.device();
    if (device != torch::kCPU) {
        // make a CPU copy in case the tensor is on a device
        auto tensorCPU = tensor.to(torch::kCPU);
        memcpy(data, tensorCPU.data_ptr<T_Data>(), size*sizeof(T_Data));
    }
    else {
        memcpy(data, tensor.data_ptr<T_Data>(), size*sizeof(T_Data));
    }
}

// Array
template<typename T_Data, std::size_t N>
INLINE void copyFromTensor(const torch::Tensor& tensor, std::array<T_Data, N>& array)
{
    if (!tensor.dtype().Match<T_Data>())
        throw std::runtime_error("Tensor and array data types do not match");

    if (tensor.numel() != N)
        throw std::runtime_error("Unequal number of elements in array and tensor");

    const auto device = tensor.device();
    if (device != torch::kCPU) {
        // make a CPU copy in case the tensor is on a device
        auto tensorCPU = tensor.to(torch::kCPU);
        memcpy(array.data(), tensorCPU.data_ptr<T_Data>(), N*sizeof(T_Data));
    }
    else {
        memcpy(array.data(), tensor.data_ptr<T_Data>(), N*sizeof(T_Data));
    }
}

// Vector
template<typename T_Data>
INLINE void copyFromTensor(const torch::Tensor& tensor, std::vector<T_Data>& vector)
{
    if (!tensor.dtype().Match<T_Data>())
        throw std::runtime_error("Tensor and vector data types do not match");

    auto size = tensor.numel();
    vector.resize(size); // here we actually have the luxury of setting the vector size

    const auto device = tensor.device();
    if (device != torch::kCPU) {
        // make a CPU copy in case the tensor is on a device
        auto tensorCPU = tensor.to(torch::kCPU);
        memcpy(vector.data(), tensorCPU.data_ptr<T_Data>(), size*sizeof(T_Data));
    }
    else {
        memcpy(vector.data(), tensor.data_ptr<T_Data>(), size*sizeof(T_Data));
    }
}


// Utility functions for copying data from data structures in main memory to torch tensors

// Raw buffer
template<typename T_Data>
INLINE void copyToTensor(const T_Data* data, std::size_t size, torch::Tensor& tensor)
{
    if (!tensor.dtype().Match<T_Data>())
        throw std::runtime_error("Tensor and buffer data types do not match");

    if (tensor.numel() != size)
        throw std::runtime_error("Unequal number of elements in buffer and tensor");

    const auto device = tensor.device();
    if (device != torch::kCPU) {
        // reinitialize the tensor in case it's on a device
        tensor = torch::empty_like(tensor, torch::TensorOptions()
            .dtype(ToTorchType<T_Data>::Type)
            .device(torch::kCPU));
        memcpy(tensor.data_ptr<T_Data>(), data, size*sizeof(T_Data));
        // and then move it back to where it belongs
        tensor = tensor.to(device);
    }
    else {
        memcpy(tensor.data_ptr<T_Data>(), data, size*sizeof(T_Data));
    }
}

// Array
template<typename T_Data, std::size_t N>
INLINE void copyToTensor(const std::array<T_Data, N>& array, torch::Tensor& tensor)
{
    if (!tensor.dtype().Match<T_Data>())
        throw std::runtime_error("Tensor and array data types do not match");

    if (tensor.numel() != N)
        throw std::runtime_error("Unequal number of elements in array and tensor");

    const auto device = tensor.device();
    if (device != torch::kCPU) {
        // reinitialize the tensor in case it's on a device
        tensor = torch::empty_like(tensor, torch::TensorOptions()
            .dtype(ToTorchType<T_Data>::Type)
            .device(torch::kCPU));
        memcpy(tensor.data_ptr<T_Data>(), array.data(), N*sizeof(T_Data));
        // and then move it back to where it belongs
        tensor = tensor.to(device);
    }
    else {
        memcpy(tensor.data_ptr<T_Data>(), array.data(), N*sizeof(T_Data));
    }
}

// Vector
template<typename T_Data>
INLINE void copyToTensor(const std::vector<T_Data>& vector, torch::Tensor& tensor)
{
    if (!tensor.dtype().Match<T_Data>())
        throw std::runtime_error("Tensor and vector data types do not match");

    if (tensor.numel() != vector.size())
        throw std::runtime_error("Unequal number of elements in vector and tensor");

    const auto device = tensor.device();
    if (device != torch::kCPU) {
        // reinitialize the tensor in case it's on a device
        tensor = torch::empty_like(tensor, torch::TensorOptions()
            .dtype(ToTorchType<T_Data>::Type)
            .device(torch::kCPU));
        memcpy(tensor.data_ptr<T_Data>(), vector.data(), vector.size()*sizeof(T_Data));
        // and then move it back to where it belongs
        tensor = tensor.to(device);
    }
    else {
        memcpy(tensor.data_ptr<T_Data>(), vector.data(), vector.size()*sizeof(T_Data));
    }
}


INLINE torch::Tensor normalDistribution(const torch::Tensor& mean, const torch::Tensor& x, double sigma)
{
    constexpr double sqrt2Pi = constexprSqrt(2.0*M_PI);
    return (1.0/(sigma*sqrt2Pi)) * torch::exp(-0.5*((x-mean)/(sigma)).square());
}

INLINE torch::Tensor standardNormalDistribution(const torch::Tensor& x)
{
    constexpr double sqrt2Pi = constexprSqrt(2.0*M_PI);
    return (1.0/sqrt2Pi) * torch::exp(-0.5*x.square());
}

// Positional embedding for vision transformers
torch::Tensor positionalEmbedding2D(int h, int w, int dim, double temperature = 10000.0);
