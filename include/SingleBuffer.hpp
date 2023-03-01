//
// Project: DooT2
// File: SingleBuffer.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>


template <typename T_Data>
class SingleBuffer {
public:
    // Class for providing read access protected with RAII mechanism
    class ReadHandle {
    public:
        ReadHandle(const ReadHandle&) = delete;
        ReadHandle(ReadHandle&&) noexcept = default;
        ReadHandle& operator=(const ReadHandle&) = delete;
        ReadHandle& operator=(ReadHandle&&) noexcept = default;
        ~ReadHandle();

        const T_Data& operator*() const noexcept;
        const T_Data* operator->() const noexcept;

        friend class SingleBuffer<T_Data>;

    private:
        explicit ReadHandle(SingleBuffer<T_Data>& singleBuffer);

        const T_Data*                   _data;
        std::atomic_bool*               _handleActive;
        std::condition_variable*        _handleCV;
        std::unique_lock<std::mutex>    _lock;
    };

    // Class for providing write access protected with RAII mechanism
    class WriteHandle {
    public:
        WriteHandle(const WriteHandle&) = delete;
        WriteHandle(WriteHandle&&) noexcept = default;
        WriteHandle& operator=(const WriteHandle&) = delete;
        WriteHandle& operator=(WriteHandle&&) noexcept = default;
        ~WriteHandle();

        T_Data& operator*() noexcept;
        T_Data* operator->() noexcept;

        friend class SingleBuffer<T_Data>;

    private:
        explicit WriteHandle(SingleBuffer<T_Data>& singleBuffer);

        T_Data*                         _data;
        std::atomic_bool*               _handleActive;
        std::condition_variable*        _handleCV;
        std::unique_lock<std::mutex>    _lock;
    };

    explicit SingleBuffer(const T_Data& data = T_Data());

    const ReadHandle read();
    WriteHandle write();

    friend class ReadHandle;
    friend class WriteHandle;

private:
    T_Data                  _data;
    
    std::atomic_bool        _handleActive;
    std::mutex              _handleMutex;
    std::condition_variable _handleCV;
};


#include "SingleBuffer.inl"
