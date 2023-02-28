//
// Project: DooT2
// File: DoubleBuffer.hpp
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
class DoubleBuffer {
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

        friend class DoubleBuffer<T_Data>;

    private:
        explicit ReadHandle(DoubleBuffer<T_Data>& doubleBuffer);

        const T_Data*                   _data;
        std::atomic_bool*               _readHandleActive;
        std::condition_variable*        _readHandleCV;
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

        friend class DoubleBuffer<T_Data>;

    private:
        explicit WriteHandle(DoubleBuffer<T_Data>& doubleBuffer);

        T_Data*                         _data;
        std::atomic_bool*               _updated;
        std::atomic_bool*               _writeHandleActive;
        std::condition_variable*        _writeHandleCV;
        std::unique_lock<std::mutex>    _lock;
    };

    explicit DoubleBuffer(const T_Data& data = T_Data());

    const ReadHandle read();
    WriteHandle write();

    friend class ReadHandle;
    friend class WriteHandle;

private:
    T_Data                  _data1;
    T_Data                  _data2;
    T_Data*                 _read;
    T_Data*                 _write;
    std::atomic_bool        _updated; // has the buffer been written into?

    std::atomic_bool        _readHandleActive;
    std::mutex              _readHandleMutex;
    std::condition_variable _readHandleCV;

    std::atomic_bool        _writeHandleActive;
    std::mutex              _writeHandleMutex;
    std::condition_variable _writeHandleCV;
};


#include "DoubleBuffer.inl"
