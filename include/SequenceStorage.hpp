//
// Project: DooT2
// File: SequenceStorage.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Image.hpp"


class SequenceStorage {
public:
    struct Entry {
        Image<uint8_t>  frame;
    };

    class BatchHandle {
    public:
        Entry& operator[](std::size_t id);
        const Entry& operator[](std::size_t id) const noexcept;

        friend class SequenceStorage;

    private:
        BatchHandle();
        Entry*          _entry;
        const Entry*    _cEntry;
    };

    SequenceStorage(std::size_t batchSize, std::size_t length=0);
    SequenceStorage(const SequenceStorage&) = default;
    SequenceStorage(SequenceStorage&&) = default;
    SequenceStorage& operator=(const SequenceStorage&) = default;
    SequenceStorage& operator=(SequenceStorage&&) = default;

    // Access a batch
    BatchHandle& operator[](std::size_t id);
    const BatchHandle operator[](std::size_t id) const noexcept;

    size_t size() const;

private:
    std::size_t         _batchSize;
    std::vector<Entry>  _data;
    BatchHandle         _batchHandle;
};
