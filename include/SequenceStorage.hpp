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
        Image<uint8_t>  bgraFrame;
    };

    class BatchPointer {
    public:
        Entry& operator[](std::size_t id);

        friend class SequenceStorage;

    private:
        BatchPointer();
        Entry*  _entry;
    };

    SequenceStorage(std::size_t batchSize, std::size_t length=0);

    BatchPointer& operator[](std::size_t id);

private:
    std::size_t         _batchSize;
    std::vector<Entry>  _data;
    BatchPointer        _batchPointer;
};
