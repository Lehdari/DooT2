//
// Project: DooT2
// File: SequenceStorage.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "SequenceStorage.hpp"


SequenceStorage::Entry& SequenceStorage::BatchPointer::operator[](std::size_t id)
{
    return _entry[id];
}

SequenceStorage::BatchPointer::BatchPointer() :
    _entry  (nullptr)
{
}

SequenceStorage::SequenceStorage(std::size_t batchSize, std::size_t length) :
    _batchSize      (batchSize),
    _data           (length*_batchSize)
{
}

SequenceStorage::BatchPointer& SequenceStorage::operator[](std::size_t id)
{
    if (_data.size() < id*_batchSize) {
        _data.resize(id*_batchSize);
    }
    _batchPointer._entry = &_data[id*_batchSize];
    return _batchPointer;
}
