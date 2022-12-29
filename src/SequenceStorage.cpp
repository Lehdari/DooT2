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


SequenceStorage::Entry& SequenceStorage::BatchHandle::operator[](std::size_t id)
{
    return _entry[id];
}

const SequenceStorage::Entry& SequenceStorage::BatchHandle::operator[](std::size_t id) const noexcept
{
    return _cEntry[id];
}

SequenceStorage::BatchHandle::BatchHandle() :
    _entry  (nullptr),
    _cEntry (nullptr)
{
}

SequenceStorage::SequenceStorage(std::size_t batchSize, std::size_t length) :
    _batchSize      (batchSize),
    _data           (length*_batchSize)
{
}

SequenceStorage::BatchHandle& SequenceStorage::operator[](std::size_t id)
{
    if (_data.size() < id*_batchSize) {
        _data.resize(id*_batchSize);
    }
    _batchHandle._entry = &_data[id*_batchSize];
    return _batchHandle;
}

const SequenceStorage::BatchHandle SequenceStorage::operator[](std::size_t id) const noexcept
{
    BatchHandle batchHandle;
    batchHandle._cEntry = &_data.at(id*_batchSize);
    return batchHandle;
}

size_t SequenceStorage::size() const
{
    return _data.size() / _batchSize;
}
