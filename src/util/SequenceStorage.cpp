//
// Project: DooT2
// File: SequenceStorage.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "util/SequenceStorage.hpp"


// static members
uint32_t                    SequenceStorage::typeIdCounter {0};
SequenceStorage::Storage   SequenceStorage::storage;


SequenceStorage::SequenceStorage(int64_t batchSize) :
    _size       (0),
    _batchSize  (batchSize)
{
    storage[this];
}

SequenceStorage::~SequenceStorage()
{
    // remove the data for this instance from the storage
    storage.erase(this);
}

SequenceStorage::SequenceStorage(const SequenceStorage& other) :
    _size       (other._size),
    _batchSize  (other._batchSize)
{
    storage[this] = storage[&other];
}

SequenceStorage::SequenceStorage(SequenceStorage&& other) noexcept :
    _size       (other._size),
    _batchSize  (other._batchSize)
{
    other._size = 0;
    other._batchSize = 0;

    // change the storage key (pointer to the host object) from &other to this
    auto data = storage.extract(&other);
    data.key() = this;
    storage.insert(std::move(data));
}

SequenceStorage& SequenceStorage::operator=(const SequenceStorage& other)
{
    if (this == &other)
        return *this;

    _size = other._size;
    _batchSize = other._batchSize;
    storage[this] = storage[&other];

    return *this;
}

SequenceStorage& SequenceStorage::operator=(SequenceStorage&& other) noexcept
{
    if (this == &other)
        return *this;

    _size = other._size;
    _batchSize = other._batchSize;
    other._size = 0;
    other._batchSize = 0;

    // remove the existing data dedicated for this instance
    assert(storage.contains(this));
    storage.erase(this);

    // change the storage key (pointer to the host object) from &other to this
    auto data = storage.extract(&other);
    data.key() = this;
    storage.insert(std::move(data));

    return *this;
}

void SequenceStorage::addEntry(const std::string& sequenceName, const torch::Tensor& entry)
{
    auto& instanceStorage = storage[this];

    if (!instanceStorage.contains(sequenceName))
        throw std::runtime_error("No sequence with name \"" + sequenceName + "\"");

    for (auto& [name, sequence] : instanceStorage) {
        size_t sequenceSize = 0;
        if (name == sequenceName) {
            sequenceSize = sequence.addEntry(entry);
        }
        else {
            sequenceSize = sequence.addEntry(); // add default value to all other sequence except the one specified
        }
        assert(sequenceSize == _size+1); // indicates a bug in SequenceStorage implementation (some of the sequence has a differing length)
    }

    ++_size;
}

void SequenceStorage::resize(size_t newLength)
{
    auto& instanceStorage = storage[this];

    for (auto& [name, sequence] : instanceStorage) {
        sequence.resize(newLength);
    }

    _size = newLength;
}

std::vector<std::string> SequenceStorage::getSequenceNames() const
{
    auto& instanceStorage = storage[this];
    std::vector<std::string> names;
    names.reserve(instanceStorage.size());
    for (auto& [name, series] : instanceStorage)
        names.emplace_back(name);
    return names;
}

size_t SequenceStorage::getNumSequences() const noexcept
{
    auto& instanceStorage = storage[this];
    return instanceStorage.size();
}

size_t SequenceStorage::length() const noexcept
{
    return _size;
}

int64_t SequenceStorage::batchSize() const noexcept
{
    return _batchSize;
}

size_t SequenceStorage::getNumInstances()
{
    return storage.size();
}

void SequenceStorage::addEntriesRecursive()
{
    // dummy function for ending tail recursion
}

SequenceStorage::SequenceWrapper::SequenceWrapper(const SequenceStorage::SequenceWrapper& other) :
    _data               ((other.*other._dataCopier)()), // my favourite syntax
    _defaultValue       ((other.*other._defaultCopier)()),
    _dataTypeId         (other._dataTypeId),
    _defaultValueTypeId (other._defaultValueTypeId),
    _adder              (other._adder),
    _adderTensor        (other._adderTensor),
    _deleter            (other._deleter),
    _dataCopier         (other._dataCopier),
    _defaultCopier      (other._defaultCopier),
    _resizer            (other._resizer)
{
}


SequenceStorage::SequenceWrapper::SequenceWrapper(SequenceStorage::SequenceWrapper&& other) noexcept :
    _data               (other._data),
    _defaultValue       (other._defaultValue),
    _dataTypeId         (other._dataTypeId),
    _defaultValueTypeId (other._defaultValueTypeId),
    _adder              (other._adder),
    _adderTensor        (other._adderTensor),
    _deleter            (other._deleter),
    _dataCopier         (other._dataCopier),
    _defaultCopier      (other._defaultCopier),
    _resizer            (other._resizer)
{
    other._data = nullptr;
    other._defaultValue = nullptr;
    other._deleter = nullptr;
}

SequenceStorage::SequenceWrapper& SequenceStorage::SequenceWrapper::operator=(
    const SequenceStorage::SequenceWrapper& other)
{
    if (this == &other)
        return *this;

    assert(_dataTypeId == other._dataTypeId);
    assert(_defaultValueTypeId == other._defaultValueTypeId);

    // replace the data
    (this->*_deleter)();
    _data = (other.*other._dataCopier)();
    _defaultValue = (other.*other._defaultCopier)();

    return *this;
}

SequenceStorage::SequenceWrapper& SequenceStorage::SequenceWrapper::operator=(
    SequenceStorage::SequenceWrapper&& other) noexcept
{
    if (this == &other)
        return *this;

    assert(_dataTypeId == other._dataTypeId);
    assert(_defaultValueTypeId == other._defaultValueTypeId);

    _data               = other._data;
    _defaultValue       = other._defaultValue;
    _dataTypeId         = other._dataTypeId;
    _defaultValueTypeId = other._defaultValueTypeId;
    _adder              = other._adder;
    _adderTensor        = other._adderTensor;
    _deleter            = other._deleter;
    _dataCopier         = other._dataCopier;
    _defaultCopier      = other._defaultCopier;
    _resizer            = other._resizer;

    other._data = nullptr;
    other._defaultValue = nullptr;
    other._deleter = nullptr;

    return *this;
}

SequenceStorage::SequenceWrapper::~SequenceWrapper()
{
    if (this->_deleter != nullptr)
        (this->*_deleter)();
}

size_t SequenceStorage::SequenceWrapper::addEntry(const void* entry)
{
    return (this->*_adder)(entry);
}

size_t SequenceStorage::SequenceWrapper::addEntry(const torch::Tensor& entry)
{
    return (this->*_adderTensor)(entry);
}

void SequenceStorage::SequenceWrapper::resize(size_t size)
{
    (this->*_resizer)(size);
}

uint32_t SequenceStorage::SequenceWrapper::getTypeId() const noexcept
{
    return _dataTypeId;
}
