//
// Project: DooT2
// File: TimeSeries.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "TimeSeries.hpp"


using Json = nlohmann::json;


// static members
uint32_t            TimeSeries::typeIdCounter   {0};
TimeSeries::Storage TimeSeries::storage;


TimeSeries::TimeSeries() :
    _size   (0)
{
    storage[this];
}

TimeSeries::~TimeSeries()
{
    // remove the data for this instance from the storage
    storage.erase(this);
}

TimeSeries::TimeSeries(const TimeSeries& other) :
    _size   (other._size)
{
    storage[this] = storage[&other];
}

TimeSeries::TimeSeries(TimeSeries&& other) noexcept :
    _size   (other._size)
{
    // change the storage key (pointer to the host object) from &other to this
    auto data = storage.extract(&other);
    data.key() = this;
    storage.insert(std::move(data));
}

TimeSeries& TimeSeries::operator=(const TimeSeries& other)
{
    if (this == &other)
        return *this;

    _size = other._size;
    storage[this] = storage[&other];

    return *this;
}

TimeSeries& TimeSeries::operator=(TimeSeries&& other) noexcept
{
    if (this == &other)
        return *this;

    _size = other._size;
    other._size = 0;

    // remove the existing data dedicated for this instance
    assert(storage.contains(this));
    storage.erase(this);

    // change the storage key (pointer to the host object) from &other to this
    auto data = storage.extract(&other);
    data.key() = this;
    storage.insert(std::move(data));

    return *this;
}

std::vector<std::string> TimeSeries::getSeriesNames() const
{
    auto& instanceStorage = storage[this];
    std::vector<std::string> names;
    names.reserve(instanceStorage.size());
    for (auto& [name, series] : instanceStorage)
        names.emplace_back(name);
    return names;
}

size_t TimeSeries::getNumSeries() const noexcept
{
    auto& instanceStorage = storage[this];
    return instanceStorage.size();
}

size_t TimeSeries::length()
{
    return _size;
}

Json TimeSeries::toJson() const
{
    Json json;

    auto& instanceStorage = storage[this];
    for (auto& [name, series] : instanceStorage) {
        series.serialize(name, json);
    }

    return json;
}

size_t TimeSeries::getNumInstances()
{
    return storage.size();
}

void TimeSeries::addEntriesRecursive()
{
    // dummy function for ending tail recursion
}

TimeSeries::Series::Series(const TimeSeries::Series& other) :
    _data           ((other.*other._dataCopier)()), // my favourite syntax
    _defaultValue   ((other.*other._defaultCopier)()),
    _typeId         (other._typeId),
    _adder          (other._adder),
    _deleter        (other._deleter),
    _dataCopier     (other._dataCopier),
    _defaultCopier  (other._defaultCopier),
    _resizer        (other._resizer),
    _serializer     (other._serializer)
{
}

TimeSeries::Series& TimeSeries::Series::operator=(const TimeSeries::Series& other)
{
    if (this == &other)
        return *this;

    assert(_typeId == other._typeId);

    // replace the data
    (this->*_deleter)();
    _data = (other.*other._dataCopier)();
    _defaultValue = (other.*other._defaultCopier)();

    return *this;
}

TimeSeries::Series::~Series()
{
    (this->*_deleter)();
}

size_t TimeSeries::Series::addEntry(const void* entry)
{
    return (this->*_adder)(entry);
}

void TimeSeries::Series::resize(size_t size)
{
    (this->*_resizer)(size);
}

void TimeSeries::Series::serialize(const std::string& seriesName, nlohmann::json& json) const
{
    (this->*_serializer)(seriesName, json);
}

uint32_t TimeSeries::Series::getTypeId() const noexcept
{
    return _typeId;
}
