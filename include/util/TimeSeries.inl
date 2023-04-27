//
// Project: DooT2
// File: TimeSeries.inl
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

template<typename T_Entry>
void TimeSeries::addSeries(const std::string& seriesName, const T_Entry& defaultValue)
{
    auto& series = getSeries<T_Entry>(seriesName, defaultValue);
    series.resize(_size);
}

template<typename T_Entry>
void TimeSeries::addEntry(const std::string& seriesName, const T_Entry& entry)
{
    auto& instanceStorage = storage[this];

    if (!instanceStorage.contains(seriesName))
        addSeries(seriesName, entry);

    for (auto& [name, series] : instanceStorage) {
        size_t seriesSize = 0;
        if (name == seriesName) {
            assert(series.getTypeId() == typeId<T_Entry>()); // check the correct type
            seriesSize = series.addEntry(&entry);
        }
        else {
            seriesSize = series.addEntry(); // add default value to all other series except the one specified
        }
        assert(seriesSize == _size+1); // indicates a bug in TimeSeries implementation (some of the series has a differing length)
    }

    ++_size;
}

template<typename... T_NamesAndEntries>
void TimeSeries::addEntries(const T_NamesAndEntries&... namesAndEntries)
{
    static_assert(sizeof...(T_NamesAndEntries) % 2 == 0); // check for even number of arguments
    addEntriesRecursive(namesAndEntries...);

    ++_size;

    // resize all series not specified in the arguments to match the new size
    auto& instanceStorage = storage[this];
    for (auto& [name, series] : instanceStorage) {
        series.resize(_size);
    }
}

template<typename T_Entry>
const std::vector<T_Entry>* TimeSeries::getSeriesVector(const std::string& seriesName) const
{
    auto& instanceStorage = storage[this];
    if (!instanceStorage.contains(seriesName))
        return nullptr; // no such series

    // check the type
    assert(instanceStorage.at(seriesName).getTypeId() == typeId<T_Entry>());

    return &instanceStorage.at(seriesName).getVector<T_Entry>();
}

template<typename T_Entry>
void TimeSeries::fromJson(const nlohmann::json& json)
{
    clear(); // delete all previous data

    for (auto& [name, jsonData] : json.items()) {
        assert(jsonData.is_array());
        auto& series = getSeries<T_Entry>(name, T_Entry());
        // Copy the data directly to avoid overhead from all the function pointer business
        auto& seriesData = *static_cast<std::vector<T_Entry>*>(series._data);
        seriesData = jsonData.template get<std::vector<T_Entry>>();
        if (_size == 0)
            _size = seriesData.size();
        else
            assert(_size == seriesData.size());
    }
}

template<typename T_FirstEntry, typename... T_NamesAndEntries>
void TimeSeries::addEntriesRecursive(
    const std::string& seriesName,
    const T_FirstEntry& firstEntry,
    const T_NamesAndEntries&... entries)
{
    auto& instanceStorage = storage[this];

    if (!instanceStorage.contains(seriesName))
        addSeries(seriesName, firstEntry);

    auto& series = instanceStorage.at(seriesName);
    assert(series.getTypeId() == typeId<T_FirstEntry>()); // check the correct type

    size_t seriesSize = series.addEntry(&firstEntry);
    assert(seriesSize == _size+1); // indicates a bug

    // process rest of the entries
    addEntriesRecursive(entries...);
}


template<typename T_Entry>
size_t TimeSeries::Series::adder(const void* entry)
{
    assert(_typeId == TimeSeries::typeId<T_Entry>());
    auto& data = *static_cast<std::vector<T_Entry>*>(_data);
    if (entry == nullptr) {
        data.push_back(*static_cast<const T_Entry*>(_defaultValue));
    }
    else {
        data.push_back(*static_cast<const T_Entry*>(entry));
    }
    return static_cast<std::vector<T_Entry>*>(_data)->size();
}

template<typename T_Entry>
void TimeSeries::Series::deleter()
{
    assert(_typeId == TimeSeries::typeId<T_Entry>());
    delete static_cast<std::vector<T_Entry>*>(_data);
    delete static_cast<const T_Entry*>(_defaultValue);
}

template<typename T_Entry>
void* TimeSeries::Series::dataCopier() const
{
    assert(_typeId == TimeSeries::typeId<T_Entry>());
    // copy construct a new data vector
    return new std::vector<T_Entry>(*static_cast<std::vector<T_Entry>*>(_data));
}

template<typename T_Entry>
const void* TimeSeries::Series::defaultCopier() const
{
    assert(_typeId == TimeSeries::typeId<T_Entry>());
    // copy construct a new default type
    return new T_Entry(*static_cast<const T_Entry*>(_defaultValue));
}

template<typename T_Entry>
void TimeSeries::Series::resizer(size_t size)
{
    assert(_typeId == TimeSeries::typeId<T_Entry>());
    static_cast<std::vector<T_Entry>*>(_data)->resize(size, *static_cast<const T_Entry*>(_defaultValue));
}

template<typename T_Entry>
void TimeSeries::Series::serializer(const std::string& seriesName, nlohmann::json& json) const
{
    assert(_typeId == TimeSeries::typeId<T_Entry>());
    json[seriesName] = *static_cast<std::vector<T_Entry>*>(_data);
}

template<typename T_Entry>
TimeSeries::Series::Series(const T_Entry& defaultValue) :
    _data           (new std::vector<T_Entry>()),
    _defaultValue   (new T_Entry(defaultValue)),
    _typeId         (TimeSeries::typeId<T_Entry>()),
    _adder          (&TimeSeries::Series::adder<T_Entry>),
    _deleter        (&TimeSeries::Series::deleter<T_Entry>),
    _dataCopier     (&TimeSeries::Series::dataCopier<T_Entry>),
    _defaultCopier  (&TimeSeries::Series::defaultCopier<T_Entry>),
    _resizer        (&TimeSeries::Series::resizer<T_Entry>),
    _serializer     (&TimeSeries::Series::serializer<T_Entry>)
{
}

template<typename T_Entry>
const std::vector<T_Entry>& TimeSeries::Series::getVector()
{
    auto newTypeId = TimeSeries::typeId<T_Entry>();
    assert(_typeId == newTypeId);
    return *static_cast<std::vector<T_Entry>*>(_data);
}

template<typename T_Entry>
TimeSeries::Series& TimeSeries::getSeries(
    const std::string& seriesName, const T_Entry& defaultValue)
{
    // storage dedicated for this instance
    auto& instanceStorage = storage[this];
    // checking of the existing entry needs to be done manually due to the templated
    // constructor of Series
    if (!instanceStorage.contains(seriesName))
        instanceStorage.emplace(std::make_pair(seriesName, defaultValue));
    return instanceStorage.at(seriesName);
}

template<typename T_Entry>
uint32_t TimeSeries::typeId()
{
    static uint32_t typeId {typeIdCounter++};
    return typeId;
}
